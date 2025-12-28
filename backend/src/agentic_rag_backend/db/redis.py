"""Redis client with Streams support for job queue management."""

import json
from datetime import datetime
from typing import Any, AsyncGenerator, Optional
from uuid import UUID

import redis.asyncio as redis
import structlog

from agentic_rag_backend.core.errors import RedisError

logger = structlog.get_logger(__name__)

# Stream names for job queue
CRAWL_JOBS_STREAM = "crawl.jobs"
PARSE_JOBS_STREAM = "parse.jobs"
INDEX_JOBS_STREAM = "index.jobs"

# Consumer group names
CRAWL_CONSUMER_GROUP = "crawl-workers"
PARSE_CONSUMER_GROUP = "parse-workers"
INDEX_CONSUMER_GROUP = "index-workers"


def _serialize_value(value: Any) -> str:
    """Serialize a value for Redis storage."""
    if isinstance(value, (UUID, datetime)):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def _deserialize_message(data: dict[bytes, bytes]) -> dict[str, Any]:
    """Deserialize a Redis message to a dictionary."""
    result = {}
    for key, value in data.items():
        key_str = key.decode("utf-8") if isinstance(key, bytes) else key
        value_str = value.decode("utf-8") if isinstance(value, bytes) else value
        # Try to parse JSON values
        try:
            result[key_str] = json.loads(value_str)
        except (json.JSONDecodeError, TypeError):
            result[key_str] = value_str
    return result


class RedisClient:
    """
    Redis client with Streams support for async job processing.

    Implements producer/consumer patterns for the ingestion pipeline:
    - API Request -> Redis Stream (crawl.jobs) -> Crawler Worker
    - Crawler Worker -> Redis Stream (parse.jobs) -> Parse Worker
    - Parse Worker -> Redis Stream (index.jobs) -> Index Worker
    """

    def __init__(self, url: str) -> None:
        """
        Initialize Redis client.

        Args:
            url: Redis connection URL (e.g., redis://localhost:6379)
        """
        self.url = url
        self._client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Establish connection to Redis."""
        if self._client is None:
            self._client = redis.from_url(
                self.url,
                encoding="utf-8",
                decode_responses=False,  # We handle decoding ourselves
            )
            logger.info("redis_connected", url=self.url)

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("redis_disconnected")

    @property
    def client(self) -> redis.Redis:
        """Get the Redis client, raising error if not connected."""
        if self._client is None:
            raise RedisError("connection", "Redis client not connected")
        return self._client

    async def publish_job(
        self,
        stream: str,
        job_data: dict[str, Any],
    ) -> str:
        """
        Publish a job to a Redis Stream.

        Args:
            stream: Name of the stream (e.g., 'crawl.jobs')
            job_data: Job data dictionary

        Returns:
            Message ID assigned by Redis

        Raises:
            RedisError: If publish fails
        """
        try:
            # Serialize all values for Redis
            serialized = {k: _serialize_value(v) for k, v in job_data.items()}
            message_id = await self.client.xadd(stream, serialized)
            logger.info(
                "job_published",
                stream=stream,
                message_id=message_id.decode("utf-8") if isinstance(message_id, bytes) else message_id,
                job_id=job_data.get("job_id"),
            )
            return message_id.decode("utf-8") if isinstance(message_id, bytes) else message_id
        except redis.RedisError as e:
            raise RedisError("publish_job", str(e)) from e

    async def ensure_consumer_group(
        self,
        stream: str,
        group: str,
    ) -> None:
        """
        Ensure a consumer group exists for a stream.

        Creates the group if it doesn't exist. Also creates the stream
        if it doesn't exist (mkstream=True).

        Args:
            stream: Stream name
            group: Consumer group name
        """
        try:
            await self.client.xgroup_create(
                stream,
                group,
                id="0",
                mkstream=True,
            )
            logger.info("consumer_group_created", stream=stream, group=group)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise RedisError("ensure_consumer_group", str(e)) from e
            # Group already exists, which is fine

    async def consume_jobs(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 1,
        block_ms: int = 5000,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Consume jobs from a Redis Stream using a consumer group.

        This is an async generator that yields jobs as they become available.
        Jobs are automatically acknowledged after being yielded.

        Args:
            stream: Stream name to consume from
            group: Consumer group name
            consumer: Consumer name (unique identifier for this worker)
            count: Number of messages to fetch at once
            block_ms: Milliseconds to block waiting for messages

        Yields:
            Job data dictionaries with 'message_id' field added

        Raises:
            RedisError: If consumption fails
        """
        await self.ensure_consumer_group(stream, group)

        while True:
            try:
                messages = await self.client.xreadgroup(
                    groupname=group,
                    consumername=consumer,
                    streams={stream: ">"},
                    count=count,
                    block=block_ms,
                )

                if not messages:
                    continue

                for stream_name, entries in messages:
                    for message_id, data in entries:
                        message_id_str = (
                            message_id.decode("utf-8")
                            if isinstance(message_id, bytes)
                            else message_id
                        )
                        job_data = _deserialize_message(data)
                        job_data["message_id"] = message_id_str

                        logger.info(
                            "job_received",
                            stream=stream,
                            message_id=message_id_str,
                            job_id=job_data.get("job_id"),
                        )

                        yield job_data

                        # Acknowledge the message after successful processing
                        await self.client.xack(stream, group, message_id)
                        logger.debug(
                            "job_acknowledged",
                            stream=stream,
                            message_id=message_id_str,
                        )

            except redis.RedisError as e:
                logger.error("consume_error", stream=stream, error=str(e))
                raise RedisError("consume_jobs", str(e)) from e

    async def get_pending_count(self, stream: str, group: str) -> int:
        """
        Get the count of pending messages in a consumer group.

        Args:
            stream: Stream name
            group: Consumer group name

        Returns:
            Number of pending messages
        """
        try:
            info = await self.client.xpending(stream, group)
            return info["pending"] if info else 0
        except redis.RedisError:
            return 0

    async def get_stream_length(self, stream: str) -> int:
        """
        Get the total length of a stream.

        Args:
            stream: Stream name

        Returns:
            Number of messages in the stream
        """
        try:
            return await self.client.xlen(stream)
        except redis.RedisError:
            return 0


# Global Redis client instance
_redis_client: Optional[RedisClient] = None


async def get_redis_client(url: Optional[str] = None) -> RedisClient:
    """
    Get or create the global Redis client instance.

    Args:
        url: Redis connection URL. Required on first call.

    Returns:
        RedisClient instance
    """
    global _redis_client
    if _redis_client is None:
        if url is None:
            raise RedisError("init", "Redis URL required for first initialization")
        _redis_client = RedisClient(url)
        await _redis_client.connect()
    return _redis_client


async def close_redis_client() -> None:
    """Close the global Redis client connection."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.disconnect()
        _redis_client = None
