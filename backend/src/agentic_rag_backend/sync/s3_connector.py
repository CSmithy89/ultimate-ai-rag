"""AWS S3 sync connector.

Story 20-H3: Implement External Data Source Sync

This module provides a connector for syncing documents from AWS S3 buckets.
"""

import asyncio
from datetime import datetime
from typing import AsyncIterator, Optional

import structlog

from .base import BaseConnector
from .models import SyncConfig, SyncContent, SyncItem, SyncSourceType

logger = structlog.get_logger(__name__)

# Supported document types for text extraction
SUPPORTED_EXTENSIONS = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".csv": "text/csv",
    ".html": "text/html",
    ".htm": "text/html",
    ".xml": "application/xml",
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


class S3Connector(BaseConnector):
    """Sync connector for AWS S3 buckets.

    Syncs documents from S3 buckets, supporting incremental sync
    via ETags and filtering by prefix.

    Example:
        config = SyncConfig(
            source_type=SyncSourceType.S3,
            credentials={"aws_access_key_id": "xxx", "aws_secret_access_key": "xxx"},
            settings={"bucket": "my-docs", "prefix": "documents/"},
        )
        connector = S3Connector(config)
        result = await connector.sync()
    """

    def __init__(self, config: SyncConfig) -> None:
        """Initialize S3 connector.

        Args:
            config: Configuration with S3 settings
        """
        super().__init__(config)
        self._client = None

    @property
    def source_type(self) -> SyncSourceType:
        """Return S3 source type."""
        return SyncSourceType.S3

    @property
    def bucket(self) -> str:
        """Return the configured bucket name."""
        return self._config.settings.get("bucket", "")

    @property
    def prefix(self) -> str:
        """Return the configured prefix filter."""
        return self._config.settings.get("prefix", "")

    def _get_client(self):
        """Get or create the S3 client."""
        if self._client is None:
            try:
                import boto3

                credentials = self._config.credentials
                self._client = boto3.client(
                    "s3",
                    aws_access_key_id=credentials.get("aws_access_key_id"),
                    aws_secret_access_key=credentials.get("aws_secret_access_key"),
                    region_name=credentials.get("region_name", "us-east-1"),
                )
            except ImportError as e:
                raise ImportError(
                    "boto3 is required for S3 sync. "
                    "Install with: pip install boto3"
                ) from e

        return self._client

    async def list_items(
        self,
        max_items: Optional[int] = None,
        incremental: bool = True,
    ) -> AsyncIterator[SyncItem]:
        """List objects in the S3 bucket.

        Args:
            max_items: Maximum number of items to list
            incremental: If True, only list items changed since last sync

        Yields:
            SyncItem for each object in the bucket
        """
        client = self._get_client()

        if not self.bucket:
            self._logger.error("s3_no_bucket_configured")
            return

        paginator = client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(
            Bucket=self.bucket,
            Prefix=self.prefix,
        )

        items_yielded = 0

        for page in page_iterator:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]

                # Skip directories
                if key.endswith("/"):
                    continue

                # Check if file type is supported
                ext = self._get_extension(key)
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                item = SyncItem(
                    id=key,
                    source_type=SyncSourceType.S3,
                    name=key.split("/")[-1],
                    path=key,
                    content_type=SUPPORTED_EXTENSIONS.get(ext, "application/octet-stream"),
                    size_bytes=obj.get("Size"),
                    last_modified=obj.get("LastModified"),
                    etag=obj.get("ETag", "").strip('"'),
                    metadata={
                        "bucket": self.bucket,
                        "storage_class": obj.get("StorageClass"),
                    },
                )

                # Check if item needs sync (incremental mode)
                if incremental and not self._should_sync_item(item):
                    continue

                yield item
                items_yielded += 1

                if max_items and items_yielded >= max_items:
                    return

            # Run event loop to avoid blocking
            await asyncio.sleep(0)

    async def fetch_content(self, item: SyncItem) -> SyncContent:
        """Fetch content from S3.

        Args:
            item: The S3 object to fetch

        Returns:
            SyncContent with the object content
        """
        client = self._get_client()

        try:
            response = client.get_object(
                Bucket=self.bucket,
                Key=item.path,
            )

            content = response["Body"].read()

            # Extract text based on content type
            text = self._extract_text(content, item.content_type)

            self._update_state_for_item(item)

            return SyncContent(
                item=item,
                content=content,
                text=text,
                metadata={
                    "content_length": response.get("ContentLength"),
                    "content_encoding": response.get("ContentEncoding"),
                    "last_modified": response.get("LastModified"),
                },
            )
        except Exception as e:
            self._logger.error(
                "s3_fetch_failed",
                key=item.path,
                error=str(e),
            )
            raise

    @staticmethod
    def _get_extension(key: str) -> str:
        """Get file extension from S3 key."""
        if "." in key:
            return "." + key.rsplit(".", 1)[-1].lower()
        return ""

    def _extract_text(self, content: bytes, content_type: str) -> str:
        """Extract text from content based on type.

        Args:
            content: Raw content bytes
            content_type: MIME type of content

        Returns:
            Extracted text
        """
        try:
            if content_type in ("text/plain", "text/markdown", "text/csv"):
                return content.decode("utf-8")

            if content_type == "application/json":
                import json
                data = json.loads(content.decode("utf-8"))
                return json.dumps(data, indent=2)

            if content_type in ("application/yaml", "application/x-yaml"):
                return content.decode("utf-8")

            if content_type in ("text/html", "text/htm"):
                # Basic HTML text extraction
                text = content.decode("utf-8")
                # Remove script and style tags
                import re
                text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
                text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
                # Remove HTML tags
                text = re.sub(r"<[^>]+>", " ", text)
                # Normalize whitespace
                text = re.sub(r"\s+", " ", text).strip()
                return text

            if content_type == "application/xml":
                return content.decode("utf-8")

            # For binary formats (PDF, DOC, DOCX), return empty
            # Full implementation would use document processing libraries
            self._logger.debug(
                "s3_binary_content",
                content_type=content_type,
                note="Binary content extraction not implemented",
            )
            return ""

        except Exception as e:
            self._logger.warning(
                "s3_text_extraction_failed",
                content_type=content_type,
                error=str(e),
            )
            return ""

    async def validate_connection(self) -> bool:
        """Validate S3 connection and bucket access.

        Returns:
            True if connection is valid
        """
        try:
            client = self._get_client()
            client.head_bucket(Bucket=self.bucket)
            self._logger.info("s3_connection_validated", bucket=self.bucket)
            return True
        except Exception as e:
            self._logger.warning(
                "s3_connection_failed",
                bucket=self.bucket,
                error=str(e),
            )
            return False
