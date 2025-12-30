"""API route modules."""

from .ingest import router as ingest_router
from .knowledge import router as knowledge_router
from .copilot import router as copilot_router
from .workspace import router as workspace_router
from .mcp import router as mcp_router
from .a2a import router as a2a_router

__all__ = [
    "ingest_router",
    "knowledge_router",
    "copilot_router",
    "workspace_router",
    "mcp_router",
    "a2a_router",
]
