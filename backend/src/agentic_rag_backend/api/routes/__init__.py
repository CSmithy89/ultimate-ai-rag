"""API route modules."""

from .ingest import router as ingest_router
from .knowledge import router as knowledge_router
from .copilot import router as copilot_router
from .workspace import router as workspace_router
from .mcp import router as mcp_router
from .a2a import router as a2a_router
from .ag_ui import router as ag_ui_router
from .ops import router as ops_router
from .codebase import router as codebase_router
from .memories import router as memories_router
from .communities import router as communities_router
from .lazy_rag import router as lazy_rag_router
from .query_router import router as query_router
from .dual_level import router as dual_level_router

__all__ = [
    "ingest_router",
    "knowledge_router",
    "copilot_router",
    "workspace_router",
    "mcp_router",
    "a2a_router",
    "ag_ui_router",
    "ops_router",
    "codebase_router",
    "memories_router",
    "communities_router",
    "lazy_rag_router",
    "query_router",
    "dual_level_router",
]
