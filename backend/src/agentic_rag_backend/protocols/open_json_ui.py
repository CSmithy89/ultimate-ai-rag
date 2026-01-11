"""
Open-JSON-UI Pydantic Models

Backend models for generating Open-JSON-UI payloads that agents can return
to render declarative UI components in the frontend.

Story 22-C2: Implement Open-JSON-UI Renderer

Version: 1.0.0-internal

Example:
    >>> from agentic_rag_backend.protocols.open_json_ui import (
    ...     create_open_json_ui,
    ...     OpenJSONUIText,
    ...     OpenJSONUIButton,
    ... )
    >>> payload = create_open_json_ui([
    ...     OpenJSONUIText(content="Hello world"),
    ...     OpenJSONUIButton(label="Click me", action="click_action"),
    ... ])
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class OpenJSONUIText(BaseModel):
    """Text component for displaying styled text content."""

    type: Literal["text"] = "text"
    content: str = Field(..., description="Text content to display")
    style: str | None = Field(
        None,
        description="Text style variant: normal, muted, error, success",
    )


class OpenJSONUIHeading(BaseModel):
    """Heading component for section headers (h1-h6)."""

    type: Literal["heading"] = "heading"
    level: Annotated[int, Field(ge=1, le=6, description="Heading level 1-6")]
    content: str = Field(..., description="Heading text content")


class OpenJSONUICode(BaseModel):
    """Code component for syntax-highlighted code blocks."""

    type: Literal["code"] = "code"
    content: str = Field(..., description="Code content")
    language: str | None = Field(
        None, description="Programming language for syntax highlighting"
    )


class OpenJSONUITable(BaseModel):
    """Table component for tabular data display."""

    type: Literal["table"] = "table"
    headers: list[str] = Field(..., description="Column headers")
    rows: list[list[str]] = Field(..., description="Table rows (array of cell values)")
    caption: str | None = Field(None, description="Optional table caption")


class OpenJSONUIImage(BaseModel):
    """Image component for displaying images with alt text."""

    type: Literal["image"] = "image"
    src: str = Field(..., description="Image source URL (must be http/https)")
    alt: str = Field(..., description="Alt text for accessibility")
    width: int | None = Field(None, description="Optional image width")
    height: int | None = Field(None, description="Optional image height")


class OpenJSONUIButton(BaseModel):
    """Button component for interactive actions."""

    type: Literal["button"] = "button"
    label: str = Field(..., description="Button label text")
    action: str = Field(..., description="Action identifier sent to callback")
    variant: str | None = Field(
        None,
        description="Button variant: default, destructive, outline, ghost, secondary",
    )


class OpenJSONUIList(BaseModel):
    """List component for ordered/unordered lists."""

    type: Literal["list"] = "list"
    items: list[str] = Field(..., description="List item contents")
    ordered: bool = Field(False, description="Whether the list is ordered (numbered)")


class OpenJSONUILink(BaseModel):
    """Link component for hyperlinks."""

    type: Literal["link"] = "link"
    text: str = Field(..., description="Link text")
    href: str = Field(..., description="Link URL (must be http/https)")
    target: str | None = Field(
        None, description="Link target: _self or _blank"
    )


class OpenJSONUIDivider(BaseModel):
    """Divider component for visual separation."""

    type: Literal["divider"] = "divider"


class OpenJSONUIProgress(BaseModel):
    """Progress component for progress bars."""

    type: Literal["progress"] = "progress"
    value: Annotated[int, Field(ge=0, le=100, description="Progress value 0-100")]
    label: str | None = Field(None, description="Optional label for the progress bar")


class OpenJSONUIAlert(BaseModel):
    """Alert component for notifications and messages."""

    type: Literal["alert"] = "alert"
    description: str = Field(..., description="Alert description/message")
    title: str | None = Field(None, description="Optional alert title")
    variant: str | None = Field(
        None,
        description="Alert variant: default, destructive, warning, success",
    )


# Union type of all components
OpenJSONUIComponent = Union[
    OpenJSONUIText,
    OpenJSONUIHeading,
    OpenJSONUICode,
    OpenJSONUITable,
    OpenJSONUIImage,
    OpenJSONUIButton,
    OpenJSONUIList,
    OpenJSONUILink,
    OpenJSONUIDivider,
    OpenJSONUIProgress,
    OpenJSONUIAlert,
]


class OpenJSONUIPayload(BaseModel):
    """Full Open-JSON-UI payload wrapper."""

    type: Literal["open_json_ui"] = "open_json_ui"
    components: list[dict] = Field(
        ..., description="Array of UI components to render"
    )


def create_open_json_ui(components: list[BaseModel]) -> OpenJSONUIPayload:
    """
    Create an Open-JSON-UI payload from component models.

    This factory function converts a list of typed component models
    into a serializable payload that can be returned from agent responses.

    Args:
        components: List of Open-JSON-UI component models

    Returns:
        OpenJSONUIPayload ready for serialization

    Example:
        >>> payload = create_open_json_ui([
        ...     OpenJSONUIHeading(level=1, content="Results"),
        ...     OpenJSONUIText(content="Found 5 documents"),
        ...     OpenJSONUIProgress(value=100, label="Complete"),
        ...     OpenJSONUIButton(label="View All", action="view_all"),
        ... ])
        >>> payload.model_dump()
        {
            "type": "open_json_ui",
            "components": [...]
        }
    """
    return OpenJSONUIPayload(
        components=[c.model_dump() for c in components]
    )


def create_text(
    content: str,
    style: str | None = None,
) -> OpenJSONUIText:
    """Create a text component."""
    return OpenJSONUIText(content=content, style=style)


def create_heading(
    level: int,
    content: str,
) -> OpenJSONUIHeading:
    """Create a heading component."""
    return OpenJSONUIHeading(level=level, content=content)


def create_code(
    content: str,
    language: str | None = None,
) -> OpenJSONUICode:
    """Create a code component."""
    return OpenJSONUICode(content=content, language=language)


def create_table(
    headers: list[str],
    rows: list[list[str]],
    caption: str | None = None,
) -> OpenJSONUITable:
    """Create a table component."""
    return OpenJSONUITable(headers=headers, rows=rows, caption=caption)


def create_image(
    src: str,
    alt: str,
    width: int | None = None,
    height: int | None = None,
) -> OpenJSONUIImage:
    """Create an image component."""
    return OpenJSONUIImage(src=src, alt=alt, width=width, height=height)


def create_button(
    label: str,
    action: str,
    variant: str | None = None,
) -> OpenJSONUIButton:
    """Create a button component."""
    return OpenJSONUIButton(label=label, action=action, variant=variant)


def create_list(
    items: list[str],
    ordered: bool = False,
) -> OpenJSONUIList:
    """Create a list component."""
    return OpenJSONUIList(items=items, ordered=ordered)


def create_link(
    text: str,
    href: str,
    target: str | None = None,
) -> OpenJSONUILink:
    """Create a link component."""
    return OpenJSONUILink(text=text, href=href, target=target)


def create_divider() -> OpenJSONUIDivider:
    """Create a divider component."""
    return OpenJSONUIDivider()


def create_progress(
    value: int,
    label: str | None = None,
) -> OpenJSONUIProgress:
    """Create a progress component."""
    return OpenJSONUIProgress(value=value, label=label)


def create_alert(
    description: str,
    title: str | None = None,
    variant: str | None = None,
) -> OpenJSONUIAlert:
    """Create an alert component."""
    return OpenJSONUIAlert(description=description, title=title, variant=variant)


__all__ = [
    # Component models
    "OpenJSONUIText",
    "OpenJSONUIHeading",
    "OpenJSONUICode",
    "OpenJSONUITable",
    "OpenJSONUIImage",
    "OpenJSONUIButton",
    "OpenJSONUIList",
    "OpenJSONUILink",
    "OpenJSONUIDivider",
    "OpenJSONUIProgress",
    "OpenJSONUIAlert",
    "OpenJSONUIComponent",
    "OpenJSONUIPayload",
    # Factory functions
    "create_open_json_ui",
    "create_text",
    "create_heading",
    "create_code",
    "create_table",
    "create_image",
    "create_button",
    "create_list",
    "create_link",
    "create_divider",
    "create_progress",
    "create_alert",
]
