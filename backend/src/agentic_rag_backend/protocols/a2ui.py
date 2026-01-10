"""A2UI Protocol Support for Google's Agent-to-UI specification.

Story 21-D1: Enable A2UI Support

This module provides A2UI widget definitions and helper functions for
agents to emit rich UI payloads in their responses. These widgets are
rendered by the frontend A2UIRenderer component.

A2UI Specification Reference:
- Widgets describe declarative UI that the frontend can render
- Each widget has a type and properties
- Widgets are emitted via STATE_SNAPSHOT events to CopilotKit
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# Widget type literals
A2UIWidgetType = Literal["card", "table", "form", "chart", "image", "list"]


class A2UIAction(BaseModel):
    """Action button for A2UI widgets."""

    label: str = Field(..., description="Button label text")
    action: str = Field(..., description="Action identifier")
    variant: Literal["primary", "secondary", "destructive"] = Field(
        default="secondary", description="Button variant"
    )
    disabled: bool = Field(default=False, description="Whether action is disabled")


class A2UIFormField(BaseModel):
    """Form field definition for A2UI form widgets."""

    name: str = Field(..., description="Field name/key")
    label: str = Field(..., description="Field label")
    type: Literal["text", "number", "email", "password", "textarea", "select", "checkbox"] = Field(
        default="text", description="Field input type"
    )
    placeholder: Optional[str] = Field(default=None, description="Placeholder text")
    required: bool = Field(default=False, description="Whether field is required")
    options: Optional[list[dict[str, str]]] = Field(
        default=None, description="Options for select fields"
    )
    default_value: Optional[Any] = Field(
        default=None, alias="defaultValue", description="Default field value"
    )

    model_config = {"populate_by_name": True}


class A2UIWidget(BaseModel):
    """A2UI widget payload for declarative UI rendering.

    This is the base model for all A2UI widgets. Each widget type
    has specific properties that control its rendering.
    """

    type: A2UIWidgetType = Field(..., description="Widget type")
    properties: dict[str, Any] = Field(default_factory=dict, description="Widget properties")
    id: Optional[str] = Field(default=None, description="Optional widget identifier")


# ============================================
# WIDGET FACTORY FUNCTIONS
# ============================================


def create_a2ui_card(
    title: str,
    content: str,
    subtitle: Optional[str] = None,
    actions: Optional[list[A2UIAction]] = None,
    footer: Optional[str] = None,
    image_url: Optional[str] = None,
    widget_id: Optional[str] = None,
) -> A2UIWidget:
    """Create an A2UI card widget.

    Args:
        title: Card title
        content: Main content text (supports markdown)
        subtitle: Optional subtitle
        actions: Optional list of action buttons
        footer: Optional footer text
        image_url: Optional header image URL
        widget_id: Optional unique identifier

    Returns:
        A2UIWidget with type="card"
    """
    properties: dict[str, Any] = {
        "title": title,
        "content": content,
    }

    if subtitle:
        properties["subtitle"] = subtitle
    if actions:
        properties["actions"] = [a.model_dump() for a in actions]
    if footer:
        properties["footer"] = footer
    if image_url:
        properties["imageUrl"] = image_url

    return A2UIWidget(type="card", properties=properties, id=widget_id)


def create_a2ui_table(
    headers: list[str],
    rows: list[list[Any]],
    caption: Optional[str] = None,
    sortable: bool = False,
    widget_id: Optional[str] = None,
) -> A2UIWidget:
    """Create an A2UI table widget.

    Args:
        headers: Column header labels
        rows: Table row data (2D array)
        caption: Optional table caption
        sortable: Whether columns are sortable
        widget_id: Optional unique identifier

    Returns:
        A2UIWidget with type="table"
    """
    properties: dict[str, Any] = {
        "headers": headers,
        "rows": rows,
        "sortable": sortable,
    }

    if caption:
        properties["caption"] = caption

    return A2UIWidget(type="table", properties=properties, id=widget_id)


def create_a2ui_form(
    title: str,
    fields: list[A2UIFormField],
    submit_label: str = "Submit",
    submit_action: str = "form_submit",
    description: Optional[str] = None,
    widget_id: Optional[str] = None,
) -> A2UIWidget:
    """Create an A2UI form widget.

    Args:
        title: Form title
        fields: List of form field definitions
        submit_label: Submit button label
        submit_action: Action identifier for form submission
        description: Optional form description
        widget_id: Optional unique identifier

    Returns:
        A2UIWidget with type="form"
    """
    properties: dict[str, Any] = {
        "title": title,
        "fields": [f.model_dump(by_alias=True) for f in fields],
        "submitLabel": submit_label,
        "submitAction": submit_action,
    }

    if description:
        properties["description"] = description

    return A2UIWidget(type="form", properties=properties, id=widget_id)


def create_a2ui_chart(
    chart_type: Literal["bar", "line", "pie", "area", "scatter"],
    data: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    widget_id: Optional[str] = None,
) -> A2UIWidget:
    """Create an A2UI chart widget.

    Args:
        chart_type: Type of chart (bar, line, pie, area, scatter)
        data: Chart data points
        x_key: Key for x-axis values in data
        y_key: Key for y-axis values in data
        title: Optional chart title
        x_label: Optional x-axis label
        y_label: Optional y-axis label
        widget_id: Optional unique identifier

    Returns:
        A2UIWidget with type="chart"
    """
    properties: dict[str, Any] = {
        "chartType": chart_type,
        "data": data,
        "xKey": x_key,
        "yKey": y_key,
    }

    if title:
        properties["title"] = title
    if x_label:
        properties["xLabel"] = x_label
    if y_label:
        properties["yLabel"] = y_label

    return A2UIWidget(type="chart", properties=properties, id=widget_id)


def create_a2ui_image(
    url: str,
    alt: str,
    caption: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    widget_id: Optional[str] = None,
) -> A2UIWidget:
    """Create an A2UI image widget.

    Args:
        url: Image URL
        alt: Alt text for accessibility
        caption: Optional image caption
        width: Optional width in pixels
        height: Optional height in pixels
        widget_id: Optional unique identifier

    Returns:
        A2UIWidget with type="image"
    """
    properties: dict[str, Any] = {
        "url": url,
        "alt": alt,
    }

    if caption:
        properties["caption"] = caption
    if width:
        properties["width"] = width
    if height:
        properties["height"] = height

    return A2UIWidget(type="image", properties=properties, id=widget_id)


def create_a2ui_list(
    items: list[dict[str, Any]],
    title: Optional[str] = None,
    ordered: bool = False,
    selectable: bool = False,
    widget_id: Optional[str] = None,
) -> A2UIWidget:
    """Create an A2UI list widget.

    Each item should have at minimum a "text" key. Optional keys:
    - icon: Icon name or URL
    - description: Secondary text
    - badge: Badge text
    - href: Link URL

    Args:
        items: List of item dictionaries
        title: Optional list title
        ordered: Whether list is ordered (numbered)
        selectable: Whether items are selectable
        widget_id: Optional unique identifier

    Returns:
        A2UIWidget with type="list"
    """
    properties: dict[str, Any] = {
        "items": items,
        "ordered": ordered,
        "selectable": selectable,
    }

    if title:
        properties["title"] = title

    return A2UIWidget(type="list", properties=properties, id=widget_id)


# ============================================
# UTILITY FUNCTIONS
# ============================================


def widgets_to_state(widgets: list[A2UIWidget]) -> dict[str, list[dict[str, Any]]]:
    """Convert A2UI widgets to state format for STATE_SNAPSHOT event.

    Args:
        widgets: List of A2UIWidget instances

    Returns:
        Dict with "a2ui_widgets" key containing serialized widgets
    """
    return {
        "a2ui_widgets": [w.model_dump() for w in widgets],
    }
