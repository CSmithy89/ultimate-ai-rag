"""Tests for A2UI Protocol Support.

Story 21-D1: Enable A2UI Support
"""

import pytest

from agentic_rag_backend.protocols.a2ui import (
    A2UIWidget,
    A2UIAction,
    A2UIFormField,
    create_a2ui_card,
    create_a2ui_table,
    create_a2ui_form,
    create_a2ui_chart,
    create_a2ui_image,
    create_a2ui_list,
    widgets_to_state,
)


class TestA2UIWidget:
    """Tests for A2UIWidget model."""

    def test_widget_basic(self) -> None:
        """Test creating a basic widget."""
        widget = A2UIWidget(type="card", properties={"title": "Test"})

        assert widget.type == "card"
        assert widget.properties == {"title": "Test"}
        assert widget.id is None

    def test_widget_with_id(self) -> None:
        """Test widget with optional ID."""
        widget = A2UIWidget(type="table", properties={}, id="widget-123")

        assert widget.id == "widget-123"

    def test_widget_serialization(self) -> None:
        """Test widget serializes correctly."""
        widget = A2UIWidget(
            type="card",
            properties={"title": "Test", "content": "Content"},
            id="test-id",
        )

        data = widget.model_dump()

        assert data["type"] == "card"
        assert data["properties"]["title"] == "Test"
        assert data["id"] == "test-id"


class TestA2UIAction:
    """Tests for A2UIAction model."""

    def test_action_basic(self) -> None:
        """Test creating a basic action."""
        action = A2UIAction(label="Click me", action="click_action")

        assert action.label == "Click me"
        assert action.action == "click_action"
        assert action.variant == "secondary"
        assert action.disabled is False

    def test_action_with_variant(self) -> None:
        """Test action with custom variant."""
        action = A2UIAction(
            label="Delete",
            action="delete_item",
            variant="destructive",
            disabled=True,
        )

        assert action.variant == "destructive"
        assert action.disabled is True


class TestA2UIFormField:
    """Tests for A2UIFormField model."""

    def test_form_field_basic(self) -> None:
        """Test creating a basic form field."""
        field = A2UIFormField(name="email", label="Email Address")

        assert field.name == "email"
        assert field.label == "Email Address"
        assert field.type == "text"
        assert field.required is False

    def test_form_field_with_options(self) -> None:
        """Test form field with select options."""
        field = A2UIFormField(
            name="country",
            label="Country",
            type="select",
            options=[
                {"value": "us", "label": "United States"},
                {"value": "uk", "label": "United Kingdom"},
            ],
            required=True,
        )

        assert field.type == "select"
        assert len(field.options) == 2
        assert field.required is True

    def test_form_field_alias(self) -> None:
        """Test form field defaultValue alias."""
        field = A2UIFormField(
            name="count",
            label="Count",
            type="number",
            default_value=10,
        )

        data = field.model_dump(by_alias=True)

        assert data["defaultValue"] == 10


class TestCreateA2UICard:
    """Tests for create_a2ui_card function."""

    def test_card_minimal(self) -> None:
        """Test creating a minimal card."""
        widget = create_a2ui_card(title="Test Title", content="Test content")

        assert widget.type == "card"
        assert widget.properties["title"] == "Test Title"
        assert widget.properties["content"] == "Test content"

    def test_card_with_all_options(self) -> None:
        """Test creating a card with all options."""
        actions = [A2UIAction(label="View", action="view_details")]
        widget = create_a2ui_card(
            title="Full Card",
            content="Full content",
            subtitle="Subtitle",
            actions=actions,
            footer="Footer text",
            image_url="https://example.com/image.jpg",
            widget_id="card-123",
        )

        props = widget.properties
        assert props["title"] == "Full Card"
        assert props["subtitle"] == "Subtitle"
        assert props["footer"] == "Footer text"
        assert props["imageUrl"] == "https://example.com/image.jpg"
        assert len(props["actions"]) == 1
        assert widget.id == "card-123"


class TestCreateA2UITable:
    """Tests for create_a2ui_table function."""

    def test_table_basic(self) -> None:
        """Test creating a basic table."""
        widget = create_a2ui_table(
            headers=["Name", "Value"],
            rows=[["Item 1", "100"], ["Item 2", "200"]],
        )

        assert widget.type == "table"
        assert widget.properties["headers"] == ["Name", "Value"]
        assert len(widget.properties["rows"]) == 2
        assert widget.properties["sortable"] is False

    def test_table_with_options(self) -> None:
        """Test creating a table with all options."""
        widget = create_a2ui_table(
            headers=["Column A", "Column B"],
            rows=[["A1", "B1"]],
            caption="Test Table",
            sortable=True,
            widget_id="table-1",
        )

        assert widget.properties["caption"] == "Test Table"
        assert widget.properties["sortable"] is True
        assert widget.id == "table-1"


class TestCreateA2UIForm:
    """Tests for create_a2ui_form function."""

    def test_form_basic(self) -> None:
        """Test creating a basic form."""
        fields = [
            A2UIFormField(name="name", label="Name"),
            A2UIFormField(name="email", label="Email", type="email", required=True),
        ]
        widget = create_a2ui_form(title="Contact Form", fields=fields)

        assert widget.type == "form"
        assert widget.properties["title"] == "Contact Form"
        assert len(widget.properties["fields"]) == 2
        assert widget.properties["submitLabel"] == "Submit"

    def test_form_with_options(self) -> None:
        """Test creating a form with all options."""
        fields = [A2UIFormField(name="query", label="Search Query")]
        widget = create_a2ui_form(
            title="Search Form",
            fields=fields,
            submit_label="Search",
            submit_action="perform_search",
            description="Enter your search query",
            widget_id="search-form",
        )

        props = widget.properties
        assert props["submitLabel"] == "Search"
        assert props["submitAction"] == "perform_search"
        assert props["description"] == "Enter your search query"


class TestCreateA2UIChart:
    """Tests for create_a2ui_chart function."""

    def test_chart_basic(self) -> None:
        """Test creating a basic chart."""
        data = [
            {"month": "Jan", "sales": 100},
            {"month": "Feb", "sales": 150},
        ]
        widget = create_a2ui_chart(
            chart_type="bar",
            data=data,
            x_key="month",
            y_key="sales",
        )

        assert widget.type == "chart"
        assert widget.properties["chartType"] == "bar"
        assert widget.properties["xKey"] == "month"
        assert widget.properties["yKey"] == "sales"
        assert len(widget.properties["data"]) == 2

    def test_chart_with_labels(self) -> None:
        """Test creating a chart with labels."""
        widget = create_a2ui_chart(
            chart_type="line",
            data=[{"x": 1, "y": 2}],
            x_key="x",
            y_key="y",
            title="Sales Trend",
            x_label="Month",
            y_label="Revenue",
            widget_id="chart-1",
        )

        props = widget.properties
        assert props["title"] == "Sales Trend"
        assert props["xLabel"] == "Month"
        assert props["yLabel"] == "Revenue"


class TestCreateA2UIImage:
    """Tests for create_a2ui_image function."""

    def test_image_basic(self) -> None:
        """Test creating a basic image widget."""
        widget = create_a2ui_image(
            url="https://example.com/image.png",
            alt="Example image",
        )

        assert widget.type == "image"
        assert widget.properties["url"] == "https://example.com/image.png"
        assert widget.properties["alt"] == "Example image"

    def test_image_with_dimensions(self) -> None:
        """Test creating an image with dimensions."""
        widget = create_a2ui_image(
            url="https://example.com/photo.jpg",
            alt="Photo",
            caption="A beautiful photo",
            width=800,
            height=600,
            widget_id="photo-1",
        )

        props = widget.properties
        assert props["caption"] == "A beautiful photo"
        assert props["width"] == 800
        assert props["height"] == 600


class TestCreateA2UIList:
    """Tests for create_a2ui_list function."""

    def test_list_basic(self) -> None:
        """Test creating a basic list widget."""
        items = [
            {"text": "Item 1"},
            {"text": "Item 2", "description": "Description"},
        ]
        widget = create_a2ui_list(items=items)

        assert widget.type == "list"
        assert len(widget.properties["items"]) == 2
        assert widget.properties["ordered"] is False
        assert widget.properties["selectable"] is False

    def test_list_with_options(self) -> None:
        """Test creating a list with options."""
        items = [
            {"text": "Step 1", "icon": "check"},
            {"text": "Step 2", "badge": "New"},
        ]
        widget = create_a2ui_list(
            items=items,
            title="Steps to Complete",
            ordered=True,
            selectable=True,
            widget_id="steps-list",
        )

        props = widget.properties
        assert props["title"] == "Steps to Complete"
        assert props["ordered"] is True
        assert props["selectable"] is True


class TestWidgetsToState:
    """Tests for widgets_to_state utility."""

    def test_empty_widgets(self) -> None:
        """Test converting empty widget list."""
        result = widgets_to_state([])

        assert result == {"a2ui_widgets": []}

    def test_single_widget(self) -> None:
        """Test converting single widget."""
        widget = create_a2ui_card(title="Test", content="Content")
        result = widgets_to_state([widget])

        assert len(result["a2ui_widgets"]) == 1
        assert result["a2ui_widgets"][0]["type"] == "card"

    def test_multiple_widgets(self) -> None:
        """Test converting multiple widgets."""
        widgets = [
            create_a2ui_card(title="Card", content="Card content"),
            create_a2ui_table(headers=["A"], rows=[["1"]]),
            create_a2ui_list(items=[{"text": "Item"}]),
        ]
        result = widgets_to_state(widgets)

        assert len(result["a2ui_widgets"]) == 3
        assert result["a2ui_widgets"][0]["type"] == "card"
        assert result["a2ui_widgets"][1]["type"] == "table"
        assert result["a2ui_widgets"][2]["type"] == "list"
