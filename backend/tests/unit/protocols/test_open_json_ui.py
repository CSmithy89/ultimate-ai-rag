"""Unit tests for Open-JSON-UI Pydantic models.

Story 22-C2: Implement Open-JSON-UI Renderer
"""

import pytest
from pydantic import ValidationError

from agentic_rag_backend.protocols.open_json_ui import (
    # Component models
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
    OpenJSONUIPayload,
    # Factory functions
    create_open_json_ui,
    create_text,
    create_heading,
    create_code,
    create_table,
    create_image,
    create_button,
    create_list,
    create_link,
    create_divider,
    create_progress,
    create_alert,
)


class TestOpenJSONUIText:
    """Tests for OpenJSONUIText model."""

    def test_valid_text(self) -> None:
        """OpenJSONUIText should accept valid content."""
        text = OpenJSONUIText(content="Hello world")
        assert text.type == "text"
        assert text.content == "Hello world"
        assert text.style is None

    def test_text_with_style(self) -> None:
        """OpenJSONUIText should accept style variants."""
        for style in ["normal", "muted", "error", "success"]:
            text = OpenJSONUIText(content="Styled text", style=style)
            assert text.style == style

    def test_text_required_content(self) -> None:
        """OpenJSONUIText should require content field."""
        with pytest.raises(ValidationError) as exc_info:
            OpenJSONUIText()  # type: ignore
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "content" in field_names


class TestOpenJSONUIHeading:
    """Tests for OpenJSONUIHeading model."""

    def test_valid_heading(self) -> None:
        """OpenJSONUIHeading should accept valid level and content."""
        heading = OpenJSONUIHeading(level=1, content="Main Title")
        assert heading.type == "heading"
        assert heading.level == 1
        assert heading.content == "Main Title"

    def test_heading_levels_1_to_6(self) -> None:
        """OpenJSONUIHeading should accept levels 1-6."""
        for level in range(1, 7):
            heading = OpenJSONUIHeading(level=level, content=f"Level {level}")
            assert heading.level == level

    def test_heading_level_below_1(self) -> None:
        """OpenJSONUIHeading should reject level below 1."""
        with pytest.raises(ValidationError):
            OpenJSONUIHeading(level=0, content="Invalid")

    def test_heading_level_above_6(self) -> None:
        """OpenJSONUIHeading should reject level above 6."""
        with pytest.raises(ValidationError):
            OpenJSONUIHeading(level=7, content="Invalid")

    def test_heading_required_fields(self) -> None:
        """OpenJSONUIHeading should require level and content."""
        with pytest.raises(ValidationError) as exc_info:
            OpenJSONUIHeading()  # type: ignore
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "level" in field_names
        assert "content" in field_names


class TestOpenJSONUICode:
    """Tests for OpenJSONUICode model."""

    def test_valid_code(self) -> None:
        """OpenJSONUICode should accept valid content."""
        code = OpenJSONUICode(content="const x = 1;")
        assert code.type == "code"
        assert code.content == "const x = 1;"
        assert code.language is None

    def test_code_with_language(self) -> None:
        """OpenJSONUICode should accept language specification."""
        code = OpenJSONUICode(content="print('hello')", language="python")
        assert code.language == "python"

    def test_code_required_content(self) -> None:
        """OpenJSONUICode should require content field."""
        with pytest.raises(ValidationError) as exc_info:
            OpenJSONUICode()  # type: ignore
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "content" in field_names


class TestOpenJSONUITable:
    """Tests for OpenJSONUITable model."""

    def test_valid_table(self) -> None:
        """OpenJSONUITable should accept valid headers and rows."""
        table = OpenJSONUITable(
            headers=["Name", "Value"],
            rows=[["foo", "1"], ["bar", "2"]],
        )
        assert table.type == "table"
        assert table.headers == ["Name", "Value"]
        assert len(table.rows) == 2
        assert table.caption is None

    def test_table_with_caption(self) -> None:
        """OpenJSONUITable should accept caption."""
        table = OpenJSONUITable(
            headers=["Column"],
            rows=[["Data"]],
            caption="Table caption",
        )
        assert table.caption == "Table caption"

    def test_empty_table(self) -> None:
        """OpenJSONUITable should accept empty headers and rows."""
        table = OpenJSONUITable(headers=[], rows=[])
        assert table.headers == []
        assert table.rows == []

    def test_table_required_fields(self) -> None:
        """OpenJSONUITable should require headers and rows."""
        with pytest.raises(ValidationError) as exc_info:
            OpenJSONUITable()  # type: ignore
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "headers" in field_names
        assert "rows" in field_names


class TestOpenJSONUIImage:
    """Tests for OpenJSONUIImage model."""

    def test_valid_image(self) -> None:
        """OpenJSONUIImage should accept valid src and alt."""
        image = OpenJSONUIImage(
            src="https://example.com/image.png",
            alt="Example image",
        )
        assert image.type == "image"
        assert image.src == "https://example.com/image.png"
        assert image.alt == "Example image"
        assert image.width is None
        assert image.height is None

    def test_image_with_dimensions(self) -> None:
        """OpenJSONUIImage should accept width and height."""
        image = OpenJSONUIImage(
            src="https://example.com/img.png",
            alt="Sized image",
            width=300,
            height=200,
        )
        assert image.width == 300
        assert image.height == 200

    def test_image_required_fields(self) -> None:
        """OpenJSONUIImage should require src and alt."""
        with pytest.raises(ValidationError) as exc_info:
            OpenJSONUIImage()  # type: ignore
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "src" in field_names
        assert "alt" in field_names


class TestOpenJSONUIButton:
    """Tests for OpenJSONUIButton model."""

    def test_valid_button(self) -> None:
        """OpenJSONUIButton should accept valid label and action."""
        button = OpenJSONUIButton(label="Click me", action="click_action")
        assert button.type == "button"
        assert button.label == "Click me"
        assert button.action == "click_action"
        assert button.variant is None

    def test_button_with_variant(self) -> None:
        """OpenJSONUIButton should accept variant."""
        for variant in ["default", "destructive", "outline", "ghost", "secondary"]:
            button = OpenJSONUIButton(
                label="Button",
                action="action",
                variant=variant,
            )
            assert button.variant == variant

    def test_button_required_fields(self) -> None:
        """OpenJSONUIButton should require label and action."""
        with pytest.raises(ValidationError) as exc_info:
            OpenJSONUIButton()  # type: ignore
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "label" in field_names
        assert "action" in field_names


class TestOpenJSONUIList:
    """Tests for OpenJSONUIList model."""

    def test_valid_list(self) -> None:
        """OpenJSONUIList should accept valid items."""
        list_comp = OpenJSONUIList(items=["Item 1", "Item 2", "Item 3"])
        assert list_comp.type == "list"
        assert len(list_comp.items) == 3
        assert list_comp.ordered is False

    def test_ordered_list(self) -> None:
        """OpenJSONUIList should accept ordered flag."""
        list_comp = OpenJSONUIList(items=["First", "Second"], ordered=True)
        assert list_comp.ordered is True

    def test_empty_list(self) -> None:
        """OpenJSONUIList should accept empty items."""
        list_comp = OpenJSONUIList(items=[])
        assert list_comp.items == []

    def test_list_required_items(self) -> None:
        """OpenJSONUIList should require items field."""
        with pytest.raises(ValidationError) as exc_info:
            OpenJSONUIList()  # type: ignore
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "items" in field_names


class TestOpenJSONUILink:
    """Tests for OpenJSONUILink model."""

    def test_valid_link(self) -> None:
        """OpenJSONUILink should accept valid text and href."""
        link = OpenJSONUILink(text="Click here", href="https://example.com")
        assert link.type == "link"
        assert link.text == "Click here"
        assert link.href == "https://example.com"
        assert link.target is None

    def test_link_with_target(self) -> None:
        """OpenJSONUILink should accept target."""
        link = OpenJSONUILink(
            text="New tab",
            href="https://example.com",
            target="_blank",
        )
        assert link.target == "_blank"

    def test_link_required_fields(self) -> None:
        """OpenJSONUILink should require text and href."""
        with pytest.raises(ValidationError) as exc_info:
            OpenJSONUILink()  # type: ignore
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "text" in field_names
        assert "href" in field_names


class TestOpenJSONUIDivider:
    """Tests for OpenJSONUIDivider model."""

    def test_divider(self) -> None:
        """OpenJSONUIDivider should create with type 'divider'."""
        divider = OpenJSONUIDivider()
        assert divider.type == "divider"

    def test_divider_serialization(self) -> None:
        """OpenJSONUIDivider should serialize correctly."""
        divider = OpenJSONUIDivider()
        data = divider.model_dump()
        assert data == {"type": "divider"}


class TestOpenJSONUIProgress:
    """Tests for OpenJSONUIProgress model."""

    def test_valid_progress(self) -> None:
        """OpenJSONUIProgress should accept valid value."""
        progress = OpenJSONUIProgress(value=50)
        assert progress.type == "progress"
        assert progress.value == 50
        assert progress.label is None

    def test_progress_with_label(self) -> None:
        """OpenJSONUIProgress should accept label."""
        progress = OpenJSONUIProgress(value=75, label="Loading...")
        assert progress.label == "Loading..."

    def test_progress_at_boundaries(self) -> None:
        """OpenJSONUIProgress should accept 0 and 100."""
        progress_0 = OpenJSONUIProgress(value=0)
        assert progress_0.value == 0

        progress_100 = OpenJSONUIProgress(value=100)
        assert progress_100.value == 100

    def test_progress_below_0(self) -> None:
        """OpenJSONUIProgress should reject value below 0."""
        with pytest.raises(ValidationError):
            OpenJSONUIProgress(value=-1)

    def test_progress_above_100(self) -> None:
        """OpenJSONUIProgress should reject value above 100."""
        with pytest.raises(ValidationError):
            OpenJSONUIProgress(value=101)

    def test_progress_required_value(self) -> None:
        """OpenJSONUIProgress should require value field."""
        with pytest.raises(ValidationError) as exc_info:
            OpenJSONUIProgress()  # type: ignore
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "value" in field_names


class TestOpenJSONUIAlert:
    """Tests for OpenJSONUIAlert model."""

    def test_valid_alert(self) -> None:
        """OpenJSONUIAlert should accept valid description."""
        alert = OpenJSONUIAlert(description="This is an alert")
        assert alert.type == "alert"
        assert alert.description == "This is an alert"
        assert alert.title is None
        assert alert.variant is None

    def test_alert_with_title(self) -> None:
        """OpenJSONUIAlert should accept title."""
        alert = OpenJSONUIAlert(title="Warning", description="Be careful")
        assert alert.title == "Warning"

    def test_alert_with_variant(self) -> None:
        """OpenJSONUIAlert should accept variant."""
        for variant in ["default", "destructive", "warning", "success"]:
            alert = OpenJSONUIAlert(description="Message", variant=variant)
            assert alert.variant == variant

    def test_alert_required_description(self) -> None:
        """OpenJSONUIAlert should require description field."""
        with pytest.raises(ValidationError) as exc_info:
            OpenJSONUIAlert()  # type: ignore
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "description" in field_names


class TestOpenJSONUIPayload:
    """Tests for OpenJSONUIPayload model."""

    def test_valid_payload(self) -> None:
        """OpenJSONUIPayload should accept valid components."""
        payload = OpenJSONUIPayload(
            components=[
                {"type": "text", "content": "Hello"},
                {"type": "button", "label": "Click", "action": "click"},
            ]
        )
        assert payload.type == "open_json_ui"
        assert len(payload.components) == 2

    def test_empty_payload(self) -> None:
        """OpenJSONUIPayload should accept empty components."""
        payload = OpenJSONUIPayload(components=[])
        assert payload.components == []

    def test_payload_required_components(self) -> None:
        """OpenJSONUIPayload should require components field."""
        with pytest.raises(ValidationError) as exc_info:
            OpenJSONUIPayload()  # type: ignore
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "components" in field_names

    def test_payload_serialization(self) -> None:
        """OpenJSONUIPayload should serialize correctly."""
        payload = OpenJSONUIPayload(
            components=[{"type": "divider"}]
        )
        data = payload.model_dump()
        assert data["type"] == "open_json_ui"
        assert len(data["components"]) == 1


class TestCreateOpenJsonUi:
    """Tests for create_open_json_ui factory function."""

    def test_create_payload(self) -> None:
        """create_open_json_ui should create valid payload."""
        payload = create_open_json_ui([
            OpenJSONUIText(content="Hello"),
            OpenJSONUIButton(label="Click", action="click"),
        ])
        assert payload.type == "open_json_ui"
        assert len(payload.components) == 2

    def test_create_empty_payload(self) -> None:
        """create_open_json_ui should create empty payload."""
        payload = create_open_json_ui([])
        assert payload.components == []

    def test_create_payload_serialization(self) -> None:
        """create_open_json_ui should produce serializable payload."""
        payload = create_open_json_ui([
            OpenJSONUIHeading(level=1, content="Title"),
            OpenJSONUIText(content="Description"),
            OpenJSONUIDivider(),
        ])
        data = payload.model_dump()

        assert data["type"] == "open_json_ui"
        assert len(data["components"]) == 3
        assert data["components"][0]["type"] == "heading"
        assert data["components"][1]["type"] == "text"
        assert data["components"][2]["type"] == "divider"

    def test_create_complex_payload(self) -> None:
        """create_open_json_ui should handle all component types."""
        payload = create_open_json_ui([
            OpenJSONUIHeading(level=1, content="Results"),
            OpenJSONUIText(content="Found 5 items"),
            OpenJSONUITable(
                headers=["Name", "Status"],
                rows=[["Item 1", "Active"], ["Item 2", "Pending"]],
            ),
            OpenJSONUIProgress(value=100, label="Complete"),
            OpenJSONUIAlert(
                title="Success",
                description="All done!",
                variant="success",
            ),
            OpenJSONUIButton(label="View All", action="view_all"),
        ])

        data = payload.model_dump()
        assert len(data["components"]) == 6


class TestIndividualFactoryFunctions:
    """Tests for individual factory functions."""

    def test_create_text(self) -> None:
        """create_text should create text component."""
        text = create_text("Hello", style="muted")
        assert text.type == "text"
        assert text.content == "Hello"
        assert text.style == "muted"

    def test_create_text_default_style(self) -> None:
        """create_text should have None style by default."""
        text = create_text("Hello")
        assert text.style is None

    def test_create_heading(self) -> None:
        """create_heading should create heading component."""
        heading = create_heading(2, "Section Title")
        assert heading.type == "heading"
        assert heading.level == 2
        assert heading.content == "Section Title"

    def test_create_code(self) -> None:
        """create_code should create code component."""
        code = create_code("const x = 1;", language="javascript")
        assert code.type == "code"
        assert code.content == "const x = 1;"
        assert code.language == "javascript"

    def test_create_code_without_language(self) -> None:
        """create_code should work without language."""
        code = create_code("some code")
        assert code.language is None

    def test_create_table(self) -> None:
        """create_table should create table component."""
        table = create_table(
            headers=["Col1", "Col2"],
            rows=[["a", "b"]],
            caption="Test table",
        )
        assert table.type == "table"
        assert table.headers == ["Col1", "Col2"]
        assert len(table.rows) == 1
        assert table.caption == "Test table"

    def test_create_image(self) -> None:
        """create_image should create image component."""
        image = create_image(
            src="https://example.com/img.png",
            alt="Test image",
            width=300,
            height=200,
        )
        assert image.type == "image"
        assert image.src == "https://example.com/img.png"
        assert image.alt == "Test image"
        assert image.width == 300
        assert image.height == 200

    def test_create_image_minimal(self) -> None:
        """create_image should work with minimal parameters."""
        image = create_image(src="https://example.com/img.png", alt="Image")
        assert image.width is None
        assert image.height is None

    def test_create_button(self) -> None:
        """create_button should create button component."""
        button = create_button(
            label="Submit",
            action="submit_form",
            variant="destructive",
        )
        assert button.type == "button"
        assert button.label == "Submit"
        assert button.action == "submit_form"
        assert button.variant == "destructive"

    def test_create_button_minimal(self) -> None:
        """create_button should work with minimal parameters."""
        button = create_button(label="Click", action="click")
        assert button.variant is None

    def test_create_list(self) -> None:
        """create_list should create list component."""
        list_comp = create_list(
            items=["Item 1", "Item 2"],
            ordered=True,
        )
        assert list_comp.type == "list"
        assert list_comp.items == ["Item 1", "Item 2"]
        assert list_comp.ordered is True

    def test_create_list_unordered(self) -> None:
        """create_list should default to unordered."""
        list_comp = create_list(items=["A", "B"])
        assert list_comp.ordered is False

    def test_create_link(self) -> None:
        """create_link should create link component."""
        link = create_link(
            text="Click here",
            href="https://example.com",
            target="_blank",
        )
        assert link.type == "link"
        assert link.text == "Click here"
        assert link.href == "https://example.com"
        assert link.target == "_blank"

    def test_create_link_minimal(self) -> None:
        """create_link should work with minimal parameters."""
        link = create_link(text="Link", href="https://example.com")
        assert link.target is None

    def test_create_divider(self) -> None:
        """create_divider should create divider component."""
        divider = create_divider()
        assert divider.type == "divider"

    def test_create_progress(self) -> None:
        """create_progress should create progress component."""
        progress = create_progress(value=75, label="Loading...")
        assert progress.type == "progress"
        assert progress.value == 75
        assert progress.label == "Loading..."

    def test_create_progress_minimal(self) -> None:
        """create_progress should work with minimal parameters."""
        progress = create_progress(value=50)
        assert progress.label is None

    def test_create_alert(self) -> None:
        """create_alert should create alert component."""
        alert = create_alert(
            description="Something happened",
            title="Notice",
            variant="warning",
        )
        assert alert.type == "alert"
        assert alert.description == "Something happened"
        assert alert.title == "Notice"
        assert alert.variant == "warning"

    def test_create_alert_minimal(self) -> None:
        """create_alert should work with minimal parameters."""
        alert = create_alert(description="Message")
        assert alert.title is None
        assert alert.variant is None


class TestEndToEndSerialization:
    """End-to-end tests for payload creation and serialization."""

    def test_full_payload_json_serialization(self) -> None:
        """Payload should be JSON-serializable."""
        import json

        payload = create_open_json_ui([
            create_heading(1, "Dashboard"),
            create_text("Welcome to your dashboard", style="muted"),
            create_divider(),
            create_progress(value=65, label="Task completion"),
            create_table(
                headers=["Task", "Status", "Priority"],
                rows=[
                    ["Design review", "Complete", "High"],
                    ["Code review", "In Progress", "Medium"],
                ],
            ),
            create_alert(
                title="Reminder",
                description="You have 2 pending tasks",
                variant="warning",
            ),
            create_button(label="View All Tasks", action="view_tasks"),
        ])

        # Should not raise
        json_str = json.dumps(payload.model_dump())
        assert isinstance(json_str, str)

        # Should parse back correctly
        parsed = json.loads(json_str)
        assert parsed["type"] == "open_json_ui"
        assert len(parsed["components"]) == 7

    def test_nested_content_serialization(self) -> None:
        """Components with special characters should serialize correctly."""
        import json

        payload = create_open_json_ui([
            create_text('Text with "quotes" and <brackets>'),
            create_code('const x = { key: "value" };\n// comment', language="js"),
            create_table(
                headers=["Name & Description"],
                rows=[['Item with special chars: < > & "']],
            ),
        ])

        json_str = json.dumps(payload.model_dump())
        parsed = json.loads(json_str)

        assert parsed["components"][0]["content"] == 'Text with "quotes" and <brackets>'
        assert "// comment" in parsed["components"][1]["content"]
