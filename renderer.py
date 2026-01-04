from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text


# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class Theme:
    user_title: str = "User"
    user_style: str = "bold blue"

    assistant_title: str = "Assistant"
    assistant_style: str = "bold green"

    tool_call_title: str = "Tool Call"
    tool_call_style: str = "bold magenta"

    tool_output_title: str = "Tool Output"
    tool_output_style: str = "bold yellow"

    file_title: str = "File Artifact"
    file_style: str = "bold cyan"

    system_title: str = "System"
    system_style: str = "dim"

    divider_style: str = "dim white"


# -----------------------------
# Renderer
# -----------------------------

class RichAgentRenderer:
    """
    Render streamed events and final outputs from LangChain/LangGraph-style agents
    using Rich panels.

    Design goals:
    - Minimal dependencies on LangGraph internals (e.g., Overwrite wrapper)
    - Robust to multiple message/content formats
    - Deterministic, readable output with safe truncation
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        theme: Theme = Theme(),
        markdown_lexer: str = "markdown",
        syntax_theme: str = "monokai",
        file_preview_lines: int = 999,
    ) -> None:
        self.console = console or Console()
        self.theme = theme
        self.markdown_lexer = markdown_lexer
        self.syntax_theme = syntax_theme
        self.file_preview_lines = file_preview_lines

    # -------------------------
    # Public API
    # -------------------------

    def show_prompt(self, prompt_text: str, title: str = "Prompt", border_style: str = "blue") -> None:
        """
        Render a system or orchestrator prompt in a styled panel.
        """
        text = Text(prompt_text)
        text.highlight_regex(r"^#+.*$", style="bold magenta")
        text.highlight_regex(r"<[^>]+>", style="bold cyan")

        self.console.print(
            Panel(
                text,
                title=f"[bold green]{title}[/bold green]",
                border_style=border_style,
                padding=(1, 2),
            )
        )

    def render_stream_event(self, event: Mapping[str, Any]) -> None:
        """
        Render a single streamed event emitted by an agent.

        Expected shape:
            {"event_type": payload}
        """
        event_type, payload = self._extract_single_kv(event)

        self._divider(event_type)

        if event_type in {"PatchToolCallsMiddleware.before_agent", "model", "tools"}:
            for msg in self._extract_messages(payload):
                self.render_message(msg)

            if event_type == "tools":
                self._render_files_from_payload(payload)

        else:
            # Unknown/middleware events: show payload in JSON form
            self._render_system_payload(payload)

    def render_final_output(self, result: Mapping[str, Any]) -> None:
        """
        Render the final output returned by agent.invoke(...).

        Expected shape:
            {"messages": [...], "files": {...}, ...}
        """
        self._divider("FINAL OUTPUT")

        for msg in self._extract_messages(result):
            self.render_message(msg)

        files = self._get_payload_value(result, "files", default={}) or {}
        if isinstance(files, Mapping) and files:
            self._divider("FILES")
            for path, meta in files.items():
                self._render_file_meta(str(path), meta)

    def render_message(self, message: Any) -> None:
        """
        Render a single LangChain-style message or compatible object.
        """
        cls_name = message.__class__.__name__

        # Support common LangChain message class names
        if cls_name == "HumanMessage":
            self._render_text_panel(
                text=self._format_message_content(message),
                title=self.theme.user_title,
                border_style=self.theme.user_style,
            )
            return

        if cls_name == "AIMessage":
            self._render_text_panel(
                text=self._format_message_content(message) or "",
                title=self.theme.assistant_title,
                border_style=self.theme.assistant_style,
            )
            self._render_tool_calls_from_message(message)
            return

        if cls_name == "ToolMessage":
            self._render_tool_output_message(message)
            return

        # Fallback: render whatever we can
        self._render_text_panel(
            text=self._format_message_content(message),
            title=cls_name,
            border_style="white",
        )

    # -------------------------
    # Message formatting
    # -------------------------

    def _format_message_content(self, message: Any) -> str:
        """
        Extract and render message.content robustly. Supports:
        - Plain strings
        - List-of-blocks with {"type": "text"} / {"type": "tool_use"}
        - Any other object via str(...)
        """
        content = getattr(message, "content", "")

        parts: List[str] = []
        tool_calls_from_content = False

        if isinstance(content, str):
            parts.append(content)

        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    parts.append(str(item))
                    continue

                item_type = item.get("type")
                if item_type == "text":
                    parts.append(str(item.get("text", "")))
                elif item_type in {"tool_use", "tool_call"}:
                    tool_calls_from_content = True
                    name = item.get("name", "unknown_tool")
                    tool_input = item.get("input", item.get("args", {}))
                    call_id = item.get("id", "N/A")
                    parts.append(f"[{self.theme.tool_call_title}] {name}")
                    parts.append(json.dumps(tool_input, indent=2, ensure_ascii=False))
                    parts.append(f"id: {call_id}")
                else:
                    parts.append(str(item))

        else:
            parts.append(str(content))

        # If tool calls were not embedded in content blocks, also check tool_calls attribute
        if not tool_calls_from_content:
            self._append_tool_calls_attribute(parts, message)

        return "\n".join(p for p in parts if p is not None and p != "")

    def _append_tool_calls_attribute(self, parts: List[str], message: Any) -> None:
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return

        # tool_calls usually: [{"name": ..., "args": ..., "id": ...}, ...]
        if isinstance(tool_calls, Sequence):
            for call in tool_calls:
                if not isinstance(call, Mapping):
                    continue
                parts.append(f"[{self.theme.tool_call_title}] {call.get('name', 'unknown_tool')}")
                parts.append(json.dumps(call.get("args", {}), indent=2, ensure_ascii=False))
                parts.append(f"id: {call.get('id', 'N/A')}")

    def _render_tool_calls_from_message(self, message: Any) -> None:
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls or not isinstance(tool_calls, Sequence):
            return

        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue
            name = str(call.get("name", "unknown_tool"))
            args = call.get("args", {})
            call_id = call.get("id", "N/A")

            self._render_json_panel(
                data={"name": name, "args": args, "id": call_id},
                title=f"{self.theme.tool_call_title} — {name}",
                border_style=self.theme.tool_call_style,
            )

    def _render_tool_output_message(self, message: Any) -> None:
        content = getattr(message, "content", "")

        # Tool outputs are often JSON strings; parse if possible.
        if isinstance(content, str):
            parsed = self._try_parse_json(content)
            if parsed is not None:
                self._render_json_panel(
                    data=parsed,
                    title=self.theme.tool_output_title,
                    border_style=self.theme.tool_output_style,
                )
                return

        self._render_text_panel(
            text=str(content),
            title=self.theme.tool_output_title,
            border_style=self.theme.tool_output_style,
        )

    # -------------------------
    # Stream payload helpers
    # -------------------------

    def _extract_messages(self, payload: Any) -> List[Any]:
        """
        Extract 'messages' from dict-like payloads; returns an empty list if missing.
        Normalizes singleton -> list.
        """
        payload = self._unwrap_overwrite(payload)

        messages = self._get_payload_value(payload, "messages", default=[])
        messages = self._unwrap_overwrite(messages)

        if messages is None:
            return []
        if isinstance(messages, (list, tuple)):
            return list(messages)
        return [messages]

    def _render_files_from_payload(self, payload: Any) -> None:
        payload = self._unwrap_overwrite(payload)
        files = self._get_payload_value(payload, "files", default={}) or {}
        if not isinstance(files, Mapping):
            return
        for path, meta in files.items():
            self._render_file_meta(str(path), meta)

    def _render_file_meta(self, path: str, meta: Any) -> None:
        meta = meta or {}
        if not isinstance(meta, Mapping):
            meta = {"meta": str(meta)}

        preview = meta.get("content", [])
        if isinstance(preview, list):
            preview = preview[: self.file_preview_lines]
        else:
            preview = str(preview)

        self._render_json_panel(
            data={
                "path": path,
                "created_at": meta.get("created_at"),
                "modified_at": meta.get("modified_at"),
                "preview": preview,
            },
            title=f"{self.theme.file_title} — {path}",
            border_style=self.theme.file_style,
        )

    def _render_system_payload(self, payload: Any) -> None:
        payload = self._unwrap_overwrite(payload)

        # Prefer JSON panel for dict/list payloads
        if isinstance(payload, (dict, list)):
            self._render_json_panel(payload, self.theme.system_title, self.theme.system_style)
        else:
            self._render_text_panel(str(payload), self.theme.system_title, self.theme.system_style)

    # -------------------------
    # Low-level rendering
    # -------------------------

    def _divider(self, label: str = "") -> None:
        self.console.print(Rule(label, style=self.theme.divider_style))

    def _render_text_panel(self, text: str, title: str, border_style: str) -> None:
        syntax = Syntax(
            text or "",
            lexer=self.markdown_lexer,
            theme=self.syntax_theme,
            word_wrap=True,
        )
        self.console.print(
            Panel(
                syntax,
                title=title,
                border_style=border_style,
                padding=(1, 2),
            )
        )

    def _render_json_panel(self, data: Any, title: str, border_style: str) -> None:
        self.console.print(
            Panel(
                JSON.from_data(data, indent=2),
                title=title,
                border_style=border_style,
                padding=(1, 1),
            )
        )

    # -------------------------
    # Utilities
    # -------------------------

    @staticmethod
    def _extract_single_kv(mapping: Mapping[str, Any]) -> Tuple[str, Any]:
        if not mapping:
            raise ValueError("Event payload is empty.")
        if len(mapping) != 1:
            # Keep deterministic behavior, but avoid silent misrender
            raise ValueError(f"Expected exactly 1 top-level key in event, got {len(mapping)}.")
        return next(iter(mapping.items()))

    @staticmethod
    def _try_parse_json(text: str) -> Optional[Any]:
        try:
            return json.loads(text)
        except Exception:
            return None

    @staticmethod
    def _unwrap_overwrite(value: Any) -> Any:
        """
        Unwrap LangGraph Overwrite wrapper without requiring a direct import.
        """
        if value is None:
            return None
        # Common pattern: Overwrite(value=<payload>)
        if getattr(value, "__class__", None) and value.__class__.__name__ == "Overwrite":
            return getattr(value, "value", value)
        return value

    @staticmethod
    def _get_payload_value(payload: Any, key: str, default: Any = None) -> Any:
        payload = RichAgentRenderer._unwrap_overwrite(payload)
        if isinstance(payload, Mapping):
            return payload.get(key, default)
        return getattr(payload, key, default)


# -----------------------------
# Convenience functions
# -----------------------------

_DEFAULT_RENDERER = RichAgentRenderer()
