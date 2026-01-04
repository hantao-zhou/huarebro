from pathlib import Path
from langchain.tools import tool
from langgraph.types import Overwrite

from smartagent.renderer import _DEFAULT_RENDERER
from rich.panel import Panel
from rich.text import Text

from pathlib import Path

WORKSPACE_ROOT = Path("./workspace").resolve()


def resolve_workspace_path(virtual_path: str) -> Path:
    """
    Resolve a virtual workspace path into a real filesystem path.

    Accepts:
    - /workspace
    - /workspace/
    - /workspace/relative/path

    Rejects:
    - Any path outside the workspace namespace
    - Any path attempting directory traversal
    """

    if virtual_path == "/workspace":
        return WORKSPACE_ROOT

    if virtual_path.startswith("/workspace/"):
        relative = virtual_path[len("/workspace/") :]
        real_path = (WORKSPACE_ROOT / relative).resolve()
    else:
        raise ValueError(f"Invalid workspace path: {virtual_path}")

    # Enforce sandboxing
    if not str(real_path).startswith(str(WORKSPACE_ROOT)):
        raise ValueError(f"Path traversal detected: {virtual_path}")

    return real_path


def safe_fix_zip_filename(name: str) -> str:
    """
    Attempt to fix garbled ZIP filenames produced by legacy tools.
    Never raises UnicodeEncodeError.
    """
    try:
        raw = name.encode("latin1")  # latin1 is always reversible
    except Exception:
        return name

    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue

    return name


_DEFAULT_RENDERER.console.print(
    Panel(
        Text("Initialize unzip tool."),
        title="STEP 3: Initialize Tools",
        border_style="purple",
        padding=(1, 2),
    )
)
import zipfile
