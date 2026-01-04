from __future__ import annotations

from pathlib import Path
import zipfile
import shutil
import time

from langchain.tools import tool

from smartagent.workspace import WORKSPACE_ROOT, resolve_workspace_path, safe_fix_zip_filename

@tool(parse_docstring=True)
def unzip_workspace_file(virtual_zip_path: str) -> dict:
    """
    Unzip a ZIP archive located inside the workspace.

    This preprocessing function extracts the contents of a ZIP file referenced
    by a virtual workspace path. The archive is unpacked into a directory with
    the same base name as the ZIP file, located in the same workspace directory.

    The function operates strictly within the workspace sandbox:
    - Input paths are virtual (e.g. `/workspace/foo.zip`)
    - Execution is performed on resolved real paths
    - Output paths are returned in virtual-path form for agent consumption

    Typical agent usage:
    - Call this when a ZIP file is detected during workspace inspection
    - Follow with a tree-view operation on the extracted directory

    Args:
        virtual_zip_path: Virtual path to a ZIP file inside the workspace
            (for example `/workspace/datasets/images.zip`). PATH MUST START WITH `/workspace`.

    Returns:
        A dictionary containing:
        - status: Execution status string
        - zip: The input virtual ZIP path
        - extracted_to: Virtual path of the extracted directory
        - num_files: Number of files listed in the ZIP archive

    Raises:
        FileNotFoundError: If the ZIP file does not exist.
        ValueError: If the provided path does not point to a ZIP archive.
    """

    # zip_path = resolve_workspace_path(virtual_zip_path)

    # if not zip_path.exists():
    #     raise FileNotFoundError(zip_path)

    # if zip_path.suffix.lower() != ".zip":
    #     raise ValueError("Provided file is not a zip archive")

    # output_dir = zip_path.with_suffix("")
    # output_dir.mkdir(parents=True, exist_ok=True)

    # with zipfile.ZipFile(zip_path, "r") as zf:
    #     zf.extractall(output_dir)

    # return {
    #     "status": "ok",
    #     "zip": virtual_zip_path,
    #     "extracted_to": f"/workspace/{output_dir.name}",
    #     "num_files": len(zf.namelist()),
    # }

    zip_path = resolve_workspace_path(virtual_zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    if zip_path.suffix.lower() != ".zip":
        raise ValueError("Provided file is not a zip archive")

    output_dir = zip_path.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()
        for info in infos:
            info.filename = safe_fix_zip_filename(info.filename)
            zf.extract(info, output_dir)

    return {
        "status": "ok",
        "zip": virtual_zip_path,
        "extracted_to": f"/workspace/{output_dir.name}",
        "num_files": len(infos),
    }

@tool(parse_docstring=True)
def tree_view_workspace(
    virtual_path: str,
    max_depth: int = 4,
    max_entries: int = 200,
) -> dict:
    """
    Generate a hierarchical tree view of a workspace directory.

    This preprocessing function inspects the directory structure rooted at a
    given virtual workspace path and produces a readable tree representation.
    It is designed for lightweight structural understanding rather than full
    content analysis.

    The function supports depth and entry limits to prevent excessive output
    and is suitable for agent-driven exploration and reporting.

    Typical agent usage:
    - Inspect workspace contents before deciding on further preprocessing
    - Summarize directory layout in reports such as `/final_report.md`
    - Validate results of prior operations such as unzip or file generation

    Args:
        virtual_path: Virtual workspace path to inspect, such as `/workspace`
            or `/workspace/data` PATH MUST START WITH `/workspace`.
        max_depth: Maximum directory depth to traverse. The default value limits
            traversal to a small number of levels to avoid excessive output.
        max_entries: Maximum total number of files or directories to include
            in the output. The default value prevents large directories from
            producing oversized results.

    Returns:
        A dictionary containing:
        - root: The inspected virtual path
        - max_depth: The depth limit used during traversal
        - entries: Number of directory entries included
        - tree: A newline-separated string representing the directory tree
        - truncated: Whether traversal stopped early due to entry limits

    Raises:
        FileNotFoundError: If the target path does not exist.
    """

    root = resolve_workspace_path(virtual_path)

    if not root.exists():
        raise FileNotFoundError(root)

    lines = []
    count = 0

    def walk(path: Path, prefix: str = "", depth: int = 0):
        nonlocal count
        if depth > max_depth or count >= max_entries:
            return

        for p in sorted(path.iterdir()):
            if count >= max_entries:
                return

            lines.append(f"{prefix}{p.name}")
            count += 1

            if p.is_dir():
                walk(p, prefix + "  ", depth + 1)

    walk(root)

    return {
        "root": virtual_path,
        "max_depth": max_depth,
        "entries": count,
        "tree": "\n".join(lines),
        "truncated": count >= max_entries,
    }

@tool(parse_docstring=True)
def move_workspace_file(source_path: str, destination_path: str) -> dict:
    """
    Move or rename a file/directory within the workspace.

    This tool allows the agent to organize files by moving them into specific
    folders or renaming them. It automatically creates any missing parent
    directories for the destination path.

    Args:
        source_path: The virtual path of the file to move (e.g., "/workspace/data.txt")
        destination_path: The target virtual path (e.g., "/workspace/documents/data.txt")

    Returns:
        A dictionary with the status and new location.
    """
    src_real = resolve_workspace_path(source_path)
    dst_real = resolve_workspace_path(destination_path)

    if not src_real.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    if dst_real.exists():
        raise FileExistsError(f"Destination already exists: {destination_path}")

    # Ensure destination directory exists
    dst_real.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(str(src_real), str(dst_real))

    return {
        "status": "moved",
        "from": source_path,
        "to": destination_path
    }

@tool(parse_docstring=True)
def delete_workspace_file(virtual_path: str) -> dict:
    """
    Safely delete a file by moving it to a .trash folder.

    Instead of permanently deleting files, this tool moves them to a hidden
    '/workspace/.trash' directory. This provides a safety mechanism allowing
    recovery if the agent makes a mistake.

    Args:
        virtual_path: The virtual path of the file to delete (e.g., "/workspace/junk.tmp")

    Returns:
        A dictionary with the status and trash location.
    """
    target_real = resolve_workspace_path(virtual_path)

    if not target_real.exists():
        raise FileNotFoundError(f"File not found: {virtual_path}")

    # Define trash directory
    trash_dir = WORKSPACE_ROOT / ".trash"
    trash_dir.mkdir(exist_ok=True)

    # Create a unique name to prevent overwriting in trash
    # e.g., filename_1708456.txt
    timestamp = int(time.time())
    trash_name = f"{target_real.stem}_{timestamp}{target_real.suffix}"
    trash_path = trash_dir / trash_name

    shutil.move(str(target_real), str(trash_path))

    return {
        "status": "deleted (moved to trash)",
        "original_path": virtual_path,
        "trash_path": f"/workspace/.trash/{trash_name}"
    }
