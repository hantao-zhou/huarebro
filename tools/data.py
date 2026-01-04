from __future__ import annotations

from langchain.tools import tool

from smartagent.workspace import resolve_workspace_path

@tool(parse_docstring=True)
def excel_schema_reader(virtual_excel_path: str) -> dict:
    """
    Inspect an Excel file located inside the workspace and return its schema.

    This preprocessing function reads an Excel file referenced by a virtual
    workspace path and extracts high-level structural information, including
    sheet names, row counts, and column metadata.

    The function operates strictly within the workspace sandbox:
    - Input paths are virtual (e.g. `/workspace/data.xlsx`)
    - Execution is performed on resolved real paths
    - No file content is modified

    Typical agent usage:
    - Call this before extracting rows from an Excel file
    - Use the schema to decide which sheet and columns to target
    - Provide structural summaries in reports

    Args:
        virtual_excel_path: Virtual path to an Excel file inside the workspace
            (for example `/workspace/tables/data.xlsx`).
            PATH MUST START WITH `/workspace`.

    Returns:
        A dictionary containing:
        - status: Execution status string
        - excel: The input virtual Excel path
        - sheets: Mapping from sheet name to schema information:
            - num_rows: Number of rows in the sheet
            - columns: List of column descriptors with name and dtype

    Raises:
        FileNotFoundError: If the Excel file does not exist.
        ValueError: If the file is not a valid Excel document.
    """

    excel_path = resolve_workspace_path(virtual_excel_path)

    if not excel_path.exists():
        raise FileNotFoundError(excel_path)

    try:
        sheets = pd.read_excel(excel_path, sheet_name=None)
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}")

    schema = {}

    for sheet_name, df in sheets.items():
        schema[sheet_name] = {
            "num_rows": len(df),
            "columns": [
                {
                    "name": col,
                    "dtype": str(df[col].dtype),
                }
                for col in df.columns
            ],
        }

    return {
        "status": "ok",
        "excel": virtual_excel_path,
        "sheets": schema,
    }

from typing import Optional, List, Dict, Any

@tool(parse_docstring=True)
def excel_entry_extractor(
    virtual_excel_path: str,
    sheet_name: str,
    columns: Optional[List[str]] = None,
    max_rows: int = 50,
    filters: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Extract structured row entries from a specific sheet of an Excel file.

    This preprocessing function loads a specified sheet from an Excel file
    located inside the workspace and returns row-level data as structured
    dictionaries. Optional column selection and exact-match filtering are
    supported.

    The function operates strictly within the workspace sandbox:
    - Input paths are virtual (e.g. `/workspace/data.xlsx`)
    - Execution is performed on resolved real paths
    - Returned data is JSON-serializable for agent consumption

    Typical agent usage:
    - Extract tabular records for reasoning or summarization
    - Retrieve example rows for inspection
    - Perform simple rule-based filtering before LLM analysis

    Args:
        virtual_excel_path: Virtual path to an Excel file inside the workspace
            (for example `/workspace/tables/data.xlsx`).
            PATH MUST START WITH `/workspace`.
        sheet_name: Name of the Excel sheet to extract rows from.
        columns: Optional list of column names to return.
            If omitted, all columns are included.
        max_rows: Maximum number of rows to return.
        filters: Optional exact-match filters in the form `{column_name: value}`. Filters are applied sequentially.

    Returns:
        A dictionary containing:
        - status: Execution status string
        - excel: The input virtual Excel path
        - sheet: The targeted sheet name
        - rows_returned: Number of rows returned
        - entries: List of row dictionaries

    """

    excel_path = resolve_workspace_path(virtual_excel_path)

    if not excel_path.exists():
        raise FileNotFoundError(excel_path)

    sheets = pd.read_excel(excel_path, sheet_name=None)

    if sheet_name not in sheets:
        raise ValueError(f"Sheet not found: {sheet_name}")

    df = sheets[sheet_name]

    if filters:
        for col, val in filters.items():
            if col in df.columns:
                df = df[df[col] == val]

    if columns:
        df = df[columns]

    df = df.head(max_rows)

    entries = df.to_dict(orient="records")

    return {
        "status": "ok",
        "excel": virtual_excel_path,
        "sheet": sheet_name,
        "rows_returned": len(entries),
        "entries": entries,
    }
