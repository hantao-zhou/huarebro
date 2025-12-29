from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import httpx
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

DEFAULT_FILE = Path("workspace") / "20250528134121-_______-___-1_1_.m4a"
DEFAULT_URL = "http://127.0.0.1:8080/inference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send audio to whisper.cpp server.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Inference endpoint URL.")
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_FILE,
        help="Path to audio file to transcribe.",
    )
    parser.add_argument(
        "--response-format",
        default="json",
        choices=["json", "text", "srt", "vtt", "tsv"],
        help="Whisper server response format.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Decode temperature.")
    parser.add_argument(
        "--temperature-inc",
        type=float,
        default=0.2,
        help="Temperature increment for fallback decoding.",
    )
    parser.add_argument("--language", default=None, help="Spoken language (e.g. en).")
    parser.add_argument("--prompt", default=None, help="Optional initial prompt.")
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Do not convert non-WAV input to WAV before upload.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds.",
    )
    return parser.parse_args()


def format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "-"
    try:
        total_ms = int(round(float(value) * 1000))
    except (TypeError, ValueError):
        return "-"
    hours, rem = divmod(total_ms, 3600 * 1000)
    minutes, rem = divmod(rem, 60 * 1000)
    seconds, millis = divmod(rem, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def extract_result(payload: Any) -> Any:
    if isinstance(payload, dict):
        if "text" in payload or "segments" in payload:
            return payload
        if isinstance(payload.get("result"), dict):
            return payload["result"]
    return payload


def build_segments_table(segments: Iterable[Dict[str, Any]]) -> Table:
    table = Table(title="Segments", show_lines=False)
    table.add_column("#", style="cyan", justify="right", no_wrap=True)
    table.add_column("Start", style="green", no_wrap=True)
    table.add_column("End", style="green", no_wrap=True)
    table.add_column("Text", style="white")
    for idx, segment in enumerate(segments, start=1):
        start = format_seconds(segment.get("start"))
        end = format_seconds(segment.get("end"))
        text = (segment.get("text") or "").strip()
        table.add_row(str(idx), start, end, text)
    return table


def print_result(console: Console, response: httpx.Response, elapsed: float) -> None:
    summary = Text()
    summary.append("Status: ", style="bold")
    summary.append(f"{response.status_code} {response.reason_phrase}\n")
    summary.append("Elapsed: ", style="bold")
    summary.append(f"{elapsed:.2f}s")
    console.print(Panel(summary, title="Request Summary", expand=False))

    if response.headers.get("content-type", "").startswith("application/json"):
        payload = response.json()
    else:
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = response.text

    if isinstance(payload, str):
        console.print(Panel(payload.strip(), title="Transcript", expand=False))
        return

    result = extract_result(payload)
    if isinstance(result, dict):
        text = result.get("text")
        if text:
            console.print(Panel(Text(text.strip()), title="Transcript", expand=False))
        segments = result.get("segments")
        if isinstance(segments, list) and segments:
            console.print(build_segments_table(segments))
            return

    console.print(Panel(JSON.from_data(payload), title="Response JSON", expand=False))


def prepare_audio(
    console: Console, file_path: Path, allow_convert: bool
) -> tuple[Optional[Path], Optional[tempfile.TemporaryDirectory]]:
    if file_path.suffix.lower() == ".wav":
        return file_path, None
    if not allow_convert:
        console.print(
            Panel(
                "Input is not WAV. Enable local conversion or start the server with --convert.",
                title="Unsupported Format",
                expand=False,
            )
        )
        return None, None

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        console.print(
            Panel(
                "ffmpeg not found in PATH. Install ffmpeg or run whisper-server with --convert.",
                title="Missing Dependency",
                expand=False,
            )
        )
        return None, None

    temp_dir = tempfile.TemporaryDirectory()
    output_path = Path(temp_dir.name) / f"{file_path.stem}.wav"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(file_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        console.print(
            Panel(
                f"ffmpeg failed to convert the input.\n\n{result.stderr.strip()}",
                title="Conversion Error",
                expand=False,
            )
        )
        temp_dir.cleanup()
        return None, None

    console.print(
        Panel(
            f"Converted to WAV for upload:\n{output_path}",
            title="Preparing Audio",
            expand=False,
        )
    )
    return output_path, temp_dir


def main() -> int:
    args = parse_args()
    console = Console()
    file_path = args.file.expanduser()
    if not file_path.is_file():
        console.print(
            Panel(
                f"[red]File not found:[/red]\n{file_path}",
                title="Error",
                expand=False,
            )
        )
        return 2

    upload_path, temp_dir = prepare_audio(console, file_path, not args.no_convert)
    if upload_path is None:
        return 2

    data = {
        "temperature": str(args.temperature),
        "temperature_inc": str(args.temperature_inc),
        "response_format": args.response_format,
    }
    if args.language:
        data["language"] = args.language
    if args.prompt:
        data["prompt"] = args.prompt

    panel_lines = [
        f"Source: {file_path}",
        f"Upload: {upload_path}",
        f"URL: {args.url}",
        f"Format: {args.response_format}",
    ]
    console.print(Panel("\n".join(panel_lines), title="Whisper ASR", expand=False))

    content_type = "audio/wav" if upload_path.suffix.lower() == ".wav" else "application/octet-stream"
    try:
        with upload_path.open("rb") as handle:
            files = {"file": (upload_path.name, handle, content_type)}
            with httpx.Client(timeout=args.timeout) as client:
                start = time.perf_counter()
                response = client.post(args.url, data=data, files=files)
                elapsed = time.perf_counter() - start
    except httpx.RequestError as exc:
        console.print(
            Panel(f"[red]Request failed:[/red]\n{exc}", title="Network Error", expand=False)
        )
        return 1
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    if response.is_error:
        body = response.text.strip() or "(empty response body)"
        console.print(
            Panel(
                body,
                title=f"HTTP {response.status_code} {response.reason_phrase}",
                expand=False,
            )
        )
        return 1

    print_result(console, response, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
