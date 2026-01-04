from __future__ import annotations

from typing import Optional

from langchain.tools import tool

from smartagent.workspace import resolve_workspace_path

@tool(parse_docstring=True)
def audio_transcribe(
    virtual_audio_path: str,
    server_url: str = "http://127.0.0.1:8080/inference",
    response_format: str = "json",
    temperature: float = 0.0,
    temperature_inc: float = 0.2,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    convert_to_wav: bool = True,
    timeout_sec: float = 300.0,
) -> dict:
    """
    Transcribe an audio file located inside the workspace using whisper.cpp server.

    This preprocessing function sends a local audio file to the whisper.cpp
    HTTP server and returns the transcription results.

    The function operates strictly within the workspace sandbox:
    - Input paths are virtual (e.g. `/workspace/audio.m4a`)
    - Execution is performed on resolved real paths
    - No file content is modified

    Notes:
    - If the input is not a WAV file, this tool can convert it locally using
      ffmpeg (requires ffmpeg on PATH). Disable with convert_to_wav=False to
      rely on server-side --convert or pre-converted WAV input.

    Typical agent usage:
    - Call this when an audio file is detected during workspace inspection
    - Follow with summarization or downstream extraction tasks

    Args:
        virtual_audio_path: Virtual path to an audio file inside the workspace
            (for example `/workspace/recordings/meeting.m4a`).
            PATH MUST START WITH `/workspace`.
        server_url: Whisper server inference endpoint URL.
        response_format: Response format requested from whisper server.
        temperature: Decode temperature.
        temperature_inc: Temperature increment for fallback decoding.
        language: Spoken language code (e.g. en, zh). Use None for auto-detect.
        prompt: Optional initial prompt to bias the transcription.
        convert_to_wav: Convert non-WAV input to 16 kHz mono PCM WAV locally.
        timeout_sec: HTTP request timeout in seconds.

    Returns:
        A dictionary containing:
        - status: Execution status string
        - audio: The input virtual audio path
        - text: Transcribed text if available
        - segments: Segment list if available
        - response_format: The response format used

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the provided path is outside the workspace.
    """

    import httpx
    import shutil
    import subprocess
    import tempfile

    audio_path = resolve_workspace_path(virtual_audio_path)

    if not audio_path.exists():
        return {
            "status": "error",
            "audio": virtual_audio_path,
            "error": f"File not found: {virtual_audio_path}",
        }

    if not audio_path.is_file():
        return {
            "status": "error",
            "audio": virtual_audio_path,
            "error": "Path is not a file.",
        }

    temp_dir: Optional[tempfile.TemporaryDirectory] = None
    upload_path = audio_path
    converted = False

    if audio_path.suffix.lower() != ".wav":
        if not convert_to_wav:
            return {
                "status": "error",
                "audio": virtual_audio_path,
                "error": "Input is not WAV. Enable convert_to_wav or start the server with --convert.",
            }

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return {
                "status": "error",
                "audio": virtual_audio_path,
                "error": "ffmpeg not found on PATH. Install ffmpeg or disable convert_to_wav.",
            }

        temp_dir = tempfile.TemporaryDirectory()
        output_path = Path(temp_dir.name) / f"{audio_path.stem}.wav"
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(audio_path),
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
            temp_dir.cleanup()
            return {
                "status": "error",
                "audio": virtual_audio_path,
                "error": "ffmpeg failed to convert the input to WAV.",
                "details": result.stderr.strip() or "No stderr output.",
            }

        upload_path = output_path
        converted = True

    data = {
        "temperature": str(temperature),
        "temperature_inc": str(temperature_inc),
        "response_format": response_format,
    }
    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt

    content_type = "audio/wav" if upload_path.suffix.lower() == ".wav" else "application/octet-stream"
    try:
        with upload_path.open("rb") as handle:
            files = {"file": (upload_path.name, handle, content_type)}
            with httpx.Client(timeout=timeout_sec) as client:
                response = client.post(server_url, data=data, files=files)
    except httpx.RequestError as exc:
        return {
            "status": "error",
            "audio": virtual_audio_path,
            "error": f"Request failed: {exc}",
        }
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    if response.is_error:
        body = response.text.strip()
        return {
            "status": "error",
            "audio": virtual_audio_path,
            "status_code": response.status_code,
            "reason": response.reason_phrase,
            "error": body or "Server returned an error response.",
        }

    payload: Any
    if response.headers.get("content-type", "").startswith("application/json"):
        payload = response.json()
    else:
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = response.text

    if isinstance(payload, str):
        return {
            "status": "ok",
            "audio": virtual_audio_path,
            "converted": converted,
            "response_format": response_format,
            "text": payload.strip(),
        }

    result = payload
    if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
        result = payload["result"]

    if isinstance(result, dict):
        return {
            "status": "ok",
            "audio": virtual_audio_path,
            "converted": converted,
            "response_format": response_format,
            "text": (result.get("text") or "").strip(),
            "segments": result.get("segments", []),
        }

    return {
        "status": "ok",
        "audio": virtual_audio_path,
        "converted": converted,
        "response_format": response_format,
        "response": payload,
    }

from typing import Dict, Any
import pandas as pd
from langchain.tools import tool
