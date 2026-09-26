# server.py
import asyncio
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from websockets import connect as ws_connect
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

logger = logging.getLogger(__name__)

# Resolved relative to this file (not the process cwd) so `uvicorn
# server:app` works the same whether launched from src/ or the repo root.
INDEX_HTML = Path(__file__).resolve().parent / "index.html"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_REALTIME_WS = "wss://api.openai.com/v1/realtime?intent=transcription"
REALTIME_MODEL = "gpt-4o-transcribe"


def _resolve_cors_origins() -> tuple[list[str], bool]:
    """Restricted-by-default CORS origins, overridable via CORS_ALLOW_ORIGINS."""
    raw = os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if origins == ["*"]:
        return ["*"], False
    return origins, True


CORS_ALLOW_ORIGINS, CORS_ALLOW_CREDENTIALS = _resolve_cors_origins()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return FileResponse(INDEX_HTML)


@app.get("/health")
async def health():
    """Liveness probe: the FastAPI process is running."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness probe: required runtime configuration is available."""
    if not OPENAI_API_KEY:
        return {"status": "not_ready", "reason": "OPENAI_API_KEY not set"}
    return {"status": "ready"}


async def openai_headers():
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }


async def _cancel_task(task: asyncio.Task) -> None:
    """Cancel a relay task and consume the cancellation cleanly."""
    if task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@app.websocket("/ws/transcribe")
async def ws_transcribe(client_ws: WebSocket):
    """Proxy browser microphone audio to OpenAI Realtime transcription."""
    await client_ws.accept()
    if not OPENAI_API_KEY:
        await client_ws.send_json(
            {"type": "error", "message": "OPENAI_API_KEY not set"}
        )
        await client_ws.close()
        return

    try:
        async with ws_connect(
            OPENAI_REALTIME_WS,
            extra_headers=await openai_headers(),
        ) as openai_ws:
            await openai_ws.send(
                json.dumps(
                    {
                        "type": "transcription_session.update",
                        "session": {
                            "input_audio_format": "pcm16",
                            "input_audio_transcription": {
                                "model": REALTIME_MODEL,
                                "prompt": "",
                                "language": "en",
                            },
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.5,
                                "prefix_padding_ms": 300,
                                "silence_duration_ms": 500,
                            },
                            "input_audio_noise_reduction": {"type": "near_field"},
                            "include": [
                                "item.input_audio_transcription.logprobs"
                            ],
                        },
                    }
                )
            )

            async def pump_client_to_openai():
                audio_since_commit = False

                while True:
                    msg = await client_ws.receive_text()
                    try:
                        data = json.loads(msg)
                    except json.JSONDecodeError:
                        await client_ws.send_json(
                            {
                                "type": "error",
                                "message": "Invalid JSON message",
                            }
                        )
                        continue

                    message_type = data.get("type")

                    if message_type == "audio":
                        audio_b64 = data.get("b64")
                        if not isinstance(audio_b64, str) or not audio_b64:
                            await client_ws.send_json(
                                {
                                    "type": "error",
                                    "message": "audio message requires non-empty b64",
                                }
                            )
                            continue

                        await openai_ws.send(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": audio_b64,
                                }
                            )
                        )
                        audio_since_commit = True

                    elif message_type == "commit":
                        if audio_since_commit:
                            await openai_ws.send(
                                json.dumps({"type": "input_audio_buffer.commit"})
                            )
                            audio_since_commit = False

                    elif message_type in {"stop", "end"}:
                        if audio_since_commit:
                            await openai_ws.send(
                                json.dumps({"type": "input_audio_buffer.commit"})
                            )
                        return

                    elif message_type == "start":
                        # The upstream transcription session is already initialized.
                        continue

                    else:
                        await client_ws.send_json(
                            {
                                "type": "error",
                                "message": f"Unsupported message type: {message_type}",
                            }
                        )

            async def pump_openai_to_client():
                while True:
                    raw = await openai_ws.recv()
                    try:
                        event = json.loads(raw)
                    except (TypeError, json.JSONDecodeError):
                        event = {"type": "raw", "data": raw}
                    await client_ws.send_json(event)

            client_task = asyncio.create_task(pump_client_to_openai())
            openai_task = asyncio.create_task(pump_openai_to_client())

            done, pending = await asyncio.wait(
                {client_task, openai_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                exc = task.exception()
                if exc:
                    raise exc

            for task in pending:
                await _cancel_task(task)

    except (WebSocketDisconnect, ConnectionClosedOK):
        pass
    except ConnectionClosedError as exc:
        logger.exception("OpenAI WebSocket closed with error: %s", exc)
        try:
            await client_ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    except Exception as exc:
        logger.exception("Transcription server error: %s", exc)
        try:
            await client_ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
        try:
            await client_ws.close()
        except Exception:
            pass
