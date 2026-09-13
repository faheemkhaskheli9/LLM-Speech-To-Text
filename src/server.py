# server.py
import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from websockets import connect as ws_connect
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

# Resolved relative to this file (not the process cwd) so `uvicorn
# server:app` works the same whether launched from src/ or the repo root.
INDEX_HTML = Path(__file__).resolve().parent / "index.html"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Realtime transcription session (note the intent=transcription)
OPENAI_REALTIME_WS = "wss://api.openai.com/v1/realtime?intent=transcription"

# choose a realtime-capable transcribe model (OpenAI docs list current names)
REALTIME_MODEL = "gpt-4o-transcribe"

def _resolve_cors_origins() -> tuple[list[str], bool]:
    """Restricted-by-default CORS origins, overridable via CORS_ALLOW_ORIGINS.

    Default is localhost-only so this isn't shipped wide-open by accident.
    A comma-separated env var overrides it (e.g. for a deployed frontend's
    real origin). The literal wildcard "*" is only honored via that same
    explicit opt-in -- and per the CORS spec a wildcard origin can't be
    combined with credentials (a browser rejects it), so credentials are
    disabled automatically when the wildcard is chosen.
    """
    raw = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
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


async def openai_headers():
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "OpenAI-Beta": "realtime=v1"}


@app.websocket("/ws/transcribe")
async def ws_transcribe(client_ws: WebSocket):
    """
    1) Accept client WebSocket
    2) Connect to OpenAI Realtime WS (intent=transcription)
    3) Relay audio chunks & control messages
    4) Send transcripts back to client
    """
    await client_ws.accept()
    if not OPENAI_API_KEY:
        await client_ws.send_json(
            {"type": "error", "message": "OPENAI_API_KEY not set"}
        )
        await client_ws.close()
        return

    try:
        async with ws_connect(
            OPENAI_REALTIME_WS, extra_headers=await openai_headers()
        ) as openai_ws:
            # Create a new response stream for transcription output
            # (We’ll trigger response creation after we start receiving audio)
            # Some setups auto-start; this explicit request keeps behavior predictable.
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
                            "include": ["item.input_audio_transcription.logprobs"],
                        },
                    }
                )
            )

            # Task A: forward messages from client -> OpenAI
            async def pump_client_to_openai():
                """
                Expected from client:
                - {"type":"start"}           -> optional: begins a user turn
                - {"type":"audio","b64":...} -> PCM16 mono 16k, base64
                - {"type":"commit"}          -> finalize current buffer (VAD stop or user stop)
                - {"type":"stop"}            -> request a response now
                - {"type":"end"}             -> end session
                """
                while True:
                    msg = await client_ws.receive_text()
                    data = json.loads(msg)
                    # print(data)
                    t = data.get("type")

                    if t == "audio":
                        # Append audio chunk to the current input buffer
                        await openai_ws.send(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": data["b64"],  # base64-encoded PCM16
                                }
                            )
                        )

            # Task B: forward events from OpenAI -> client
            async def pump_openai_to_client():
                """
                We forward relevant response events back to client.
                Typical events:
                  - response.output_text.delta / completed (streamed transcripts)
                  - response.completed
                  - error
                """
                while True:
                    raw = await openai_ws.recv()
                    print(raw)
                    try:
                        event = json.loads(raw)
                    except Exception:
                        # Keep raw as string (defensive)
                        event = {"type": "raw", "data": raw}

                    etype = event.get("type", "")
                    # Relay everything; the client filters what it needs
                    await client_ws.send_json(event)

            await asyncio.gather(pump_client_to_openai(), pump_openai_to_client())

    except (WebSocketDisconnect, ConnectionClosedOK):
        pass
    except ConnectionClosedError as e:
        logging.exception("OpenAI WS closed with error: %s", e)
        try:
            await client_ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    except Exception as e:
        logging.exception("Server error: %s", e)
        try:
            await client_ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        await client_ws.close()
