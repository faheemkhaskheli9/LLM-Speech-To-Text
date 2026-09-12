# LLM Speech-To-Text

A minimal realtime speech-to-text demo: a browser page captures microphone
audio and streams it to a small FastAPI server, which relays it to OpenAI's
Realtime transcription API and streams the transcript back — a two-hop
WebSocket proxy, not a direct browser-to-OpenAI connection (keeping the
API key server-side only).

## Architecture

```text
Browser (src/index.html)          Server (src/server.py)              OpenAI
--------------------------        ------------------------------      -------------------------------
getUserMedia() -> PCM16   --ws--> /ws/transcribe (FastAPI)    --ws--> wss://api.openai.com/v1/realtime
   audio chunks, base64            forwards {"type":"audio"}           ?intent=transcription
                                    messages to OpenAI

transcript deltas          <--ws-- relays every OpenAI event   <--ws-- transcription deltas/completed
   rendered live                    back to the browser                events
```

1. `src/index.html` opens a WebSocket to `/ws/transcribe`, captures mic
   audio via the Web Audio API, downsamples it to 16kHz mono PCM16, and
   sends each chunk as `{"type": "audio", "b64": <base64 PCM16>}`.
2. `src/server.py` accepts that connection, opens a second WebSocket to
   OpenAI's Realtime API (`intent=transcription`) authenticated with
   `OPENAI_API_KEY`, and pumps messages both directions:
   client audio -> OpenAI, OpenAI transcript events -> client.
3. The client filters for
   `conversation.item.input_audio_transcription.delta`/`completed` events
   and appends the text to the live transcript div.

The server never exposes `OPENAI_API_KEY` to the browser — only the relayed
transcript events cross that boundary.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in your real OPENAI_API_KEY
export OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d= -f2)   # or use python-dotenv/direnv
uvicorn src.server:app --reload
```

Then open http://127.0.0.1:8000/ in a browser, click **Start**, and speak —
the live transcript appears under "Live Transcript". Click **Stop** to end
the session.

Requires a real `OPENAI_API_KEY` with access to a realtime-capable
transcription model (`gpt-4o-transcribe`); without one, the WebSocket
handshake completes but the server immediately sends
`{"type": "error", "message": "OPENAI_API_KEY not set"}` and closes.

## Tests

```bash
pytest tests/
```

`tests/` covers the parts that don't require a live OpenAI connection or a
real microphone (both out of scope for CI): the HTTP routes the server
exposes and that the setup docs above stay in sync with the actual code
(README/`.env.example` referencing the right variable name, `index.html`
existing where the server expects it).
