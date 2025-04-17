A single Daily room will deliver raw RTP to your EC2‑hosted Pipecat server; Deepgram converts speech to text and a Groq‑powered Llama stream voices quick replies, while a slower Gemini‑2.5 Pro branch analyses the same transcript and occasionally injects higher‑level suggestions.  Cartesia TTS streams the final audio back through Daily.  FastAPI manages pipeline startup and a minimal REST + WebSocket API; Streamlit embeds the Daily iframe and renders live captions.  Two PostgreSQL tables—one raw “turns” log and one JSON indicator record—persist everything for analytics.  The skeleton below shows only file stubs and placeholder hooks, so you can expand each piece without boilerplate overhead.

---

## 1  Transport & speech layer (Daily → Deepgram)

Daily’s Prebuilt widget (or JS‑SDK) joins a room that the server created via Daily’s REST API; the media stream arrives over `SmallWebRTCTransport` citeturn0search0.  Twilio dial‑in can reuse Daily’s SIP gateway if needed citeturn0search1.  Frames reach Deepgram’s real‑time STT endpoint, which integrates cleanly with FastAPI’s asyncio loop citeturn0search2.

---

## 2  LLM branches

* **Fast branch** – GroqCloud’s Llama‑3/70B for sub‑150 ms reasoning citeturn0search3.  
* **Smart branch** – Gemini‑2.5 Pro REST API for deeper, slower analysis citeturn0search4.  
* Both branches finish with **Cartesia TTS** streaming audio chunks in < 300 ms citeturn0search5.

---

## 3  Project skeleton

```text
voice_ai_mvp/
├── server/
│   ├── main.py              # FastAPI entry & ngrok uplift
│   ├── pipeline.py          # Pipecat ParallelPipeline stub
│   ├── models.py            # SQLAlchemy ORM (JSONB columns)
│   ├── deps.py              # load env, create Daily room token
│   ├── requirements.txt
│   └── Dockerfile
├── client/
│   ├── app.py               # Streamlit UI + WS captions
│   └── requirements.txt
└── README.md
```

*Each folder compiles in ≈ 30 lines of actual code to get the loop running.*

---

### 3.1 `server/main.py` (outline)

```python
from fastapi import FastAPI, WebSocket
from pipeline import build_pipeline
from models import SessionLocal
import ngrok, asyncio

app = FastAPI()
pipeline = build_pipeline()

@app.on_event("startup")
async def startup():
    ngrok.connect(8000)            # expose HTTPS automatically citeturn0search8
    await pipeline.start()

@app.websocket("/ws/captions")
async def captions(ws: WebSocket):
    await ws.accept()
    async for turn in pipeline.caption_stream():
        await ws.send_json(turn)
```

WebSockets come “for free” in FastAPI because it inherits Starlette’s primitives citeturn0search6.

### 3.2 `server/pipeline.py` (stub)

```python
def build_pipeline():
    # create transport = SmallWebRTCTransport(daily_domain, token)
    # define Branch A: Deepgram → Llama (Groq) → Cartesia TTS
    # define Branch B: Deepgram → Gemini‑2.5‑Pro → Cartesia TTS
    # share suggestions via Producer/Consumer
    return Pipeline([...])
```

---

## 4  Streamlit client skeleton

```python
import streamlit as st, components
DAILY_URL = st.secrets["DAILY_EMBED"]
components.iframe(DAILY_URL, width=800, height=600)   # embed room citeturn0search7

st.header("Live transcript")
spot = st.empty()

# WebSocket → captions
from websocket import WebSocketApp, enableTrace
def on_msg(ws, msg): spot.markdown(msg)
enableTrace(False)
ws = WebSocketApp("wss://<ngrok-id>.ngrok.io/ws/captions", on_message=on_msg)
ws.run_forever()
```

Streamlit’s iframe is enough for MVP; you can switch to Daily’s React SDK later.  Captions arrive through the FastAPI WS relay.

---

## 5  Data model

```python
# server/models.py
class Turn(Base):
    __tablename__ = "turns"
    id = Column(UUID, primary_key=True)
    call_id = Column(UUID, index=True)
    speaker = Column(String)               # user | llama-fast | gemini-smart
    text = Column(JSONB)                   # raw + normalised citeturn0search9
    ts = Column(TIMESTAMP)

class Indicator(Base):
    __tablename__ = "indicators"
    call_id = Column(UUID, primary_key=True)
    meta = Column(JSONB)                   # keys listed below citeturn0search10
```

`Indicator.meta` stores booleans/strings for:

```
distributor, credit_fund_aware, invests_1cr, knows_maneesh,
sophisticated, optimism, wants_zoom, follow_up, sales_referral,
english_proficiency, name, number, occupation, city
```

JSONB lets you query (`?`) each key or index popular ones with GIN.

---

## 6  Environment & deployment

* **EC2** t3.medium in the same AWS region as Deepgram edge to minimise RTT.  
* Single Docker image runs uvicorn + pipeline; ngrok provides public TLS URL for Daily widget and Streamlit dev testing.  
* `.env` stores API keys for Daily, Deepgram, Groq, Gemini, Cartesia, Postgres URL.

---

### Minimal deployment command

```bash
docker build -t voice-mvp ./server
docker run -d -p 8000:8000 --env-file .env voice-mvp
streamlit run client/app.py --server.port 8501
```

Expose both ports with ngrok if you need public endpoints during development.

---

With this skeleton every moving part—Daily transport, dual‑LLM reasoning, Cartesia TTS, FastAPI wiring, Streamlit embed, and PostgreSQL JSONB persistence—is stubbed and ready for incremental implementation, keeping the repository small yet production‑shaped.