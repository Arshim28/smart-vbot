import argparse
import asyncio
import os
import subprocess
import json
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from runner import create_room_with_token
from pipecat.transports.services.helpers.daily_rest import DailyRoomParams

# Load environment variables
load_dotenv(override=True)

# Global variables
bot_processes = {}
caption_queues = {}


class CaptionMessage(BaseModel):
    call_id: str  
    speaker: str  # user | llama-fast | gemini-smart
    text: str
    ts: float
    final: bool = True
    metadata: Optional[Dict[str, Any]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create aiohttp session for the app
    app.state.session = aiohttp.ClientSession()
    yield
    # Clean up
    for pid, (proc, _) in bot_processes.items():
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception as e:
            print(f"Error terminating process {pid}: {e}")
    
    # Close aiohttp session
    await app.state.session.close()


# Create FastAPI app
app = FastAPI(title="Smart Voice Bot API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add a simple root endpoint for health checks
@app.get("/")
async def root():
    return {"status": "ok", "message": "Smart Voice Bot API is running"}


@app.post("/connect")
async def connect() -> Dict[str, str]:
    """Create a new room and start a voice bot process.
    
    Returns:
        Dict with room_url and token for connecting to the Daily room
    """
    try:
        # Create a new Daily room and token
        room_params = DailyRoomParams(
            enable_chat=True,
            start_audio_off=False,
            start_video_off=True,
        )
        room_url, token = await create_room_with_token(app.state.session, room_params)
        
        # Start the bot process
        cmd = f"python main.py -u {room_url} -t {token}"
        
        proc = subprocess.Popen(
            cmd.split(),
            cwd=os.getcwd(),
            env=os.environ.copy(),
        )
        
        # Store process information
        call_id = room_url.split("/")[-1]
        bot_processes[proc.pid] = (proc, call_id)
        
        # Create caption queue for this call
        caption_queues[call_id] = asyncio.Queue()
        
        return {"room_url": room_url, "token": token, "call_id": call_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start voice bot: {str(e)}")


@app.post("/captions/{call_id}")
async def add_caption(call_id: str, caption: CaptionMessage):
    """Add a caption to the queue for a specific call.
    
    Args:
        call_id: ID of the call
        caption: Caption message
    """
    if call_id not in caption_queues:
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")
    
    await caption_queues[call_id].put(caption)
    return {"status": "ok"}


@app.websocket("/ws/captions/{call_id}")
async def captions_websocket(websocket: WebSocket, call_id: str):
    """WebSocket endpoint for streaming captions.
    
    Args:
        websocket: WebSocket connection
        call_id: ID of the call to stream captions for
    """
    await websocket.accept()
    
    # Create queue if it doesn't exist
    if call_id not in caption_queues:
        caption_queues[call_id] = asyncio.Queue()
    
    # Send initial connection message
    await websocket.send_json({
        "type": "connection_established",
        "call_id": call_id,
        "message": "Connected to caption stream"
    })
    
    try:
        while True:
            # Get the next caption from the queue
            caption = await caption_queues[call_id].get()
            
            # Format caption for client
            caption_data = caption.dict()
            
            # Add speaker class for styling
            if caption.speaker == "user":
                caption_data["class"] = "user-caption"
            elif "gemini" in caption.speaker:
                caption_data["class"] = "gemini-caption"
            elif "llama" in caption.speaker or "groq" in caption.speaker:
                caption_data["class"] = "llama-caption"
            
            # Send to the client
            await websocket.send_json(caption_data)
    
    except WebSocketDisconnect:
        print(f"Client disconnected from caption stream for call {call_id}")
    except Exception as e:
        print(f"Error in caption WebSocket for call {call_id}: {e}")


def main():
    """Start the FastAPI server."""
    parser = argparse.ArgumentParser(description="Smart Voice Bot Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()