import asyncio
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List

import requests
import streamlit as st
import websocket
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")
WS_URL = os.getenv("WS_URL", "ws://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Smart Voice Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom styles for captions
st.markdown("""
<style>
.user-caption {
    background-color: #f0f2f6;
    border-radius: 8px;
    padding: 10px;
    margin: 5px 0;
    border-left: 4px solid #2196F3;
}

.llama-caption {
    background-color: #e6f7ff;
    border-radius: 8px;
    padding: 10px;
    margin: 5px 0;
    border-left: 4px solid #4CAF50;
}

.gemini-caption {
    background-color: #fff8e1;
    border-radius: 8px;
    padding: 10px;
    margin: 5px 0;
    border-left: 4px solid #FF9800;
}

.caption-time {
    font-size: 12px;
    color: #666;
    margin-bottom: 5px;
}

.caption-text {
    font-size: 16px;
}

.suggestion {
    background-color: #fce4ec;
    border-radius: 8px;
    padding: 10px;
    margin: 10px 0;
    border-left: 4px solid #E91E63;
}

iframe {
    border-radius: 10px;
    border: 1px solid #ddd;
}

.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.connection-status {
    font-size: 14px;
    padding: 5px 10px;
    border-radius: 15px;
}

.connected {
    background-color: #e8f5e9;
    color: #2e7d32;
}

.disconnected {
    background-color: #ffebee;
    color: #c62828;
}
</style>
""", unsafe_allow_html=True)


class CaptionManager:
    def __init__(self):
        self.captions: List[Dict[str, Any]] = []
        self.ws = None
        self.call_id = None
        self.connected = False
    
    def connect_to_websocket(self, call_id: str):
        """Connect to the WebSocket for captions"""
        self.call_id = call_id
        
        # Close any existing connection
        if self.ws:
            self.ws.close()
            
        # Define callbacks
        def on_message(ws, message):
            data = json.loads(message)
            if data.get("type") == "connection_established":
                self.connected = True
                return
                
            # Add timestamp if not present
            if "ts" not in data:
                data["ts"] = time.time()
                
            # Add to captions
            self.captions.append(data)
            
            # Update the UI to show the new caption
            st.session_state.update_captions = True
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
            self.connected = False
            
        def on_close(ws, close_status_code, close_msg):
            print(f"WebSocket closed: {close_status_code} - {close_msg}")
            self.connected = False
            
        def on_open(ws):
            print(f"WebSocket connected for call {call_id}")
            
        # Create WebSocket connection
        ws_endpoint = f"{WS_URL}/ws/captions/{call_id}"
        self.ws = websocket.WebSocketApp(
            ws_endpoint,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Start WebSocket connection in a background thread
        import threading
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
    
    def format_caption(self, caption: Dict[str, Any]) -> str:
        """Format a caption for display"""
        speaker_name = "You" if caption.get("speaker") == "user" else caption.get("speaker", "Bot")
        
        # Format timestamp
        timestamp = caption.get("ts", time.time())
        if isinstance(timestamp, str):
            try:
                timestamp = float(timestamp)
            except ValueError:
                timestamp = time.time()
                
        time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
        
        # Determine CSS class
        css_class = caption.get("class", "user-caption" if speaker_name == "You" else "llama-caption")
        
        # Format special suggestion
        is_suggestion = caption.get("metadata", {}).get("suggestion", False)
        if is_suggestion:
            css_class = "suggestion"
        
        # Create HTML for the caption
        html = f"""
        <div class="{css_class}">
            <div class="caption-time">{speaker_name} • {time_str}</div>
            <div class="caption-text">{caption.get("text", "")}</div>
        </div>
        """
        return html
    
    def close(self):
        """Close the WebSocket connection"""
        if self.ws:
            self.ws.close()
            self.ws = None
            self.connected = False


def create_room() -> Dict[str, Any]:
    """Connect to the API to create a room"""
    try:
        response = requests.post(f"{API_URL}/connect")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error connecting to server: {e}")
        return {}


def main():
    # Initialize session state
    if "caption_manager" not in st.session_state:
        st.session_state.caption_manager = CaptionManager()
    
    if "connected" not in st.session_state:
        st.session_state.connected = False
        
    if "room_info" not in st.session_state:
        st.session_state.room_info = None
        
    if "update_captions" not in st.session_state:
        st.session_state.update_captions = False
    
    # Header
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("Smart Voice Bot")
        st.markdown("A dual-LLM voice agent with parallel processing pipelines.")
    
    with col2:
        status_class = "connected" if st.session_state.connected else "disconnected"
        status_text = "Connected" if st.session_state.connected else "Disconnected"
        st.markdown(f'<div class="connection-status {status_class}">{status_text}</div>', unsafe_allow_html=True)
        
        # Connect button
        if not st.session_state.connected:
            if st.button("Connect to Voice Bot"):
                with st.spinner("Creating room..."):
                    room_info = create_room()
                    
                    if room_info and "room_url" in room_info and "token" in room_info:
                        st.session_state.room_info = room_info
                        st.session_state.connected = True
                        
                        # Connect to captions WebSocket
                        st.session_state.caption_manager.connect_to_websocket(room_info["call_id"])
                        
                        st.success("Connected! Join the voice room below.")
                        st.rerun()
        else:
            if st.button("Disconnect"):
                st.session_state.caption_manager.close()
                st.session_state.connected = False
                st.session_state.room_info = None
                st.rerun()
    
    # Main content area
    if st.session_state.connected and st.session_state.room_info:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Voice Conversation")
            
            # Display Daily iframe
            room_url = st.session_state.room_info["room_url"]
            token = st.session_state.room_info["token"]
            daily_url = f"{room_url}?token={token}"
            
            # Embed Daily iframe - FIXED: changed width from "100%" to 800
            st.components.v1.iframe(
                src=daily_url,
                width=800,
                height=400,
                scrolling=False
            )
            
            # Show room details
            with st.expander("Room Details"):
                st.code(json.dumps(st.session_state.room_info, indent=2))
        
        with col2:
            st.subheader("Live Transcript")
            
            # Display captions
            captions_container = st.container()
            
            # Auto-update captions
            if st.session_state.update_captions or True:
                with captions_container:
                    for caption in reversed(st.session_state.caption_manager.captions[-10:]):
                        st.markdown(
                            st.session_state.caption_manager.format_caption(caption), 
                            unsafe_allow_html=True
                        )
                st.session_state.update_captions = False
    
    else:
        # Welcome message when not connected
        st.info("Click 'Connect to Voice Bot' to start a new conversation.")
        
        # Example description
        st.markdown("""
        ### How it works
        
        This voice bot uses a dual-branch processing architecture:
        
        1. **Fast Branch** - Groq's Llama model provides quick responses (<150ms)
        2. **Smart Branch** - Gemini processes the same inputs for deeper insights
        
        The smart branch shares suggestions with the fast branch, which incorporates
        them into its responses. This creates a more intelligent conversation experience
        with the speed of the fast model and the intelligence of the slower model.
        """)


if __name__ == "__main__":
    main()