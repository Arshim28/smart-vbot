import argparse
import os
from typing import Optional, Tuple

import aiohttp
from dotenv import load_dotenv

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams

# Load environment variables
load_dotenv(override=True)

# Constants
DEFAULT_DAILY_API_URL = "https://api.daily.co/v1"
DEFAULT_TOKEN_EXPIRY = 60 * 60  # 1 hour


async def configure(session: aiohttp.ClientSession) -> Tuple[str, str]:
    """Get or create a Daily room URL and token.
    
    Args:
        session: Active aiohttp client session
        
    Returns:
        Tuple of (room_url, token)
    """
    parser = argparse.ArgumentParser(description="Smart Voice Bot")
    parser.add_argument("-u", "--url", type=str, help="Daily room URL")
    parser.add_argument("-k", "--apikey", type=str, help="Daily API key")
    parser.add_argument("-t", "--token", type=str, help="Daily room token")
    
    args, _ = parser.parse_known_args()
    
    # Get room URL from args or environment
    room_url = args.url or os.getenv("DAILY_ROOM_URL")
    api_key = args.apikey or os.getenv("DAILY_API_KEY")
    token = args.token
    
    if not api_key:
        raise ValueError(
            "No Daily API key provided. Use --apikey flag or set DAILY_API_KEY environment variable."
        )
    
    # Create REST helper for Daily API
    rest_helper = DailyRESTHelper(
        daily_api_key=api_key,
        daily_api_url=os.getenv("DAILY_API_URL", DEFAULT_DAILY_API_URL),
        aiohttp_session=session,
    )
    
    # Create room if not provided
    if not room_url:
        room_params = DailyRoomParams(
            enable_chat=True,
            enable_recording="cloud",
            start_audio_off=False,
            start_video_off=True,
        )
        
        room = await rest_helper.create_room(room_params)
        if not room or not room.url:
            raise RuntimeError("Failed to create Daily room")
        
        room_url = room.url
        print(f"Created new Daily room: {room_url}")
    
    # Create token if not provided
    if not token:
        expiry_time = float(os.getenv("DAILY_TOKEN_EXPIRY", DEFAULT_TOKEN_EXPIRY))
        token = await rest_helper.get_token(room_url, expiry_time)
        
        if not token:
            raise RuntimeError(f"Failed to create token for room: {room_url}")
    
    return room_url, token


async def create_room_with_token(session: aiohttp.ClientSession, params: Optional[DailyRoomParams] = None) -> Tuple[str, str]:
    """Create a new Daily room and token.
    
    Args:
        session: Active aiohttp client session
        params: Optional room parameters
        
    Returns:
        Tuple of (room_url, token)
    """
    api_key = os.getenv("DAILY_API_KEY")
    if not api_key:
        raise ValueError("DAILY_API_KEY environment variable is required")
    
    # Create REST helper
    rest_helper = DailyRESTHelper(
        daily_api_key=api_key,
        daily_api_url=os.getenv("DAILY_API_URL", DEFAULT_DAILY_API_URL),
        aiohttp_session=session,
    )
    
    # Create room
    if not params:
        params = DailyRoomParams(
            enable_chat=True,
            enable_recording="cloud",
            start_audio_off=False,
            start_video_off=True,
        )
    
    room = await rest_helper.create_room(params)
    if not room or not room.url:
        raise RuntimeError("Failed to create Daily room")
    
    # Create token
    expiry_time = float(os.getenv("DAILY_TOKEN_EXPIRY", DEFAULT_TOKEN_EXPIRY))
    token = await rest_helper.get_token(room.url, expiry_time)
    
    if not token:
        raise RuntimeError(f"Failed to create token for room: {room.url}")
    
    return room.url, token