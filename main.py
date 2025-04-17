import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any

import aiohttp
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.consumer_processor import ConsumerProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.producer_processor import ProducerProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService, GoogleLLMContext
from pipecat.services.google.rtvi import GoogleRTVIObserver
from pipecat.services.groq import GroqLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter
from pipecat.transcriptions.language import Language

from runner import configure

# Setup logging
logger.remove(0)
logger.add(sys.stderr, level="INFO")


class SuggestionProducer(ProducerProcessor):
    def __init__(self):
        super().__init__()
        self.suggestion_count = 0
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if hasattr(frame, "text") and direction == FrameDirection.DOWNSTREAM:
            # Only produce frames from Gemini branch that contain higher-level suggestions
            if frame.metadata.get("source") == "gemini" and "here's a suggestion:" in frame.text.lower():
                self.suggestion_count += 1
                logger.info(f"Producing suggestion {self.suggestion_count}: {frame.text}")
                
                # Add metadata to indicate this is a suggestion from Gemini
                frame.metadata["suggestion"] = True
                frame.metadata["suggestion_id"] = self.suggestion_count
                
                # Share this frame with the consumer in the fast branch
                await self.produce(frame)
        
        await self.push_frame(frame)


class FrameLogger(FrameProcessor):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if hasattr(frame, "text") and direction == FrameDirection.DOWNSTREAM:
            logger.info(f"[{self.name}] {frame.text}")
        await self.push_frame(frame)


async def create_voice_bot(session: aiohttp.ClientSession, room_url: str, token: str):
    # Create Daily transport
    transport = DailyTransport(
        room_url,
        token,
        "Smart Voice Bot",
        DailyParams(
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )
    
    # Create shared STT service (used by both branches)
    deepgram_stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2",
        interim_results=False,
        endpointing=500,
    )
    
    # Create RTVI processor for client UI events
    rtvi_processor = RTVIProcessor(config=RTVIConfig(config=[]))
    
    # Create system instructions
    system_instruction = """
    You are a helpful AI assistant that provides concise responses.
    Format your responses for natural speech without special characters or complex formatting.
    Do not mention that your responses will be converted to audio.
    """
    
    # Create the LLM services for each branch
    
    # Fast branch: Groq's Llama-3/70B for quick responses (<150ms)
    groq_llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        params=GroqLLMService.InputParams(
            temperature=0.5,
            max_tokens=800,  # Shorter responses for quick replies
            system=system_instruction
        )
    )
    
    # Smart branch: Gemini-2.5 Pro for deeper analysis
    gemini_llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-1.5-pro",
        system_instruction="""
        You are an AI assistant specializing in detailed analysis and higher-level suggestions.
        Provide thoughtful, insightful responses that begin with "Here's a suggestion: ".
        Your responses will be shared with another AI that handles quick replies, so focus on
        providing additional context and deeper insights rather than answering directly.
        """
    )
    
    # Create TTS service (Cartesia)
    cartesia_tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
        text_filters=[MarkdownTextFilter()],
    )
    
    # Create context for Groq branch (OpenAI compatible)
    groq_context_obj = OpenAILLMContext([
        {
            "role": "system",
            "content": system_instruction
        },
        {
            "role": "assistant", 
            "content": "Hello! I'm your voice assistant. How can I help you today?"
        }
    ])
    
    # Create separate Google context for Gemini branch with search capability
    search_tool = {
        "google_search_retrieval": {
            "dynamic_retrieval_config": {
                "mode": "MODE_DYNAMIC",
                "dynamic_threshold": 0.6  # Only search when needed
            }
        }
    }
    
    gemini_context_obj = GoogleLLMContext(
        messages=[
            {
                "role": "assistant", 
                "content": "Hello! I'm your voice assistant. How can I help you today?"
            }
        ],
        tools=[search_tool]
    )
    
    # Create context aggregators for each branch
    groq_context = groq_llm.create_context_aggregator(groq_context_obj)
    gemini_context = gemini_llm.create_context_aggregator(gemini_context_obj)
    
    # Create producer/consumer for cross-branch communication
    suggestion_producer = SuggestionProducer()
    suggestion_consumer = ConsumerProcessor(producer=suggestion_producer)
    
    # Create frame loggers for debugging
    fast_logger = FrameLogger("Fast Branch")
    smart_logger = FrameLogger("Smart Branch")
    
    # Build the parallel pipeline
    pipeline = Pipeline([
        transport.input(),
        deepgram_stt,
        rtvi_processor,
        ParallelPipeline(
            # Fast branch: Quick responses with Groq
            [
                fast_logger,
                groq_context.user(),
                groq_llm,
                suggestion_consumer,  # Receive suggestions from smart branch
                cartesia_tts,
                groq_context.assistant(),
            ],
            # Smart branch: Deeper analysis with Gemini
            [
                smart_logger,
                gemini_context.user(),
                gemini_llm,
                suggestion_producer,  # Share suggestions with fast branch
                gemini_context.assistant(),
            ]
        ),
        transport.output(),
    ])
    
    # Set up task with observers
    observers = [GoogleRTVIObserver(rtvi_processor)]
    
    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
        observers=observers,
    )
    
    # Set up event handlers
    @rtvi_processor.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi_processor.set_bot_ready()
        
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"First participant joined: {participant}")
        # Initialize both branches with their respective context frames
        await task.queue_frames([
            groq_context.user().get_context_frame(),
            gemini_context.user().get_context_frame()
        ])
        
    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"Participant left: {participant}, reason: {reason}")
        await task.cancel()
    
    # Run the pipeline
    runner = PipelineRunner()
    await runner.run(task)


async def main():
    async with aiohttp.ClientSession() as session:
        # Get room URL and token
        room_url, token = await configure(session)
        logger.info(f"Room URL: {room_url}")
        
        # Start the voice bot
        await create_voice_bot(session, room_url, token)


if __name__ == "__main__":
    asyncio.run(main())