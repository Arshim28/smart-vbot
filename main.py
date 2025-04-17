import asyncio
import os
import sys
import time

import aiohttp
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, TextFrame
from pipecat.observers.base_observer import BaseObserver
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
from pipecat.services.groq.llm import GroqLLMService
# Updated import path for Google LLM service
from pipecat.services.google.llm import GoogleLLMService, GoogleLLMContext
from pipecat.services.google.rtvi import GoogleRTVIObserver
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

from runner import configure

# 0) Logging - Set to DEBUG to observe LLM & TTS operations
logger.remove()
logger.add(sys.stderr, level="DEBUG")


# 1) Gemini Observer for smart suggestions
class GeminiObserver(BaseObserver):
    """
    Observer that watches conversation to generate periodic suggestions
    using Gemini and injects them into the main pipeline.
    """
    
    def __init__(self, gemini_service, gemini_context, task=None):
        self.gemini = gemini_service
        self.context = gemini_context
        self.task = task
        self.transcript = []
        self.last_suggestion_time = 0
        self.suggestion_interval = 30  # seconds
    
    async def on_push_frame(self, src, dst, frame, direction, timestamp):
        # Only process downstream text frames with content
        if direction == FrameDirection.DOWNSTREAM and hasattr(frame, "text") and frame.text.strip():
            # Add to transcript
            entry = {
                "role": "user" if frame.metadata.get("speaker") == "user" else "assistant",
                "content": frame.text,
                "timestamp": time.time()
            }
            self.transcript.append(entry)
            
            # Update context for Gemini
            if entry["role"] == "user":
                # Add user message to Gemini context
                self.context.messages.append({
                    "role": "user", 
                    "content": entry["content"]
                })
            
            # Check if it's time for a suggestion
            current_time = time.time()
            if (current_time - self.last_suggestion_time >= self.suggestion_interval 
                    and len(self.transcript) >= 2):
                
                # Generate suggestion using Gemini
                response = await self.gemini.generate(context=self.context)
                
                if response and response.choices:
                    suggestion_text = response.choices[0].message.content
                    
                    # Create suggestion frame
                    suggestion_frame = TextFrame(
                        content=suggestion_text,
                        metadata={
                            "type": "suggestion",
                            "source": "gemini",
                            "suggestion": True,
                            "suggestion_id": len(self.transcript)
                        }
                    )
                    
                    # Log the suggestion
                    logger.info(f"Gemini suggestion: {suggestion_text}")
                    
                    # Inject into pipeline if task is set
                    if self.task:
                        await self.task.queue_frame(suggestion_frame)
                    
                    # Update suggestion time
                    self.last_suggestion_time = current_time
                    
                    # Add assistant response to context
                    self.context.messages.append({
                        "role": "assistant",
                        "content": suggestion_text
                    })


# 2) Suggestion filter for producer/consumer
async def is_suggestion(frame):
    return (hasattr(frame, "text") and 
            frame.metadata.get("type") == "suggestion" and 
            frame.metadata.get("source") == "gemini")


# 3) Simple frame logger
class FrameLogger(FrameProcessor):
    def __init__(self, name_prefix):
        # Pass name to FrameProcessor via super().__init__(name=...)
        super().__init__(name=name_prefix)
        self._name_prefix = name_prefix

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if hasattr(frame, "text") and direction == FrameDirection.DOWNSTREAM:
            source = frame.metadata.get("source", "unknown")
            suggestion = " [SUGGESTION]" if frame.metadata.get("suggestion") else ""
            logger.info(f"[{self._name_prefix}:{source}{suggestion}] {frame.text}")
        await self.push_frame(frame)


async def create_voice_bot(session, room_url, token):
    # — Transport, STT, RTVI —
    transport = DailyTransport(
        room_url, token, "Smart Voice Bot",
        DailyParams(
            audio_out_enabled=True,       # Explicit audio output enabled
            transcription_enabled=True,   # Explicit transcription enabled  
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2",
        interim_results=False,
        endpointing=500,
    )
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Shared system prompt for Groq
    groq_system_instruction = (
        "You are a helpful AI assistant that provides concise, speech‑friendly responses. "
        "Do not mention that your responses will be converted to audio. "
        "When you receive a suggestion from Gemini, incorporate that insight naturally "
        "into your responses if relevant."
    )

    # — Groq branch (fast) —
    groq_llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=800,
    )
    
    # Initialize Groq context with a system message
    groq_ctx = OpenAILLMContext(messages=[
        {"role": "system", "content": groq_system_instruction}
    ])
    
    # Create context aggregators for Groq
    groq_agg = groq_llm.create_context_aggregator(groq_ctx)

    # — Gemini setup (deep) —
    gemini_system_instruction = (
        "You are an AI assistant specializing in detailed analysis. "
        "Observe the conversation and provide insightful suggestions that would be helpful. "
        "Begin suggestions with \"Here's a suggestion: \" and keep them concise and speech-friendly."
    )

    gemini_llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.0-flash",
        system_instruction=gemini_system_instruction,
    )
    
    # Initialize Gemini context with proper GoogleLLMContext
    # Note: system instruction is set in the service, not in the context
    gemini_ctx = GoogleLLMContext(
        messages=[]
    )

    # — TTS —
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
        text_filters=[MarkdownTextFilter()],
    )

    # — Suggestion producer/consumer —
    suggestion_producer = ProducerProcessor(
        filter=is_suggestion,
        transformer=lambda f: f,  # identity transform
        passthrough=False  # don't let suggestions leak into other pipelines
    )
    
    suggestion_consumer = ConsumerProcessor(
        producer=suggestion_producer,
        direction=FrameDirection.UPSTREAM  # inject into Groq's upstream flow
    )

    # — Frame loggers —
    main_log = FrameLogger(name_prefix="Main Pipeline")

    # — Assemble pipeline —
    pipeline = Pipeline([
        transport.input(),
        stt,
        rtvi,
        # Main pipeline with Groq
        main_log,
        groq_agg.user(),
        suggestion_consumer,  # inject suggestions here
        groq_llm,
        groq_agg.assistant(), # Assistant aggregator must come after LLM but before TTS
        tts,
        transport.output(),
    ])

    # — Task & observers —
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            idle_timeout_secs=30,  # 30 seconds of idle time triggers suggestion
            idle_timeout_frames=(TextFrame,),  # reset on text frames
            cancel_on_idle_timeout=False  # don't stop on idle, just trigger event
        ),
    )
    
    # Create Gemini observer
    gemini_observer = GeminiObserver(gemini_llm, gemini_ctx, task)
    
    # Add observers to task
    task.observers = [
        GoogleRTVIObserver(rtvi),
        gemini_observer,
    ]

    # — Event handlers —
    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(evt):
        await rtvi.set_bot_ready()

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(trans, participant):
        logger.info(f"First participant joined: {participant}")
        # Seed Groq context with initial system prompt (already in context creation)
        await task.queue_frame(groq_agg.user().get_context_frame())
        logger.info("Pipeline ready for conversation")

    @transport.event_handler("on_participant_left")
    async def on_participant_left(trans, participant, reason):
        logger.info(f"Participant left: {participant} (reason: {reason})")
        await task.cancel()
    
    # Idle timeout handler for suggestions
    @task.event_handler("on_idle_timeout")
    async def on_idle_timeout(task):
        logger.info("Idle timeout: Generating suggestion from Gemini")
        
        # Ask Gemini for a suggestion based on conversation context
        try:
            response = await gemini_llm.generate(
                context=gemini_ctx,
                prompt="Based on the conversation so far, provide a helpful suggestion or question to keep the conversation going."
            )
            
            if response and response.choices:
                suggestion_text = response.choices[0].message.content
                
                # Create suggestion frame
                suggestion_frame = TextFrame(
                    content=suggestion_text,
                    metadata={
                        "type": "suggestion",
                        "source": "gemini",
                        "suggestion": True,
                        "suggestion_id": "idle-suggestion"
                    }
                )
                
                # Log and inject
                logger.info(f"Idle suggestion: {suggestion_text}")
                await suggestion_producer.produce(suggestion_frame)
                
        except Exception as e:
            logger.error(f"Error generating idle suggestion: {e}")

    # Run the pipeline
    runner = PipelineRunner()
    try:
        await runner.run(task)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")


async def main():
    async with aiohttp.ClientSession() as session:
        try:
            room_url, token = await configure(session)
            logger.info(f"Room URL: {room_url}")
            await create_voice_bot(session, room_url, token)
        except Exception as e:
            logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())