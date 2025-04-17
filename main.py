import asyncio
import os
import sys

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
# Correct submodule import for Groq’s OpenAI‐compatible LLM adapter
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.google.rtvi import GoogleRTVIObserver
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

from runner import configure

# 0) Logging
logger.remove()
logger.add(sys.stderr, level="INFO")


# 1) Suggestion bridge (gemini → groq)
class SuggestionProducer(ProducerProcessor):
    def __init__(self):
        async def always(frame: Frame) -> bool:
            return True
        super().__init__(filter=always, passthrough=True)
        self.count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        text = getattr(frame, "text", "")
        if direction == FrameDirection.DOWNSTREAM and frame.metadata.get("source") == "gemini":
            if "here's a suggestion:" in text.lower():
                self.count += 1
                logger.info(f"Producing suggestion {self.count}: {text}")
                frame.metadata.update(suggestion=True, suggestion_id=self.count)
                await self.produce(frame)
        await self.push_frame(frame)


# 2) Simple frame logger
class FrameLogger(FrameProcessor):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if hasattr(frame, "text") and direction == FrameDirection.DOWNSTREAM:
            logger.info(f"[{self.name}] {frame.text}")
        await self.push_frame(frame)


async def create_voice_bot(session: aiohttp.ClientSession, room_url: str, token: str):
    # — Transport, STT, RTVI —
    transport = DailyTransport(
        room_url, token, "Smart Voice Bot",
        DailyParams(
            audio_out_enabled=True,
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

    # Shared system prompt
    system_instruction = (
        "You are a helpful AI assistant that provides concise, speech‑friendly responses. "
        "Do not mention that your responses will be converted to audio."
    )

    # — Groq branch (fast) —
    groq_llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=800,
        system_instruction=system_instruction,
    )
    # Start with an empty OpenAI‑style context; adapter will prepend the system message.
    groq_ctx = OpenAILLMContext(messages=[])
    groq_agg = groq_llm.create_context_aggregator(groq_ctx)

    # — Gemini branch (deep) —
    google_llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.0-flash",
        system_instruction=(
            "You are an AI assistant specializing in detailed analysis. "
            "Begin suggestions with \"Here's a suggestion: \"."
        ),
    )
    gem_ctx = OpenAILLMContext(messages=[])
    gem_agg = google_llm.create_context_aggregator(gem_ctx)

    # — TTS —
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
        text_filters=[MarkdownTextFilter()],
    )

    # — Suggestion bridge & loggers —
    producer = SuggestionProducer()
    consumer = ConsumerProcessor(producer=producer)
    fast_log = FrameLogger("Fast Branch")
    smart_log = FrameLogger("Smart Branch")

    # — Assemble pipeline —
    pipeline = Pipeline([
        transport.input(),
        stt,
        rtvi,
        ParallelPipeline(
            # Fast branch
            [
                fast_log,
                groq_agg.user(),
                groq_llm,
                consumer,
                tts,
                groq_agg.assistant(),
            ],
            # Smart branch
            [
                smart_log,
                gem_agg.user(),
                google_llm,
                producer,
                gem_agg.assistant(),
            ],
        ),
        transport.output(),
    ])

    # — Task & observers —
    observers = [GoogleRTVIObserver(rtvi)]
    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
        observers=observers,
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(evt):
        await rtvi.set_bot_ready()

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(trans, participant):
        logger.info(f"First participant joined: {participant}")
        # Seed both contexts (system instructions are already set in the services)
        await task.queue_frames([
            groq_agg.user().get_context_frame(),
            gem_agg.user().get_context_frame(),
        ])
        logger.info("Pipeline ready for conversation")

    @transport.event_handler("on_participant_left")
    async def on_participant_left(trans, participant, reason):
        logger.info(f"Participant left: {participant} (reason: {reason})")
        await task.cancel()

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
