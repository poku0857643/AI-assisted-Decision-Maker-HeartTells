"""Streaming chat endpoint for the HeartTells conversation page.

Replies are streamed as Server-Sent Events so the UI can render them token by
token. Each event is a JSON object with a ``type``:

- ``mood``        detected emotional tone of the user's message (drives the
                  heartbeat animation on the page)
- ``choices``     two options detected in the message ("A or B")
- ``token``       a chunk of the assistant reply
- ``suggestions`` quick-reply chips to show after the reply
- ``done``        end of stream

If no OpenAI key is configured, a local responder streams a reflective reply
so the page still feels responsive in development.
"""

import asyncio
import json
import random
import re
from typing import AsyncIterator, List, Literal, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config.config import settings

router = APIRouter(prefix="/chat", tags=["Chat"])

SYSTEM_PROMPT = (
    "You are HeartTells, a warm, curious companion that helps people make hard "
    "decisions by balancing logic (mind) with feeling (heart). Keep replies short "
    "(2-4 sentences), conversational and kind. Reflect what you hear, ask one "
    "gentle follow-up question at a time, and when the person has two options, "
    "help them notice how each one feels in their body. Never give medical advice."
)

MAX_HISTORY = 20


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: List[ChatMessage] = Field(default_factory=list)


# --- lightweight text signals -------------------------------------------------

MOOD_KEYWORDS = {
    "anxious": ["anxious", "worried", "scared", "afraid", "nervous", "stress", "panic", "overwhelm", "can't decide", "stuck", "torn"],
    "sad": ["sad", "lonely", "lost", "hurt", "cry", "miss", "regret", "tired", "exhausted"],
    "excited": ["excited", "can't wait", "amazing", "love", "thrilled", "happy", "great", "yay", "!!"],
    "calm": ["calm", "fine", "okay", "ok", "relaxed", "peaceful", "thanks", "thank you"],
}


def detect_mood(text: str) -> str:
    lowered = text.lower()
    scores = {
        mood: sum(lowered.count(word) for word in words)
        for mood, words in MOOD_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "neutral"


CHOICE_PATTERN = re.compile(
    r"(?:should i|do i|whether to|between)\s+(.+?)\s+(?:or|and)\s+(.+?)(?:[?.!]|$)",
    re.IGNORECASE,
)


def detect_choices(text: str) -> Optional[List[str]]:
    match = CHOICE_PATTERN.search(text)
    if not match:
        return None
    first, second = (part.strip(" ,\"'") for part in match.groups())
    if not first or not second or len(first) > 80 or len(second) > 80:
        return None
    return [first, second]


def suggestions_for(mood: str, choices: Optional[List[str]]) -> List[str]:
    if choices:
        return [
            f"Imagine I chose {choices[0]}",
            f"Imagine I chose {choices[1]}",
            "Check my heart",
        ]
    return {
        "anxious": ["Help me slow down", "What's really at stake?", "Check my heart"],
        "sad": ["I just need to vent", "What would help right now?"],
        "excited": ["Is this too good to be true?", "What could go wrong?"],
    }.get(mood, ["I have two options", "I'm not sure how I feel", "Check my heart"])


# --- reply generation ---------------------------------------------------------

def local_reply(message: str, mood: str, choices: Optional[List[str]]) -> str:
    if choices:
        a, b = choices
        return random.choice([
            f"Okay — {a}, or {b}. Let's sit with both for a moment. "
            f"Picture yourself a month from now having chosen {a}. What's the first feeling that shows up? "
            f"Then try the same with {b}, and notice if your chest tightens or loosens.",
            f"I hear two paths: {a} and {b}. Your mind probably already has a list of pros and cons. "
            f"Let's ask your heart instead — when you say \"{a}\" out loud, does it feel like relief or like a weight?",
        ])
    if message.strip().lower() == "check my heart":
        return (
            "Let's listen in. Take a slow breath, then record a short ECG on your watch while you hold "
            "one option in mind. Your heart often answers before your words do — I'll help you read it."
        )
    return {
        "anxious": "That sounds like a lot to carry. Let's take it one breath at a time. "
                   "What's the part of this that feels heaviest right now?",
        "sad": "I'm really glad you told me. You don't have to figure it all out at once. "
               "What's been sitting with you the most today?",
        "excited": "I can feel the spark in that! Let's make sure your heart and your head are on the same page. "
                   "What's pulling you toward it most?",
        "calm": "It's nice to hear some steadiness in your words. "
                "Is there a decision on your mind, or would you like to just check in with yourself?",
    }.get(mood, "I'm listening. Tell me a little more — is there a choice you're weighing, "
                "or a feeling you're trying to make sense of?")


async def stream_local(text: str) -> AsyncIterator[str]:
    """Stream a canned reply with a human-ish rhythm (pauses after punctuation)."""
    for word in re.findall(r"\S+\s*", text):
        yield word
        if word.rstrip().endswith((".", "?", "!")):
            await asyncio.sleep(random.uniform(0.18, 0.32))
        elif word.rstrip().endswith((",", "—", ":")):
            await asyncio.sleep(random.uniform(0.08, 0.14))
        else:
            await asyncio.sleep(random.uniform(0.025, 0.06))


async def stream_openai(request: ChatRequest) -> AsyncIterator[str]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.CHATGPT_API_KEY)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [m.model_dump() for m in request.history[-MAX_HISTORY:]]
    messages.append({"role": "user", "content": request.message})

    stream = await client.chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=messages,
        temperature=0.8,
        max_tokens=300,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def event_stream(request: ChatRequest) -> AsyncIterator[str]:
    mood = detect_mood(request.message)
    choices = detect_choices(request.message)

    yield sse({"type": "mood", "mood": mood})
    if choices:
        yield sse({"type": "choices", "choices": choices})

    # A short "thinking" beat so the typing indicator is visible.
    await asyncio.sleep(random.uniform(0.4, 0.8))

    tokens = (
        stream_openai(request)
        if settings.CHATGPT_API_KEY
        else stream_local(local_reply(request.message, mood, choices))
    )
    try:
        async for token in tokens:
            yield sse({"type": "token", "text": token})
    except Exception as exc:  # keep the conversation going on API failure
        print(f"Chat stream error: {exc}")
        async for token in stream_local(local_reply(request.message, mood, choices)):
            yield sse({"type": "token", "text": token})

    yield sse({"type": "suggestions", "suggestions": suggestions_for(mood, choices)})
    yield sse({"type": "done"})


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        event_stream(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
