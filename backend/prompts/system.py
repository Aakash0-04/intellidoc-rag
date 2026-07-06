"""System prompts and greeting detection."""


SYSTEM_PROMPT = """You are a helpful document assistant. Answer questions based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have enough information to answer that."

IMPORTANT: Do NOT mention source filenames, page numbers, or [Source: ...] in your answer text.
The sources will be displayed separately to the user. Just provide a clean, natural answer.

Context:
{context}"""

GREETING_PROMPT = """You are a friendly assistant. Respond naturally to greetings and small talk.
Keep responses brief and warm. If the user asks about documents, guide them to upload files."""


def is_greeting_or_smalltalk(query: str) -> bool:
    """Detect if query is a greeting or small talk."""
    greetings = {
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "how are you", "what's up", "thanks", "thank you", "bye", "goodbye",
        "who are you", "what can you do", "help", "how do you do",
        "i dont feel good", "i don't feel good", "i have fever", "i am sick",
        "i feel bad", "not feeling well", "feeling down",
    }
    lowered = query.lower().strip("!?., ")
    # Also check if any greeting phrase is contained in the query
    return any(g in lowered or lowered.startswith(g) for g in greetings)