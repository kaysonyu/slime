"""ASR-backed WER reward. Service failures are errors, not zero-reward samples."""

import os
import unicodedata
from pathlib import Path

import httpx


def normalized_tokens(text, language="en"):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(c).startswith(("P", "S")) else c for c in text)
    if language == "zh":
        # Explicit character-token evaluation for unsegmented Chinese (CER-style).
        return [c for c in text if not c.isspace()]
    return text.split()


def word_error_rate(reference, hypothesis, language="en"):
    expected, actual = normalized_tokens(reference, language), normalized_tokens(hypothesis, language)
    if not expected:
        raise ValueError("WER requires a non-empty normalized reference")
    previous = list(range(len(actual) + 1))
    for i, word in enumerate(expected, 1):
        current = [i]
        for j, candidate in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (word != candidate)))
        previous = current
    errors = previous[-1]
    return {
        "wer": errors / len(expected),
        "errors": errors,
        "reference_tokens": len(expected),
        "hypothesis_tokens": len(actual),
        "tokenization": "characters" if language == "zh" else "words",
    }


async def reward_func(args, sample, *, client=None):
    if sample.audio_path is None:
        transcription = ""
    else:
        if not args.asr_endpoint:
            raise ValueError("WER requires --asr-endpoint")
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=args.omni_timeout, trust_env=False)
        try:
            key = os.getenv("ASR_API_KEY")
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            with Path(sample.audio_path).open("rb") as audio:
                response = await client.post(
                    args.asr_endpoint,
                    headers=headers,
                    files={"file": ("speech.wav", audio, "audio/wav")},
                    data={"model": args.asr_model, "response_format": "json"},
                )
            response.raise_for_status()
            body = response.json()
            if not isinstance(body.get("text"), str):
                raise ValueError("ASR response must contain a transcription string")
            transcription = body["text"]
        finally:
            if owns_client:
                await client.aclose()
    reference = sample.label or sample.prompt
    score = word_error_rate(reference, transcription, args.wer_language)
    sample.metadata.update({"asr_text": transcription, **score})
    # WER can exceed one due to insertions. Preserve the real objective.
    return 1.0 - score["wer"]
