# voice_of_the_patient.py
import os

def transcribe_with_groq(GROQ_API_KEY: str, audio_filepath: str, stt_model: str = "whisper-large-v3") -> str:
    """
    Uploads local audio file to Groq STT endpoint and returns the transcription text.
    """
    from groq import Groq
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing in environment.")
    client = Groq(api_key=GROQ_API_KEY)
    with open(audio_filepath, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=f,
            language="en"
        )
    # Groq might return object-like or dict-like
    if hasattr(transcription, "text"):
        return transcription.text
    return transcription.get("text", str(transcription))
