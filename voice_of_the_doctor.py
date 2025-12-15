# voice_of_the_doctor.py
import os
from pydub import AudioSegment

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

def text_to_speech_with_elevenlabs(input_text: str, output_filepath: str, voice_id: str = "2qfp6zPuviqeCOZIE9RZ"):
    """
    Generate TTS via ElevenLabs and write a wav file to output_filepath.
    If ElevenLabs isn't available, raise an error and let caller fallback.
    """
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY not set.")
    # using the ElevenLabs API client (user snippet style)
    from elevenlabs.client import ElevenLabs
    from elevenlabs import save as eleven_save

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    stream = client.text_to_speech.convert(voice_id=voice_id, model_id="eleven_turbo_v2", text=input_text)
    tmp_mp3 = output_filepath + ".mp3"
    eleven_save(stream, tmp_mp3)
    sound = AudioSegment.from_mp3(tmp_mp3)
    sound.export(output_filepath, format="wav")
    try:
        os.remove(tmp_mp3)
    except Exception:
        pass
    return output_filepath

# fallback using gTTS
from gtts import gTTS
def text_to_speech_with_gtts(input_text: str, output_filepath: str):
    temp_mp3 = output_filepath + ".mp3"
    tts = gTTS(text=input_text, lang="en", slow=False)
    tts.save(temp_mp3)
    sound = AudioSegment.from_mp3(temp_mp3)
    sound.export(output_filepath, format="wav")
    try:
        os.remove(temp_mp3)
    except Exception:
        pass
    return output_filepath
