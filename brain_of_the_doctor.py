# brain_of_the_doctor.py
import os
from groq import Groq

def encode_image(image_path: str) -> str:
    """Return base64-encoded data URI content for an image file path."""
    import base64
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return encoded

def analyze_image_with_query(query: str, encoded_image: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
    """
    Ask Groq to analyze an image together with text.
    Returns textual answer string from the model.
    """
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key) if api_key else Groq()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                },
            ],
        }
    ]
    resp = client.chat.completions.create(messages=messages, model=model)
    return resp.choices[0].message.content.strip()
