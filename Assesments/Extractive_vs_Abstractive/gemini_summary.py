import os
from dotenv import load_dotenv
from google import genai

# Load .env file
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# Create Gemini client
client = genai.Client(api_key=API_KEY)


def abstractive_summary(text, length):

    if length == "Short":
        instruction = "Summarize this text in about 50 words."

    elif length == "Medium":
        instruction = "Summarize this text in about 100 words."

    else:
        instruction = "Summarize this text in about 200 words."

    prompt = f"""
    {instruction}

    Text:

    {text}
    """

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text