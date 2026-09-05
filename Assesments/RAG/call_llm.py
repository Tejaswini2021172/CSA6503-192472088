import os
from dotenv import load_dotenv
import google.generativeai as genai
from build_prompt import build_prompt
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-3.5-flash")
def answer(question):
    prompt = build_prompt(question)
    response = model.generate_content(prompt)
    return response.text