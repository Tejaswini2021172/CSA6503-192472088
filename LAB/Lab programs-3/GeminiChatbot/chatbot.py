import os
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

print("=" * 50)
print("🤖 Gemini AI Chatbot")
print("Type 'exit' to quit")
print("=" * 50)

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=user_input
    )

    print("\nGemini:", response.text)