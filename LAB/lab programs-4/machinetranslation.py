from groq import Groq

client = Groq(api_key="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")

text = input("Enter text: ")
language = input("Enter target language: ")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": f"Translate the following text into {language}. Give only the translation:\n\n{text}"
        }
    ]
)
print("\nTranslation:")
print(response.choices[0].message.content)