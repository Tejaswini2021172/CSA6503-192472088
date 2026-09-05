from groq import Groq

client = Groq(api_key="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")

#prompt = input("WRITE A THREE LINE OF STORY ABOUT GOOD BEHAVIOUR ")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "WRITE A THREE LINE OF STORY ABOUT GOOD BEHAVIOUR"}]
)

print("\nGenerated Text:")
print(response.choices[0].message.content)