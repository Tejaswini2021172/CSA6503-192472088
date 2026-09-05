from groq import Groq

client = Groq(api_key="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")

text = """
Artificial Intelligence (AI) is one of the most important technologies in the modern world.
It allows computers and machines to perform tasks that normally require human intelligence,
such as learning, reasoning, problem-solving, understanding language, and recognizing images.
AI is being used in many different industries. In healthcare, it helps doctors detect diseases,
analyze medical images, and discover new medicines. In education, AI can provide personalized
learning experiences and intelligent tutoring systems. In banking, it is used to detect fraud,
analyze transactions, and improve customer service. In transportation, AI supports navigation,
traffic prediction, and autonomous driving systems. Machine Learning is a major part of AI
where computers learn patterns from data without being explicitly programmed for every task.
Deep Learning is a further development that uses neural networks with many layers to solve
complex problems involving images, speech, and natural language. As AI continues to develop,
it is expected to create new opportunities while also requiring responsible and ethical use.
"""

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": f"Summarize the following text in exactly 2 or 3 short sentences:\n\n{text}"
        }
    ]
)

print("Original Text:")
print(text)

print("\nSummary:")
print(response.choices[0].message.content)