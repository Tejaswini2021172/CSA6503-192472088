from groq import Groq

client = Groq(api_key="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")

context = """
Artificial Intelligence is a branch of computer science that focuses on
creating machines capable of performing tasks that normally require human
intelligence. AI is used in healthcare, banking, education, transportation,
and many other industries. Machine learning is a major technique used in AI
where computers learn patterns from data and use them to make predictions.
"""

question = input("Enter your question: ")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": f"""
Answer the question based only on the following context.

Context:
{context}

Question:
{question}

Give a short and clear answer.
"""
        }
    ]
)

print("\nAnswer:")
print(response.choices[0].message.content)