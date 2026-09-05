from retrieve import retrieve

PROMPT = """
You are a helpful AI assistant.

You have access to the following context retrieved from an uploaded PDF.

Instructions:
1. If the context contains the answer, answer using the context.
2. If the context is not sufficient, answer using your own knowledge.
3. If the answer comes from the uploaded document, cite chunk numbers like [c2].
4. If the answer comes from your own knowledge, mention:
   "Note: This answer is based on general knowledge, not the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""

def build_prompt(question):
    docs, metas = retrieve(question)

    context = "\n".join(
        f"[c{m['chunk']}] {d}"
        for d, m in zip(docs, metas)
    )

    return PROMPT.format(
        context=context,
        question=question
    )