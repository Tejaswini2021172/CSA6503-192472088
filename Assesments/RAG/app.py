from call_llm import answer


print("=" * 50)

print("🎓 College Curriculum Assistant")

print("=" * 50)

print("Type 'quit' to exit.\n")


while True:

    question = input("Ask: ")

    if question.lower() == "quit":
        break

    print()

    print(answer(question))

    print("-" * 50)