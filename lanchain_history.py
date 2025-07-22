from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = qa_chain.run("What is RAG?")
    print("📊 Token Usage:", cb.total_tokens)
    print("⏱️ Cost:", cb.total_cost)
