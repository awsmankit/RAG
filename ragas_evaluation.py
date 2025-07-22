from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy,
)
from datasets import Dataset

# ✅ Sample QA dataset — you can generate this from your RAG logs
sample_data = [
    {
        "question": "What is retrieval augmented generation?",
        "answer": "Retrieval-augmented generation (RAG) is a technique that combines retrieval and generation in NLP tasks.",
        "contexts": [
            "RAG combines a retriever module that fetches relevant documents from a knowledge base...",
            "It improves LLMs by grounding answers in external data...",
        ],
        "ground_truth": "RAG is a method that augments large language models with retrieval to improve factuality and context.",
    },
    # Add more items as needed...
]

# Convert to HuggingFace-style dataset
dataset = Dataset.from_list(sample_data)

# 🔍 Evaluate with all major metrics
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_relevancy,
    ]
)

print("📊 Evaluation Results:")
print(results)
