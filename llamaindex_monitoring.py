from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.callbacks import CallbackManager, LlamaDebugHandler

# 📌 Debug handler to track events
debug_handler = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([debug_handler])

# 🧠 Load & index data
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 🧾 Query with monitoring enabled
query_engine = index.as_query_engine(
    llm=OpenAI(model="gpt-3.5-turbo"),
    callback_manager=callback_manager
)

response = query_engine.query("What does RAG mean?")
print("✅ Final Answer:", response)
