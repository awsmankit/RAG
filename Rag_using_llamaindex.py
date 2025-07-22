# llama_rag_pipeline.py
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# Step 1: Load documents
documents = SimpleDirectoryReader("data").load_data()

# Step 2: Setup vector store with OpenAI embeddings
embedding_model = OpenAIEmbedding()
faiss_index = faiss.IndexFlatL2(1536)  # for text-embedding-ada-002
vector_store = FaissVectorStore(faiss_index=faiss_index)

# Step 3: Build LlamaIndex
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store, embed_model=embedding_model)

# Step 4: Setup query engine
query_engine = RetrieverQueryEngine.from_args(
    retriever=index.as_retriever(similarity_top_k=3),
    llm=OpenAI(model="gpt-3.5-turbo"),
)

# Step 5: Ask question
response = query_engine.query("What is retrieval-augmented generation?")
print(response)
