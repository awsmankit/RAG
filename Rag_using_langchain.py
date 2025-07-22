# langchain_rag_pipeline.py
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Step 1: Load and split document
loader = TextLoader("data/intro_to_rag.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Step 2: Embedding + Vector Store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Step 3: Setup Retriever and LLM
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Step 4: QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Step 5: Ask a question
query = "What are the main steps in building a RAG pipeline?"
result = qa_chain(query)

print("Answer:", result["result"])
for doc in result["source_documents"]:
    print(f"Source: {doc.metadata}")
