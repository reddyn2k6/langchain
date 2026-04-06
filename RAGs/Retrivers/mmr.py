from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Embedding model
embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

# Proper documents
docs = [
    Document(page_content="LangChain is a framework for building applications using large language models."),
    Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors."),
    Document(page_content="Retrievers are used to fetch relevant documents based on a query."),
    Document(page_content="Embeddings convert text into numerical vector representations.")
]

# Create vector store
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 3, 'lambda_mult': 1} # --> to balance relevance and diversity
)

# Query
query = "What is LangChain?"

# Retrieve results
results = retriever.invoke(query)

# Print results
for i, doc in enumerate(results):
    print(f"------------------Result {i}----------------------")
    print(doc.page_content)