from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI   # 👈 missing import

# Documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression."),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity."),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation."),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity."),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy."),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand."),
    Document(page_content="Python balances readability with power, making it a popular system design language."),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight."),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement."),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy."),
]

# Embeddings
embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

# Vector store
vectorstore = FAISS.from_documents(
    documents=all_docs,
    embedding=embedding
)

# Normal retriever
similarity_retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={"k": 5}
)

# LLM (IMPORTANT)
llm = ChatOpenAI(model="gpt-3.5-turbo")

# MultiQuery Retriever
# multiquery_retriever = MultiQueryRetriever.from_llm(
#     retriever=similarity_retriever,
#     llm=llm
# )

# Query
query = "How to improve mental health?"

# results = multiquery_retriever.invoke(query)

# for i, doc in enumerate(results):
#     print(f"\n--- Result {i} ---")
#     print(doc.page_content)