from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


docs = [
    Document(page_content="Machine learning is a field of AI that enables systems to learn from data."),
    Document(page_content="Deep learning is a subset of machine learning that uses neural networks."),
    Document(page_content="Natural language processing helps computers understand human language."),
    Document(page_content="Computer vision allows machines to interpret and analyze visual data.")
]


vectorstore=Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name='My_Collection'
)

retriver=vectorstore.as_retriever(search_kwargs={'k':2})

query='What is deep learning'

result=retriver.invoke(query)

print(result)