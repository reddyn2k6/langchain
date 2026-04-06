from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=OpenAIEmbeddings(model='text-embedding-3-large',dimensions=32)


documents=[
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France",
    "Hyderbad is the capital of Telangana"
]



result=embedding.embed_documents("Delhi is the capital of india")

print(str(result))