from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.5-9B",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

chat = ChatHuggingFace(llm=llm)
print("Before invoke...")
response = chat.invoke("What is the capital of India")
print("Before invoke...")

print(response.content)