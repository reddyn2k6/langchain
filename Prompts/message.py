from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model=ChatHuggingFace(llm=llm)


messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about langchain')
]

result=model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)