from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model=ChatHuggingFace(llm=llm)

messages=[ 
    SystemMessage(content='You are a helpful AI assitant here to assist Humans with there work and your name is Aadhyaa')
]



while True:
    user_input=input('You :')
    messages.append(HumanMessage(content=user_input))
    if user_input=='exit':
        print('AI: Have a nice day ')
        break
    result=model.invoke(messages)
    messages.append(AIMessage(result.content))
    print('AI: ' ,result.content)
