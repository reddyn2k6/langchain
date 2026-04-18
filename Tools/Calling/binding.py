from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.tools import tool
import requests
import os
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model=ChatHuggingFace(llm=llm)

#make tools
@tool
def multiply(a:int, b:int)->int:
    """Given 2 integers a and b it returns the product of these numbers"""
    return a*b



#tool binding
llm_with_tools=model.bind_tools([multiply]) #binding is done


#tool calling -->> LLM basically only suggest the tools to be used and LANGCHAIN calls the tool
print(llm_with_tools.invoke('Use the multiply tool to calculate 4 multiplied by 5.').tool_calls)