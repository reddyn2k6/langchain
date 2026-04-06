from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

#object of openAI is made
llm=OpenAI(model='gpt-3.5-turbo-instruct')

#invoke function is used to send prompt and get answer 
result=llm.invoke("What is the capital of india")

print(result)