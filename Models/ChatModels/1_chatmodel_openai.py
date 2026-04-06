from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI(model='gpt-4',temperature=0) #temperature basically is a measure of creativeness of the response

result=model.invoke("What is the capital of india")

print(result) #sends out all the meta data with the answer

#result.content gives out the actual answer