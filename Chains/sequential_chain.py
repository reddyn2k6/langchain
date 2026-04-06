#build an application __. first give input to LLM get output and then give that output to LLM as input and try getting another output
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

prompt1=PromptTemplate(
    template='Generate a detailder report on {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Generate a 5 point summary from the following \n {text}',
    input_variables=['text']
)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()


#sequential chain
chain=prompt1 | model | parser | prompt2 | model | parser

result=chain.invoke({'topic':'Cricket'})


print(result)


