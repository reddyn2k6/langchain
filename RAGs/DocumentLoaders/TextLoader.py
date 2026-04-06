from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

prompt=PromptTemplate(
    template='Generate a summary on {topic}',
    input_variables=['topic']
)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

loader=TextLoader('hello.txt',encoding='utf-8')

docs=loader.load()


chain=prompt | model | parser

result=chain.invoke({'topic':docs[0].page_content})

print(result)


