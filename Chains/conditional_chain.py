from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableSequence, RunnableParallel,RunnablePassthrough

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt=PromptTemplate(
    template='Give me some information about the topic {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Summarize the text --> {text}',
    input_variables=['text']
)

report_gen=RunnableSequence(prompt,model,parser)



parellel_gen=RunnableBranch(
      (lambda x:len(x.split())>200,RunnableSequence(prompt2,model,parser)),
      RunnablePassthrough()
)

chain=RunnableSequence(report_gen,parellel_gen)

result=chain.invoke({'topic':"US vs Iran"})

print(result)