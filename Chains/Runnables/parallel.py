from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.runnables import RunnableSequence,RunnableLambda,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model=ChatHuggingFace(llm=llm)


prompt1=PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Generate linkedIn post about {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'tweet':RunnableSequence(prompt1,model,parser),
    'linkedIn':RunnableSequence(prompt2,model,parser)
})

reuslt=parallel_chain.invoke({'topic':'Cricket'})

print('tweet---->>>>',reuslt['tweet'])


print('LinkedIn---->>>>',reuslt['linkedIn'])