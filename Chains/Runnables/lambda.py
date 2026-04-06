from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.runnables import RunnableSequence,RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model=ChatHuggingFace(llm=llm)

def word_count(text):
    return len(text.split())

prompt=PromptTemplate(
    template='Generate a joke about {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()

joke_generator=RunnableSequence(prompt,model,parser)

parallel_generator=RunnableParallel({
    'joke':RunnablePassthrough(),
    'number of words':RunnableLambda(lambda x: len(x.split()))
})

chain=RunnableSequence(joke_generator,parallel_generator)

result=chain.invoke('Hello how are you')

print(result)