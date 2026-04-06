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

prompt1=PromptTemplate(
    template='Generate a joke about {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Explain this joke --> {joke}',
    input_variables=['joke']
)
parser=StrOutputParser()

joke_chain=RunnableSequence(prompt1,model,parser)

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'explain':RunnableSequence(prompt2,model,parser)
})

final_chain=RunnableSequence(joke_chain,parallel_chain)

result=final_chain.invoke({'topic':'Cricket'})


print('joke-->',result['joke'])

print('explain-->',result['explain'])