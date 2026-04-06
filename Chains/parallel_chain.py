from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.runnables import RunnableParallel
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model1 = ChatHuggingFace(llm=llm)
model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text:\n{text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short quiz questions from the following text:\n{text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document.\n\nNotes:\n{notes}\n\nQuiz:\n{quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = '''Support Vector Machine (SVM) is a supervised machine learning algorithm...'''

result = chain.invoke({"text": text})

print(result)