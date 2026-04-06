from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
import os
from typing import TypedDict,Annotated,Optional,Literal

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model=ChatHuggingFace(llm=llm)

#schema
class Review(TypedDict):

    summary:Annotated[str,"A brief summary of the review"]
    sentiment:Annotated[Literal['pos','neg'],"Return sentiment of the review ither negative, positive or neutral"]
    pros:Annotated[Optional[list[str]],"Write the pros of it"]
     
    

structured_model=model.with_structured_output(Review)


result=structured_model.invoke('''hardware is great, but the software feels bloated.
There are too many pre-installed apps that I can't remove. 
Also, the UI looks outdated compared to other brands.
 Hoping for a software update to fix this''')


print(result)
print('summary',result['summary'])
print('sentiment',result['sentiment'])