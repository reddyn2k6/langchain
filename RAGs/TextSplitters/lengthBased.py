from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

text='''"The sun dipped below the horizon, casting a warm orange glow over the sleepy town. The sound of crickets filled the air, and a gentle breeze rustled the leaves of the trees. It was a peaceful evening, one that seemed to stretch on forever." 🌅'''

splitter=RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0,
)

result=splitter.split_text(text)

print('Size-->',len(result))
print(result)