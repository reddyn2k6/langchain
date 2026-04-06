from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('UNDERTAKING.pdf')

docs=loader.load()

print(len(docs))
