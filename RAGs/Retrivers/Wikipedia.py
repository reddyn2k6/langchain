from langchain_community.retrievers import WikipediaRetriever


retriver=WikipediaRetriever(top_k_results=2,lang='en')


query='the geopolitical history of india and pakistan from the prespective of the chinese'


docs=retriver.invoke(query)


