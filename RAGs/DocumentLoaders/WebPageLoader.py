from langchain_community.document_loaders import WebBaseLoader


url='https://www.flipkart.com/apple-macbook-neo-a18-pro-2026-pro-8-gb-256-gb-ssd-tahoe-mhfa4hn-a/p/itm9fce39e65bd7e?pid=COMHH8C57Y6W6NZU&marketplace=FLIPKART&lid=LSTCOMHH8C57Y6W6NZUASDOLA&q=mac+book+neo&fm=organic&pageUID=1775071535225'
loader=WebBaseLoader(url)

docs=loader.load()

print(len(docs))