#web search engine
from langchain_community.tools import DuckDuckGoSearchRun

search_tool=DuckDuckGoSearchRun()

results=search_tool.invoke('Hot news in india today')

print(results)


#shell Tool
from langchain_community.tools import ShellTool

shell=ShellTool()

results=shell.invoke('whoami')

print(results)