from langchain_core.tools import tool


#create a function --> add type hints ---> add the @tool keyword

@tool
def multiply(a: int , b: int)-> int:
    """Multiply Two Numbers"""
    return a*b

result=multiply.invoke({"a":3,"b":4})

print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)