from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int


new_person:Person={'name':'nihal','age':4}  

print(new_person)