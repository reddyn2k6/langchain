from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):

    name:str='nihal' #default fields
    age:Optional[int]=None
    email:EmailStr
    cgpa:float=Field(gt=0,lt=10)



new_student={'age':32,"email":'nihal@gmail.com',"cgpa":5}

student=Student(**new_student)

print(student) 