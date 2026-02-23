from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class Operation(str , Enum):
    addition  = 'add'
    substraction = 'sub'
    #devision = 'dev'
    multiplication = 'mul'


class InfoAcceptor(BaseModel):
    arraySize : int = Field(ge = 1000 , le = 100_000_000)
    operation : Operation

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "arraySize": 10_000,
                "operation": "add"
            }
        }
    )
