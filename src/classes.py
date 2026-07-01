import json
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Literal, Any


class TypeDetails(BaseModel):
    type: Literal["string", "number", "boolean", "integer"]


class FunctionDefinition(BaseModel):
    model_config = {"extra":"forbid"}
    name: str = Field(max_length=50, min_length=1)
    description: str = Field(max_length=100, min_length=1)
    parameters: Dict[str, Any]
    returns: TypeDetails


class FunctionCalling(BaseModel):
    model_config = {"extra":"forbid"}
    prompt: str = Field(min_length=1)


class FunctionOutput(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]


def load_definitions(file_path: str) -> list:
    file = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            x = json.load(f)
            if x:  # Handle case where JSON is just 'null' or file is empty
                for line in x:
                    file.append(FunctionDefinition(**line))
    except ValidationError as e:
        print(f"[ERROR - PARSING DEF]: {e.errors()[0]['msg']}")
    except json.JSONDecodeError:
        print(f"[ERROR - PARSING]: Invalid JSON in file {file_path}. It might be empty.")
    except FileNotFoundError as e:
        print(f"[ERORR - PARSING]: File not found!")
    return file

def load_calling(file_path: str) -> list:
    file = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            x = json.load(f)
            if x:
                if (isinstance(x, list)):
                    for line in x:
                        file.append(FunctionCalling(**line))
                else:
                    file.append(FunctionCalling(**x))
    except ValidationError as e:
        print(f"[ERROR - PARSING CALL]: {e.errors()[0]['msg']}")
    except json.JSONDecodeError:
        print(f"[ERROR - PARSING]: Invalid JSON in file {file_path}. It might be empty.")
    except FileNotFoundError as e:
        print(f"[ERORR - PARSING]: File not found!")
    
    return file