import json
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Literal, Any, List
from pathlib import Path


class TypeDetails(BaseModel):
    type: Literal["string", "number", "boolean", "integer"]


class FunctionDefinition(BaseModel):
    model_config = {"extra": "forbid"}
    name: str = Field(max_length=50, min_length=1)
    description: str = Field(max_length=100, min_length=1)
    parameters: Dict[str, TypeDetails]
    returns: TypeDetails


class FunctionCalling(BaseModel):
    model_config = {"extra": "forbid"}
    prompt: str = Field(min_length=1)


class FunctionOutput(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]


def load_definitions(file_path: str) -> List[FunctionDefinition]:
    """Loads and validates function definitions from a JSON file."""
    file: List[FunctionDefinition] = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            x: List[Dict[str, Any]] = json.load(f)
            if x and isinstance(x, list):
                for line in x:
                    file.append(FunctionDefinition(**line))
    except ValidationError as e:
        print(f"[ERROR - PARSING DEF]: {e.errors()[0]['msg']}")
    except json.JSONDecodeError:
        print(f"[ERROR - PARSING]: Invalid JSON in file {file_path}.")
    except FileNotFoundError:
        print("[ERORR - PARSING]: File not found!")

    return file


def load_calling(file_path: str) -> List[FunctionCalling]:
    """Loads and validates function calling prompts from a JSON file."""
    file: List[FunctionCalling] = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            x: Any = json.load(f)
            if x:
                if isinstance(x, list):
                    for line in x:
                        file.append(FunctionCalling(**line))
                else:
                    file.append(FunctionCalling(**x))
    except ValidationError as e:
        print(f"[ERROR - PARSING]: {e.errors()[0]['msg']}")
    except json.JSONDecodeError:
        print(f"[ERROR - PARSING]: Invalid JSON in file {file_path}.")
    except FileNotFoundError:
        print("[ERORR - PARSING]: File not found!")

    return file


def load_output(filepath: Path) -> None:
    """Loads and validates the final generated output file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            x: List[Dict[str, Any]] = json.load(f)
            if x and isinstance(x, list):
                for line in x:
                    FunctionOutput(**line)
    except ValidationError as e:
        print(f"[ERROR - OUTPUT]: {e.errors()[0]['msg']}")
    except FileNotFoundError:
        print("[ERROR - OUTPUT]: Output file not found!")
