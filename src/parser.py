from src.classes import FunctionCalling, FunctionDefinition, load_definitions, load_calling
from typing import List, Tuple
import sys

def parse(arg: list[str]) -> Tuple[List[FunctionDefinition], List[FunctionCalling]]:
    fn_def: List[FunctionDefinition] = []
    fn_call: List[FunctionCalling] = []
    try:
        if len(arg) > 7:
            raise ValueError("Too many arguments")

        if "--functions_definition" in arg:
            i = arg.index("--functions_definition")
            fn_def = load_definitions(arg[i + 1])
        else:
            fn_def = load_definitions("data/input/functions_definition.json")

        if "--input" in arg:
            i = arg.index("--input")
            fn_call = load_calling(arg[i + 1])
        else:
            fn_call = load_calling("data/input/function_calling_tests.json")
            
        if "--output" in arg:
            i = arg.index("--output")
            print(f"output def: {arg[i + 1]}")
            
    except ValueError as e:
        print(f"[PARSER - ERROR]: {e}", file=sys.stderr)

    return fn_def, fn_call