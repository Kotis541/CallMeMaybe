from src.classes import FunctionCalling, FunctionDefinition, load_definitions, load_calling
from typing import List, Tuple
import sys
from pathlib import Path

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
            file_path = Path(arg[i + 1])
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open('w') as file:
                file.close()
        else:
            file_path = Path("data/output/function_calling_results.json")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open('w') as file:
                file.close()
        if fn_call == [] or fn_def == []:
            exit()
    except ValueError as e:
        print(f"[PARSER - ERROR]: {e}", file=sys.stderr)

    return fn_def, fn_call, file_path