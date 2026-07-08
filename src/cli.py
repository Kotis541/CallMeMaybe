from typing import List, Tuple
from .models import FunctionCalling, FunctionDefinition, \
    load_definitions, load_calling
import sys
from pathlib import Path


def parse(arg: List[str]) -> Tuple[List[FunctionDefinition],
                                   List[FunctionCalling], Path]:
    """Parses command-line arguments and loads necessary files."""
    fn_def = []
    fn_call = []
    approved = ["--input", "--output", "--functions_definition"]
    i = 1

    try:
        flags = [a for a in arg if a.startswith("--")]
        if len(flags) != len(set(flags)):
            seen = set()
            for flag in flags:
                if flag in seen:
                    raise ValueError(f"Argument '{flag}' specified more \
                                     than once.")
                seen.add(flag)

        while i < len(arg):
            current_arg = arg[i]
            if current_arg.startswith("--"):
                if current_arg not in approved:
                    raise ValueError(f"Unknown argument '{current_arg}'. \
                                     Allowed arguments are {approved}")
                i += 2
            else:
                raise ValueError(f"Unexpected argument '{current_arg}'. \
                                 Only known flags are allowed.")

        if len(arg) > 7:
            raise ValueError("Too many arguments")

        if "--functions_definition" in arg:
            idx = arg.index("--functions_definition")
            fn_def = load_definitions(arg[idx + 1])
        else:
            fn_def = load_definitions("data/input/functions_definition.json")

        if "--input" in arg:
            idx = arg.index("--input")
            fn_call = load_calling(arg[idx + 1])
        else:
            fn_call = load_calling("data/input/function_calling_tests.json")

        if "--output" in arg:
            idx = arg.index("--output")
            file_path = Path(arg[idx + 1])
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()
        else:
            file_path = Path("data/output/function_calling_results.json")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()
        if fn_call == [] or fn_def == []:
            raise ValueError()

    except ValueError as e:
        print(f"[PARSER - ERROR]: {e}", file=sys.stderr)
        sys.exit(1)

    except IndexError:
        print("You need to specify file path!")
        sys.exit(1)

    return fn_def, fn_call, file_path
