*This project has been created as part of the 42 curriculum by vokotera.*

## Description
This project implements a robust function calling system for LLM. The goal is to process natural language and make 100% valid JSON. It takes a natural langauge prompt and uses small language model (Qwen3-0.6B) to map it to a specific predefined function signature

## Algorithm Explanation
The core of this project relies on **Constrained Decoding** powered by custom Finite State Machine

Instead of relying on the LLM to spontaneously generate valid JSON formatting, the generation process is intercepted token-by-token.
1. The machine tracks its current position in the JSON structure using predefined states
2. At each generation step, the state machine evaluates the model's vocabulary and filters out any tokens that would break the JSON syntax or violate the expected Pydantic schema types
3. The logits for invalid tokens are masked out by setting them to `-np.inf`.
4. The remaining highest-probability token is selected via `np.argmax`, appended to the string, and the state machine transitions accordingly until the `END` state is reached.

## Design Decisions
**Strict Type Validation:** Used `pydantic` heavily for defining and validating the input schemas. The configuration  s utilized to prevent any unexpected data structures.
* **Zero Machine Learning Dependencies in Main Config:** To comply with the strict project rules, libraries like `torch` and `transformers` are isolated within the provided `llm_sdk` module, which is imported into the project purely as a local dependency via `uv` in `pyproject.toml`.
* **Token Suffix/Prefix Analysis:** Instead of basic string matching, the token filtering algorithm calculates prefix/suffix overlaps (`overlap`) to handle tokens that might contain partial JSON delimiters or leading whitespace.

## Performance Analysis
* **Accuracy & Reliability:** The solution guarantees 100% valid JSON output and strict adherence to the provided function schema.
* **Speed:** Since the constrained decoding operates purely via Python string manipulation, vocabulary dictionary lookups, and fast NumPy array masking, the overhead added to the LLM's natural generation time is minimal.

## Challenges Faced
* **Dependency Management & Disk Space:** Installing the required `llm_sdk` fetched massive transitive dependencies like PyTorch and CUDA libraries, leading to `No space left on device` errors in the home directory. This was solved by configuring `uv` to use a dedicated cache directory on a larger storage partition.

## Testing Strategy
The logic was validated without spinning up the heavy LLM to ensure isolated testing of the constraints.
Using `pytest`, the `JSONStateMachine` is tested by feeding it simulated token chunks. These unit tests verify:
* Proper initialization and parsing of mocked `FunctionDefinition` objects.
* Correct transitions through all states (from `START` to `END`).
* Accurate vocabulary filtering (ensuring valid tokens are permitted and invalid ones are blocked).

## Instructions
This section covers the setup and execution of the project. The dependency management is handled via `uv`, and common tasks are automated using a `Makefile`.
### Installation
To install all required dependencies (including the isolated `llm_sdk` module), run:
```bash
make install
```

### Execution
```bash
make run 
or 
uv run python -m src [--functions_definition <function_definition_file>]
[--input <input_file>] [--output <output_file>]
```

### Debug and Linting
If you need to debug you can use: 
```bash
make debug
```
To run the strict type-checker and linter (verifying PEP 8 compliance and static types with flake8 and mypy):
```bash
make lint
```

## Example usage
**Input Prompt:**
```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  }
]
```

**Available Function Schema (Background context):**
```json
{
  "name": "fn_add_numbers",
  "description": "Add two numbers together and return their sum.",
  "parameters": {
    "a": { "type": "number" },
    "b": { "type": "number" }
  }
}
```

### What comes out:
Instead of a conversational answer (like "The sum is 5"), the constrained LLM forces the output into a strictly formatted JSON array containing the resolved tool calls.

**Generated Output (saved to `data/output/function_calling_results.json`):**
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2,
      "b": 3
    }
  }
]
```

## Resources
[Youtube - Finite State Machine](https://www.youtube.com/watch?v=-ZP2Xm-mY4E&t=95s)

[Youtube - Structured LLM outputs](https://www.youtube.com/watch?v=xpvFinvqRCA&t=732s)

[Youtube - another LLM video](https://www.youtube.com/watch?v=EiMPQsI2__Y&t=972s)

### AI Usage Declaration
* **Design:** Assisting with the architectural design of the finite state machine and planning the dependency management strategy to comply with the project's strict rules.
* **Implementation:** Helping to debug terminal state transitions, structure the core parsing logic, and set up the validation and testing environment.
* **No Copy-Pasting:** No core algorithm logic or code was blindly copy-pasted. All AI-assisted suggestions were critically assessed, thoroughly understood, and manually integrated to ensure full comprehension of the underlying mechanics.