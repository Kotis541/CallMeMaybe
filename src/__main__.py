import sys
import json
import os
import time
import numpy as np
from .cli import parse
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from .json_generator import JSONStateMachine, State
from .models import load_output
from typing import Dict


def load_vocabulary(model: Small_LLM_Model) -> Dict[int, str]:
    """Loads and decodes the vocabulary from the model's vocab file."""
    vocab_path = model.get_path_to_vocab_file()
    decoded = {}
    with open(vocab_path, 'r', encoding="utf-8") as f:
        vocab = json.load(f)

    for id_text, id in vocab.items():
        decoded[id] = model.decode([id])

    return decoded


def main() -> None:
    """Main function to run the constrained decoding process."""
    model = Small_LLM_Model()
    fn_defs, fn_calls, filepath = parse(sys.argv)
    vocab = load_vocabulary(model)

    fn_defs_for_prompt = [f.model_dump(exclude={'returns'}) for f in fn_defs]
    fn_defs_str = json.dumps(fn_defs_for_prompt, indent=2)

    all_outputs = []

    for call in fn_calls:
        safe_prompt = json.dumps(call.prompt)[1:-1]
        machine = JSONStateMachine(fn_defs, safe_prompt)
        text = ""

        while machine.state != State.END:
            context = "Available tools:\n" + fn_defs_str + "\n\nTask: " \
                  + call.prompt + "\n\nOutput:\n" + text
            input_ids = model.encode(context).tolist()[0]
            logits = np.array(model.get_logits_from_input_ids(input_ids))
            allowed_ids = machine.get_allowed_tokens(vocab, text)

            mask = np.full_like(logits, -np.inf)
            mask[allowed_ids] = logits[allowed_ids]
            logits = mask
            best_id = np.argmax(logits)
            token_text = model.decode([int(best_id)])
            machine.move(token_text)
            text += token_text

            # -- Visualization --
            os.system('cls' if os.name == 'nt' else 'clear')
            C_PURPLE = '\033[95m'
            C_CYAN = '\033[96m'
            C_YELLOW = '\033[93m'
            C_GREEN = '\033[92m'
            C_RED = '\033[91m'
            C_RESET = '\033[0m'

            print(f"{C_PURPLE}╔═════════════════════════════════╗{C_RESET}")
            print(f"{C_PURPLE}║ LLM CONSTRAINED DECODING ENGINE ║{C_RESET}")
            print(f"{C_PURPLE}╚═════════════════════════════════╝{C_RESET}\n")

            print(f"{C_CYAN}▶ Prompt:{C_RESET} {call.prompt}")
            print(f"{C_YELLOW}▶ Current State:{C_RESET} {machine.state}")

            print(f"{C_RED}▶ Target Sequence:{C_RESET} {repr(machine.target)}")
            print(f"{C_CYAN}▶ Allowed Tokens:{C_RESET} \
                  {len(allowed_ids)} / {len(vocab)}\n")

            print(f"{C_PURPLE}--- Generated Output So Far ---{C_RESET}")
            print(f"{C_GREEN}{text}{C_RESET}")

            time.sleep(0.05)

        cleaned_text = text.strip("[\n]")
        try:
            parsed_json = json.loads(cleaned_text)
            all_outputs.append(parsed_json)
        except json.JSONDecodeError as e:
            print(f"[ERROR]: Json is not valud: {e}")

    with open(filepath, 'w', encoding="utf-8") as file:
        json.dump(all_outputs, file, indent=2)

    load_output(filepath)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR]: An unexpected error occurred: {e}",
              file=sys.stderr)
        sys.exit(1)
