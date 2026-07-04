import sys
import json
import numpy as np
from .parser import parse
from llm_sdk.llm_sdk import Small_LLM_Model
from .statemachine import JSONStateMachine, State
from .classes import load_output



def load_vocabulary(model: Small_LLM_Model):
    vocab_path = model.get_path_to_vocab_file()
    decoded = {}
    with open(vocab_path, 'r', encoding="utf-8") as f:
        vocab = json.load(f)

    for id_text, id in vocab.items():
        decoded[id] = model.decode([id])

    return decoded


def main():
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
            context = "Available tools:\n" + fn_defs_str + "\n\nTask: " + call.prompt + "\n\nOutput:\n" + text
            input_ids = model.encode(context).tolist()[0]
            logits = np.array(model.get_logits_from_input_ids(input_ids))
            allowed_ids = machine.get_allowed_tokens(vocab, text)
            
            mask = np.full_like(logits, -np.inf)
            mask[allowed_ids] = logits[allowed_ids]
            logits = mask
            best_id = np.argmax(logits)
            token_text = model.decode(best_id)
            machine.move(token_text)
            print(token_text, end="")
            text += token_text

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
    main()