import sys
import json
from typing import Any, List
import numpy as np
from src.parser import parse
from llm_sdk.llm_sdk import Small_LLM_Model

def load_vocabulary(model: Small_LLM_Model):
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, 'r', encoding="utf-8") as f:
        vocab = json.load(f)
        return vocab


def main():
    model = Small_LLM_Model()
    fn_def, fn_call = parse(sys.argv)

    allowed_ids = []
    vocab = load_vocabulary(model)

    test_token = model.encode("{").tolist()[0]
    text = model.decode(test_token)
    print(f"Encode: {test_token}, text: {text} ")

    for id, id_text in vocab.items():
        if id == 90:
            print(f"ID: {id}, text: {id_text}")
    
if __name__ == "__main__":
    main()
