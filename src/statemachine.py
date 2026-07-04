import numpy as np
from .classes import FunctionDefinition
from typing import List, Optional

class State:
    START = "START"
    PROMPT_VALUE = "PROMPT_VALUE"  
    NAME_KEY = "NAME_KEY"
    NAME_VALUE = "NAME_VALUE"
    PARAMS_KEY = "PARAMS_KEY"
    PARAM_KEY = "PARAM_KEY"
    PARAM_VALUE = "PARAM_VALUE" 
    CLOSE_BRACE = "CLOSE_BRACE" 
    END = "END"


class JSONStateMachine:
    def __init__(self, available_fun: List[FunctionDefinition], original_prompt: str):
        self.function = available_fun
        self.original_prompt = original_prompt
        self.state = State.START
        
        self.target = '{\n  "prompt": "'
        self.progress = ""

        self.selected_function: Optional[FunctionDefinition] = None
        self.param_keys_to_generate: List[str] = []
        self.current_param_type: str = ""
    
    def move(self, token_text: str):
        self.progress += token_text
    
        if self.state == State.START:
            if self.progress == self.target:
                self.state = State.PROMPT_VALUE
                self.target = self.original_prompt
                self.progress = ""

        elif self.state == State.PROMPT_VALUE:
            if self.progress == self.target:
                self.state = State.NAME_KEY
                self.target = '",\n  "name": "'
                self.progress = ""

        elif self.state == State.NAME_KEY:
            if self.progress == self.target:
                self.state = State.NAME_VALUE
                self.progress = ""
        
        elif self.state == State.NAME_VALUE:
            for f in self.function:
                if self.progress == f.name:
                    self.selected_function = f
                    self.param_keys_to_generate = list(f.parameters.keys())
                    self.state = State.PARAMS_KEY
                    self.target = '",\n  "parameters": {'
                    self.progress = ""
                    break
        
        elif self.state == State.PARAMS_KEY:
            if self.progress == self.target:
                if len(self.param_keys_to_generate) > 0:
                    param = self.param_keys_to_generate.pop(0)
                    self.current_param_type = self.selected_function.parameters[param].get("type", "string")
                    quote = '"' if self.current_param_type == "string" else ''
                    self.state = State.PARAM_KEY
                    self.target = f'\n    "{param}": {quote}'
                    self.progress = ""
                else:
                    self.state = State.CLOSE_BRACE
                    self.target = "\n  }\n}"
                    self.progress = ""

        elif self.state == State.PARAM_KEY:
            if self.progress == self.target:
                self.state = State.PARAM_VALUE
                self.progress = ""
        
        elif self.state == State.PARAM_VALUE:
            # This is the robust fix: check the entire accumulated progress, not just the last token.
            clean_progress = self.progress.strip()
            is_done = False
            delim_used = ""
            is_last = (len(self.param_keys_to_generate) == 0)
            

            if self.current_param_type == "string":
                if '"' in token_text:
                # A string is done if it's fully wrapped in quotes.
                if clean_progress.startswith('"') and clean_progress.endswith('"') and len(clean_progress) > 1:
                    is_done = True
                    delim_used = '"'
            else:
                expected_delims = ["}"] if is_last else [","]
                for d in expected_delims:
                    if d in token_text:
                        is_done = True
                        delim_used = d
                        break
            else: # For numbers or booleans
                # They are done if a delimiter appears after them.
                if clean_progress.endswith(',') or clean_progress.endswith('}'):
                    is_done = True

            if is_done:
                idx = token_text.index(delim_used)
                overlap = token_text[idx+1:]

                if not is_last:
                # The value is complete. Now, transition cleanly.
                if len(self.param_keys_to_generate) > 0:
                    # There are more parameters to process.
                    param = self.param_keys_to_generate.pop(0)
                    self.current_param_type = self.selected_function.parameters[param].get("type", "string")
                    quote = '"' if self.current_param_type == "string" else ''
                    self.state = State.PARAM_KEY
                    
                    if delim_used == ",":
                        self.target = f'\n    "{param}": {quote}'
                    else:
                        self.target = f',\n    "{param}": {quote}'
                        
                    self.progress = overlap
                    self.target = f',\n    "{param}": {quote}' # Note the comma for separation
                    self.progress = "" # Reset progress for the new key
                else:
                    # This was the last parameter.
                    self.state = State.CLOSE_BRACE
                    if delim_used == "}":
                        self.target = "\n}"
                    else:
                        self.target = "\n  }\n}"
                    self.progress = overlap
                    self.target = "\n  }\n}"
                    self.progress = ""
        
        elif self.state == State.CLOSE_BRACE:
            if self.progress == self.target:
                self.state = State.END
    
    def allow_target(self, vocab: dict, progress: str, target_txt: str) -> List[int]:
        if not target_txt.startswith(progress):
            writed = ""
            for i in range(len(target_txt), 0, -1):
                if progress.endswith(target_txt[:i]):
                    writed = target_txt[:i]
                    break
            remaining = target_txt[len(writed):]
        else:
            remaining = target_txt[len(progress):]

        if not remaining:
            return []

        allowed = []
        for token_id, token_text in vocab.items():
            if not token_text: continue
            if remaining.startswith(token_text):
                allowed.append(int(token_id))
        
        return allowed

    def get_allowed_tokens(self, vocab: dict, text: str) -> List[int]:
        if self.state in [State.START, State.PROMPT_VALUE, State.NAME_KEY, State.PARAMS_KEY, State.PARAM_KEY, State.CLOSE_BRACE]:
            return self.allow_target(vocab, self.progress, self.target)

        elif self.state == State.NAME_VALUE:
            allowed = []
            for f in self.function:
                target = f.name
                allowed.extend(self.allow_target(vocab, self.progress, target))
            return list(set(allowed))
        
        elif self.state == State.PARAM_VALUE:
            allowed = []
            is_last = (len(self.param_keys_to_generate) == 0)
            end_tokens = ['"'] if self.current_param_type == "string" else (["}"] if is_last else [","])

            has_char = len(self.progress.strip()) > 0
            has_digit = any(c.isdigit() for c in self.progress)
            force_end = False

            if self.current_param_type != "string" and len(self.progress) > 3:
                force_end = True

            for token_id, token_text in vocab.items():
                if not token_text: continue
                
                if self.current_param_type == "string":
                    forbidden = "{}[%$\\|<>\n"
                    if '"' in token_text:
                        idx = token_text.index('"')
                        prefix = token_text[:idx]
                        if has_char or len(prefix.strip()) > 0:
                            allowed.append(int(token_id))
                    elif not any(c in forbidden for c in token_text):
                        allowed.append(int(token_id))
                else:
                    valid_chars = set("0123456789.-e truefals")
                    if force_end:
                        for delim in end_tokens:
                            if delim in token_text:
                                idx = token_text.index(delim)
                                prefix = token_text[:idx]
                                if all(c in valid_chars for c in prefix):
                                    allowed.append(int(token_id))
                                break
                        continue

            # Simplified and more robust logic for allowed tokens.
            if self.current_param_type == "string":
                # If we haven't started the string, we must force a quote.
                if not self.progress.strip():
                    return [id for id, tok in vocab.items() if tok == '"']
                # Once started, allow almost anything. The 'move' logic will find the end.
                return list(vocab.keys())
            else: # For numbers/booleans
                # Allow digits, relevant characters, and the delimiters.
                valid_chars = set("0123456789.-e,}\n\t ")
                allowed = []
                for token_id, token_text in vocab.items():
                    if not token_text: continue
                    if all(c in valid_chars for c in token_text):
                        allowed.append(int(token_id))
                        continue
                return allowed

                    for delim in end_tokens:
                        if delim in token_text:
                            idx = token_text.index(delim)
                            prefix = token_text[:idx]
                            if all(c in valid_chars for c in prefix):
                                if has_char or any(c.isalnum() for c in prefix):
                                    allowed.append(int(token_id))
                            break

            if not allowed and force_end:
                for token_id, token_text in vocab.items():
                    if token_text == end_tokens[0]:
                        allowed.append(int(token_id))
                        break
            
            return list(set(allowed))

        elif self.state == State.END:
            return []