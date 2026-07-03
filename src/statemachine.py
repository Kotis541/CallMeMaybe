import numpy as np
from .classes import FunctionDefinition
from typing import List, Optional, Dict

class State:
    START = "START"
    OPEN_BRACE = "OPEN_BRACE"
    PROMPT_KEY = "PROMPT_KEY"
    PROMPT_VALUE = "PROMPT_VALUE"  
    NAME_KEY = "NAME_KEY"
    NAME_VALUE = "NAME_VALUE"
    PARAMS_KEY = "PARAMS_KEY"
    PARAM_KEY_NAME = "PARAM_KEY_NAME"
    PARAM_VALUE = "PARAM_VALUE" 
    ClOSE_BRACE = "CLOSE_BRACE" 
    FINISHING = "FINISHING"
    END = "END"


class JSONStateMachine:
    def __init__(self, available_fun: List[FunctionDefinition], original_prompt: str):
        self.function = available_fun
        self.original_prompt = original_prompt
        self.state = State.START
        self.target = "["
        self.progress = ""

        self.selected_function: Optional[FunctionDefinition] = None
        self.param_keys_to_generate: List[str] = []
        self.current_param_type: str = ""
    
    def move(self, token_text: str):
        self.progress += token_text
    
        if self.state == State.START and self.progress == self.target:
            self.state = State.OPEN_BRACE
            self.target = "\t{"
            self.progress = ""

        elif self.state == State.OPEN_BRACE and self.progress == self.target:
            self.state = State.PROMPT_KEY
            self.target = '\t  "prompt": "'
            self.progress = ""
        
        elif self.state == State.PROMPT_KEY and self.progress == self.target:
            self.state = State.PROMPT_VALUE
            self.target = self.original_prompt + '", '
            self.progress = ""
        
        elif self.state == State.PROMPT_VALUE and self.progress == self.target:
            self.state = State.NAME_KEY
            self.target = '\t  "name": "'
            self.progress = ""

        elif self.state == State.NAME_KEY and self.progress == self.target:
            self.state = State.NAME_VALUE
            self.target = ""
            self.progress = ""
        
        elif self.state == State.NAME_VALUE and '"' in self.progress:
            func_name = self.progress.replace('"', '').replace(',', '')

            for f in self.function:
                if f.name == func_name:
                    self.selected_function = f
                    self.param_keys_to_generate = list(f.parameters.keys())
                    break
            self.state = State.PARAMS_KEY
            self.target = '\t  "parameters": {'
            self.progress = ""
        
        elif self.state == State.PARAMS_KEY and self.progress == self.target:
            self.state = State.PARAM_KEY_NAME
            self.progress = ""
        
        elif self.state == State.PARAM_KEY_NAME:
            self.state = State.PARAM_VALUE
            self.target = ""
            self.progress = ""
        
        elif self.state == State.PARAM_VALUE and self.progress.count('"') == 2:
            if len(self.param_keys_to_generate) > 0:
                next_param = self.param_keys_to_generate.pop(0)
                self.state = State.PARAM_KEY_NAME
                self.target = f', "{next_param}": '
                self.progress = ""
        
            else:
                self.state = State.ClOSE_BRACE
                self.target = "}"
                self.progress = ""
        
        elif self.state == State.ClOSE_BRACE and self.progress == self.target:
            self.state = State.FINISHING
            self.target = "]\n"
            self.progress = ""

        elif "]" in self.progress:
            self.state = State.END

        elif self.state == State.FINISHING and self.progress == self.target:
            self.state = State.END
    
    def allow_target(self, vocab: dict, txt: str, target_txt: str):
        allowed = []

        writed = ""
        for i in range(len(target_txt), 0, -1):
            if txt.endswith(target_txt[:i]):
                writed = target_txt[:i]
        
        remaining = target_txt[len(writed):]

        if not remaining:
            return []

        for token_id, token_text in vocab.items():
            if not token_text:
                continue
            
            if remaining.startswith(token_text) or token_text.startswith(remaining):
                allowed.append(token_id)
        
        return allowed

    def get_allowed_tokens(self, vocab: dict, text: str) -> List[int]:

        if self.state == State.START:
            return self.allow_target(vocab, text, "[")
        elif self.state == State.OPEN_BRACE:
            return self.allow_target(vocab, text, "\t{")
        
        elif self.state == State.PROMPT_KEY:
            return self.allow_target(vocab, text, '\t  "prompt": "')
        
        elif self.state == State.PROMPT_VALUE:
            return self.allow_target(vocab, text, self.original_prompt + '", ')
        
        elif self.state == State.NAME_KEY:
            return self.allow_target(vocab, text, '\t  "name": "')
        
        elif self.state == State.NAME_VALUE:
            allowed_ids = []

            for func_name in self.function:
                target = f'"{func_name.name}"'
                allowed_ids.extend(self.allow_target(vocab, text, target))
            
            return list(set(allowed_ids))
        
        elif self.state == State.PARAMS_KEY:
            return self.allow_target(vocab, text, '\t  "parameters": {')
        
        # elif self.state == State.PARAM_KEY_NAME:
        #     for func_name in self.function:
        #         if func_name.name == self.selected_function.name:
        #             target = f'"{self.param_keys_to_generate[0]}": '
        #     return self.allow_target(vocab, text, target)

        elif self.state == State.PARAM_VALUE:
            if self.current_param_type == "number":
                return self.allow_target(vocab, text, target)
            

        elif self.state == State.END:
            return []
