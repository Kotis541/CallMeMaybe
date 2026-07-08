from .models import FunctionDefinition
from typing import List, Optional, Dict


class State:
    """Represents the different states of the JSON generation state machine."""
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
    """Initializes the state machine for JSON generation."""

    _MAX_STRING_PARAM_LEN = 50
    _MAX_OTHER_PARAM_LEN = 10
    _FORBIDDEN_CHARS_IN_STRING = frozenset("{}[%$\\|<>\n")
    _VALID_CHARS_IN_OTHER = frozenset("0123456789.-e truefals")

    def __init__(
        self,
        available_fun: List[FunctionDefinition],
        original_prompt: str
    ):
        self.function = available_fun
        self.original_prompt = original_prompt
        self.state = State.START

        self.target = '{\n  "prompt": "'
        self.progress = ""

        self.selected_function: Optional[FunctionDefinition] = None
        self.param_keys_to_generate: List[str] = []
        self.current_param_type: str = ""

    def move(self, token_text: str) -> None:
        """Processes the next token and transitions the state machine."""
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
                    assert self.selected_function is not None
                    params = self.selected_function.parameters
                    self.current_param_type = params[param].type
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
            is_done = False
            delim_used = ""
            is_last = (len(self.param_keys_to_generate) == 0)

            expected_delims = ['"']
            if self.current_param_type != "string":
                expected_delims = ["}"] if is_last else [","]
            for d in expected_delims:
                if d in token_text:
                    is_done = True
                    delim_used = d
                    break

            if is_done:
                idx = token_text.index(delim_used)
                overlap = token_text[idx+1:]

                if not is_last:
                    param = self.param_keys_to_generate.pop(0)
                    assert self.selected_function is not None
                    params = self.selected_function.parameters
                    self.current_param_type = params[param].type
                    quote = '"' if self.current_param_type == "string" else ''
                    self.state = State.PARAM_KEY

                    if delim_used == ",":
                        self.target = f'\n    "{param}": {quote}'
                    else:
                        self.target = f',\n    "{param}": {quote}'

                    self.progress = overlap
                else:
                    self.state = State.CLOSE_BRACE
                    if delim_used == "}":
                        self.target = "\n}"
                    else:
                        self.target = "\n  }\n}"
                    self.progress = overlap

        elif self.state == State.CLOSE_BRACE:
            if self.progress == self.target:
                self.state = State.END

    def allow_target(
        self,
        vocab: Dict[int, str],
        progress: str,
        target_txt: str
    ) -> List[int]:
        """Filters vocabulary to find tokens that match the target string."""
        if target_txt.startswith(progress):
            remaining = target_txt[len(progress):]
        else:
            return []

        if not remaining:
            return []

        allowed = []
        for token_id, token_text in vocab.items():
            if not token_text:
                continue
            if remaining.startswith(token_text):
                allowed.append(int(token_id))

        if not allowed:
            for token_id, token_text in vocab.items():
                if not token_text:
                    continue
                stripped_token = token_text.lstrip()
                if remaining.startswith(stripped_token) and stripped_token:
                    allowed.append(int(token_id))

        if not allowed:
            next_char = remaining[0]
            for token_id, token_text in vocab.items():
                if token_text and token_text.startswith(next_char) \
                        and remaining.startswith(token_text):
                    allowed.append(int(token_id))

        return list(set(allowed))

    def _get_allowed_string_param_tokens(
        self, vocab: Dict[int, str], force_end: bool, has_char: bool
    ) -> List[int]:
        """
        Helper method to get allowed token IDs for a string parameter value.
        """
        allowed = []
        for token_id, token_text in vocab.items():
            if not token_text:
                continue

            if token_text == '"':
                if force_end or has_char or self.progress == "":
                    allowed.append(int(token_id))
            elif '"' in token_text:
                idx = token_text.index('"')
                prefix = token_text[:idx]
                suffix = token_text[idx+1:]

                if not all(c in " \n\r\t," for c in suffix):
                    continue

                if not \
                    any(c in self._FORBIDDEN_CHARS_IN_STRING for c in prefix) \
                        and (force_end or has_char or len(prefix.strip()) > 0):
                    allowed.append(int(token_id))

            elif (not force_end and not any(
                    c in self._FORBIDDEN_CHARS_IN_STRING for c in token_text)):
                allowed.append(int(token_id))
        return allowed

    def _get_allowed_other_param_tokens(
        self, vocab: Dict[int, str], force_end: bool, has_char: bool,
        end_tokens: List[str]
    ) -> List[int]:
        """
        Helper method to get allowed token IDs for number, boolean, etc.
        """
        allowed = []
        for token_id, token_text in vocab.items():
            if not token_text:
                continue

            is_ending_token = False
            for delim in end_tokens:
                if delim in token_text:
                    if token_text == delim:
                        allowed.append(int(token_id))
                        is_ending_token = True
                        break
                    idx = token_text.index(delim)
                    prefix = token_text[:idx]
                    suffix = token_text[idx+len(delim):]

                    if not all(c in " \n\r\t" for c in suffix):
                        continue

                    if all(c in self._VALID_CHARS_IN_OTHER for c in prefix):
                        if (force_end or has_char or
                                any(c.isalnum() for c in prefix)):
                            allowed.append(int(token_id))
                            is_ending_token = True
                    break

            if not force_end and not is_ending_token:
                if all(c in self._VALID_CHARS_IN_OTHER for c in token_text):
                    allowed.append(int(token_id))
        return allowed

    def get_allowed_tokens(
        self,
        vocab: Dict[int, str],
        text: str
    ) -> List[int]:
        """Gets a list of allowed token IDs based on the current state."""
        if self.state in [
            State.START, State.PROMPT_VALUE, State.NAME_KEY, State.PARAMS_KEY,
            State.PARAM_KEY, State.CLOSE_BRACE
        ]:
            return self.allow_target(vocab, self.progress, self.target)

        elif self.state == State.NAME_VALUE:
            allowed = []
            for f in self.function:
                target = f.name
                allowed.extend(self.allow_target(vocab, self.progress, target))
            return list(set(allowed))

        elif self.state == State.PARAM_VALUE:
            is_last = (len(self.param_keys_to_generate) == 0)
            end_tokens = ['"'] if self.current_param_type == "string" \
                else (["}"] if is_last else [","])

            has_char = len(self.progress.strip()) > 0

            force_end = (
                (self.current_param_type == "string" and
                 len(self.progress) > self._MAX_STRING_PARAM_LEN) or
                (self.current_param_type != "string" and
                 len(self.progress) > self._MAX_OTHER_PARAM_LEN)
            )

            if self.current_param_type == "string":
                allowed = self._get_allowed_string_param_tokens(
                    vocab,
                    force_end,
                    has_char)
            else:
                allowed = self._get_allowed_other_param_tokens(
                    vocab,
                    force_end,
                    has_char,
                    end_tokens)

            if not allowed and force_end:
                for token_id, token_text in vocab.items():
                    if token_text == end_tokens[0]:
                        allowed.append(int(token_id))
                        break

            return list(set(allowed))

        elif self.state == State.END:
            return []
        return []
