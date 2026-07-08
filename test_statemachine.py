import pytest
from src.json_generator import JSONStateMachine, State
from src.models import FunctionDefinition, TypeDetails


@pytest.fixture
def mock_functions():
    """Provides a list of mock FunctionDefinition objects for testing."""
    return [
        FunctionDefinition(
            name="fn_add_numbers",
            description="Adds two numbers.",
            parameters={
                "a": TypeDetails(type="number"),
                "b": TypeDetails(type="number")
            },
            returns=TypeDetails(type="number")
        ),
        FunctionDefinition(
            name="fn_greet",
            description="Greets a person.",
            parameters={
                "name": TypeDetails(type="string")
            },
            returns=TypeDetails(type="string")
        ),
        FunctionDefinition(
            name="fn_get_time",
            description="Returns the current time.",
            parameters={},
            returns=TypeDetails(type="string")
        )
    ]


def test_initial_state(mock_functions):
    """Tests that the state machine initializes in the correct state."""
    prompt = "What is 5 + 5?"
    machine = JSONStateMachine(mock_functions, prompt)

    assert machine.state == State.START
    assert machine.target == '{\n  "prompt": "'


def test_full_generation_cycle_numbers(mock_functions):
    """Tests a full generation cycle with numeric parameters."""
    prompt = "What is 5 + 5?"
    machine = JSONStateMachine(mock_functions, prompt)

    steps = [
        '{\n  "prompt": "',
        'What is 5 + 5?',
        '",\n  "name": "',
        'fn_add_numbers',
        '",\n  "parameters": {',
        '\n    "a": ',
        '5.0,',
        '\n    "b": ',
        '5.0\n  }\n}',
        ''
    ]

    for chunk in steps:
        machine.move(chunk)

    assert machine.state == State.END


def test_full_generation_cycle_string(mock_functions):
    """Tests a full generation cycle with a string parameter."""
    prompt = "Greet John"
    machine = JSONStateMachine(mock_functions, prompt)

    steps = [
        '{\n  "prompt": "',
        'Greet John',
        '",\n  "name": "',
        'fn_greet',
        '",\n  "parameters": {',
        '\n    "name": "',
        'John"\n  }\n}',
        ''
    ]

    for chunk in steps:
        machine.move(chunk)

    assert machine.state == State.END


def test_full_generation_cycle_no_params(mock_functions):
    """Tests a full generation cycle for a function with no parameters."""
    prompt = "What time is it?"
    machine = JSONStateMachine(mock_functions, prompt)

    steps = [
        '{\n  "prompt": "',
        'What time is it?',
        '",\n  "name": "',
        'fn_get_time',
        '",\n  "parameters": {',
        '\n  }\n}',
        ''
    ]

    for chunk in steps:
        machine.move(chunk)

    assert machine.state == State.END


def test_allowed_tokens_filtering(mock_functions):
    """Tests the logic for filtering allowed tokens from the vocabulary."""
    machine = JSONStateMachine(mock_functions, "Test")

    vocab = {
        "1": "{\n",
        "2": '  "prompt": "',
        "3": "bullshit",
        "4": "{\n  \"prompt\": \""
    }

    allowed = machine.get_allowed_tokens(vocab, "")

    assert 1 in allowed
    assert 4 in allowed
    assert 3 not in allowed


def test_allowed_tokens_name_value(mock_functions):
    """Tests token filtering when generating a function name."""
    machine = JSONStateMachine(mock_functions, "Test")
    machine.state = State.NAME_VALUE

    vocab = {
        1: "fn_add",
        2: "_numbers",
        3: "fn_greet",
        4: "fn_nonexistent"
    }

    # With empty progress, all function names are possible targets
    allowed = machine.get_allowed_tokens(vocab, "")
    assert 1 in allowed
    assert 3 in allowed
    assert 4 not in allowed

    # With progress, it should only allow continuations
    machine.progress = "fn_add"
    allowed = machine.get_allowed_tokens(vocab, machine.progress)
    assert 2 in allowed
    assert 1 not in allowed
    assert 3 not in allowed


def test_allowed_tokens_param_value_string(mock_functions):
    """Tests token filtering when generating a string parameter value."""
    machine = JSONStateMachine(mock_functions, "Test")
    machine.state = State.PARAM_VALUE
    machine.current_param_type = "string"

    vocab = {
        1: "hello",
        2: ' world"',
        3: 'contains{char',
        4: '"',
        5: 'unfinished'
    }

    # Allow partial strings and the closing quote
    allowed = machine.get_allowed_tokens(vocab, 'partial_')
    assert 1 in allowed
    assert 2 in allowed
    assert 4 in allowed
    assert 5 in allowed
    assert 3 not in allowed  # Contains forbidden character


def test_allowed_tokens_param_value_number(mock_functions):
    """Tests token filtering when generating a number parameter value."""
    machine = JSONStateMachine(mock_functions, "Test")
    machine.state = State.PARAM_VALUE
    machine.current_param_type = "number"
    machine.param_keys_to_generate = []  # This is the last parameter

    vocab = {
        1: "123",
        2: "456}",
        3: "invalid-char!",
        4: "}",
        5: "9.9"
    }

    allowed = machine.get_allowed_tokens(vocab, "1.")
    assert 1 in allowed
    assert 2 in allowed
    assert 4 in allowed
    assert 5 in allowed
    assert 3 not in allowed
