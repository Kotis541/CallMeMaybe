import pytest
from src.json_generator import JSONStateMachine, State
from src.models import FunctionDefinition, TypeDetails


@pytest.fixture
def mock_functions():
    """Provides a list of mock FunctionDefinition objects for testing."""
    return [
        FunctionDefinition(
            name="fn_add_numbers",
            description="Spocita dve cisla.",
            parameters={
                "a": TypeDetails(type="number"),
                "b": TypeDetails(type="number")
            },
            returns=TypeDetails(type="number")
        ),
        FunctionDefinition(
            name="fn_greet",
            description="Pozdravi cloveka.",
            parameters={
                "name": TypeDetails(type="string")
            },
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
