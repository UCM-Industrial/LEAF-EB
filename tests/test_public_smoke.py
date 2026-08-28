from pathlib import Path

import Runner


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INPUTS = [
    "Example_01_basic.yml",
    "Example_02_monte_carlo.yml",
    "Example_03_bess.yml",
    "Example_04_commodities.yml",
    "Example_05_nuclear_constant.yml",
    "Example_06_nuclear_load_following.yml",
]


def test_public_examples_are_discoverable() -> None:
    names = [path.name for path in Runner.discover_inputs()]
    assert names == EXPECTED_INPUTS


def test_all_public_examples_load_and_validate() -> None:
    for filename in EXPECTED_INPUTS:
        user_input, input_path = Runner.load_input(Path(filename).stem)
        assert input_path.name == filename
        Runner.validate_input(user_input, ROOT)


def test_monte_carlo_example_is_explicit() -> None:
    user_input, _ = Runner.load_input("Example_02_monte_carlo")
    monte_carlo = user_input["simulation"]["monte_carlo"]
    assert monte_carlo["simulations"] == 5
    assert monte_carlo["preserve_annual_targets"] is False
    assert monte_carlo["technology_uncertainty"] is False


def test_nuclear_examples_use_same_deployment() -> None:
    constant, _ = Runner.load_input("Example_05_nuclear_constant")
    following, _ = Runner.load_input("Example_06_nuclear_load_following")
    constant_nuclear = constant["sources"]["Nuclear_SMR300"]
    following_nuclear = following["sources"]["Nuclear_SMR300"]
    assert constant_nuclear["capacity_additions"] == (
        following_nuclear["capacity_additions"])
    assert constant_nuclear["hourly_operation"] == "must_run"
    assert following_nuclear["hourly_operation"] == "load_following"
