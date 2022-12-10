"""Set up tests."""

from negate import Negator


# Reuse negator model for all tests.
negator_model: Negator = None


def pytest_addoption(parser):
    parser.addoption(
        "--transformers", action="store_true", help="use Transformers")


def pytest_generate_tests(metafunc):
    if "negator" in metafunc.fixturenames:
        # Initialize negator only once.
        global negator_model
        if negator_model is None:
            use_transformers = metafunc.config.getoption("transformers")
            try:
                # `use_gpu` ignored if `use_transformers` is False.
                negator_model = Negator(
                    use_transformers=use_transformers, use_gpu=True)
            except ValueError:  # "No GPU devices detected"
                negator_model = Negator(
                    use_transformers=use_transformers, use_gpu=False)
        metafunc.parametrize("negator", [negator_model])
