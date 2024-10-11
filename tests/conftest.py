"""Set up tests."""

from contextlib import suppress

from negate import Negator

# Reuse negator model for all tests.
negator_model: Negator = None


def pytest_addoption(parser):
    parser.addoption(
        "--transformers", action="store_true", help="use Transformers")
    parser.addoption(
        "--use-cpu", action="store_true", help="use CPU instead of GPU")


def pytest_generate_tests(metafunc):
    if "negator" in metafunc.fixturenames:
        # Initialize negator only once.
        global negator_model
        if negator_model is None:
            use_transformers = metafunc.config.getoption("transformers")
            if not metafunc.config.getoption("use_cpu"):
                with suppress(ValueError, NotImplementedError):
                    # `use_gpu` ignored if `use_transformers` is False.
                    negator_model = Negator(use_transformers=use_transformers,
                                            use_gpu=True)
                    # If GPU is unsupported, we fallback to CPU.
                    negator_model.negate_sentence("I will now check GPU support!")
            else:
                negator_model = Negator(
                    use_transformers=use_transformers, use_gpu=False)
        metafunc.parametrize("negator", [negator_model])
