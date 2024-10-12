"""Set up tests."""


def pytest_addoption(parser):
    parser.addoption("--transformers", action="store_true", help="use Transformers")
    parser.addoption("--use-cpu", action="store_true", help="use CPU instead of GPU")
