"""Test utilities."""

from contextlib import suppress

from negate import Negator


def set_up_negator(request, language: str) -> Negator:
    """Instanciate the negator to be tested.

    Args:
        request (:obj:``):
        language (:obj:`str`):
            The language of the negator.
    """
    use_transformers = request.config.getoption("transformers")
    if not request.config.getoption("use_cpu"):
        with suppress(ValueError, NotImplementedError):
            # `use_gpu` ignored if `use_transformers` is False.
            negator = Negator(
                language=language, use_transformers=use_transformers, use_gpu=True
            )
            # If GPU is unsupported, we fallback to CPU.
            negator.negate_sentence("I will now check GPU support!")
            return negator
    return Negator(language=language, use_transformers=use_transformers, use_gpu=False)
