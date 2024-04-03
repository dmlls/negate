"""High-level negator."""

import importlib
from pathlib import Path
from typing import Dict, List, Optional

from .negators.supported_languages import Language


class Negator:
    """High-level negator."""

    def __init__(
        self,
        language: str,
        *,
        use_transformers: Optional[bool] = None,
        use_gpu: Optional[bool] = None,
        fail_on_unsupported: Optional[bool] = None,
        log_level: Optional[int] = None,
        **kwargs: Dict,
    ):
        """Instanciate a :obj:`Negator`.

        Args:
            use_transformers (:obj:`Optional[bool]`, defaults to :obj:`False`):
                Whether to use a Transformer model for POS tagging and
                dependency parsing.

                .. note::

                   When set to :obj:`True` the model `en_core_web_trf
                   <https://spacy.io/models/en#en_core_web_trf>`__ is used.
            use_gpu (:obj:`Optional[bool]`, defaults to :obj:`False`):
                Whether to use the GPU, if available. This parameter is
                ignored when :param:`use_transformers` is set to :obj:`False`.
            fail_on_unsupported (:obj:`Optional[bool]`, defaults to :obj:`False`):
                Whether to fail upon non-supported sentences. If set to
                :obj:`False`, a warning will be printed, and the negator
                will try to negate the sentence in a best-effort fashion.
            log_level (:obj:`Optional[int]`, defaults to ``logging.INFO``):
                The level of the logger.
            kwargs (:obj:`Dict`):
                Any other parameters to pass to the language-specific
                negators.

        Raises:
            :obj:`ValueError`: If the specified language is not supported.
        """
        if not Language.is_supported(language):
            raise ValueError(
                f'The language "{language}" is currently not supported.\n'
                f"Valid values are {Language.get_all()}"
            )
        self.language = language
        self.negator = getattr(
            importlib.import_module(
                f".negators.{language}.negator", package=Path(__file__).parent.name
            ),
            "Negator",
        )(
            use_transformers=use_transformers,
            use_gpu=use_gpu,
            fail_on_unsupported=fail_on_unsupported,
            log_level=log_level,
            **kwargs,
        )

    def negate_sentence(self, sentence: str, **kwargs) -> List[str]:
        return self.negator.negate_sentence(sentence, **kwargs)
