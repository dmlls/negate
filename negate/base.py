"""Base negator."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseNegator(ABC):
    """Base negator.

    Specific negators for different languages must inherit from this class.
    """

    @abstractmethod
    def __init__(
        self,
        use_transformers: Optional[bool] = None,
        use_gpu: Optional[bool] = None,
        fail_on_unsupported: Optional[bool] = None,
        log_level: Optional[int] = None,
        **kwargs,
    ):
        """Instanciate a :obj:`Negator`.

        Args:
            use_transformers (:obj:`Optional[bool]`, defaults to :obj:`False`):
                Whether to use a Transformer model for POS tagging and
                dependency parsing.
            use_gpu (:obj:`Optional[bool]`, defaults to :obj:`False`):
                Whether to use the GPU, if available. This parameter is ignored
                when :param:`use_transformers` is set to :obj:`False`.
            fail_on_unsupported (:obj:`Optional[bool]`, defaults to :obj:`False`):
                Whether to fail upon non-supported sentences. If set to
                :obj:`False`, a warning will be printed, and the sentence will
                try to be negated in a best-effort fashion.
            log_level (:obj:`Optional[int]`, defaults to ``logging.INFO``):
                The level of the logger.

        Raises:
            :obj:`RuntimeError`: If the sentence is not supported and
            :arg:`fail_on_unsupported` is set to :obj:`True`.
        """
        pass

    @abstractmethod
    def negate_sentence(
        self,
        sentence: str,
        **kwargs: Dict[str, Any],
    ) -> List[str]:
        """Negate a sentence.

        Affirmative sentences will be turned into negative ones and vice versa.

        Args:
            sentence (:obj:`str`):
                The sentence to negate.
            **kwargs (:obj:`Dict[str, Any]`):
                Additional parameters to pass to the concrete language negator.

        Returns:
            :obj:`List[str]`: The negated sentence(s).
        """
        pass
