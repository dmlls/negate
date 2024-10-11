"""Base negator."""

import importlib
import os
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import spacy
from spacy.tokens import Doc as SpacyDoc

from .utils.tokens import Token


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

    def _initialize_spacy_model(
        self, model_name: str, model_version: str, **kwargs
    ) -> spacy.language.Language:
        """Initialize the spaCy model to be used by the Negator.

        .. note::

           Unfortunately, direct URLs are not allowed by PyPi (see
           https://github.com/pypa/pip/issues/6301). This means that we cannot
           specify the models as a dependency via direct URLs, as recommended by
           spaCy (see https://spacy.io/usage/models#download-pip). We work
           around this by downloading the models through the spaCy CLI when the
           negator is first instanciated.

        Args:
            model_name (:obj:`str`):
                The spaCy model to load (e.g., ``en_core_web_md``).
            model_version (:obj:`str`):
                The version of the spaCy model.
            **kwargs:
                Additional arguments passed to :func:`spacy.load`.

        Returns:
            :obj:`spacy.language.Language`: The loaded spaCy model, ready to
            use.
        """

        # See https://stackoverflow.com/a/25061573/14683209
        # We don't want the messages coming from pip "polluting" stdout.
        @contextmanager
        def suppress_stdout():
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    yield
                finally:
                    sys.stdout = old_stdout

        # First check if optional dependencies are installed.
        if model_name.endswith("_trf"):
            try:
                importlib.import_module("spacy_transformers")
            except ModuleNotFoundError:
                self.logger.error(
                    "Dependencies for Transformers missing. "
                    "Install them with:\n\n"
                    ' pip install "negate[transformers]"\n'
                )
                sys.exit(1)
        try:  # Model installed?
            model_module = importlib.import_module(model_name)
        except ModuleNotFoundError:  # Download and install model.
            self.logger.info(
                "Downloading spaCy model. This only needs to happen "
                "once. Please, be patient..."
            )
            with suppress_stdout():
                spacy.cli.download(f"{model_name}-{model_version}", True, False, "-q")
            model_module = importlib.import_module(model_name)
        spacy_model = model_module.load(**kwargs)
        installed_model_version: str = spacy_model.meta["version"]
        if installed_model_version != model_version:
            self.logger.info(
                "Updating spaCy model to version %s." " Please, be patient...",
                model_version,
            )
            with suppress_stdout():
                spacy.cli.download(f"{model_name}-{model_version}", True, False, "-q")
            model_module = importlib.import_module(model_name)
        return model_module.load(**kwargs)

    def _compile_sentence(
        self,
        doc: SpacyDoc,
        remove_tokens: Optional[List[int]] = None,
        add_tokens: Optional[Dict[int, Token]] = None,
    ) -> str:
        """Process and de-tokenize a spaCy document back into a string.

        Args:
            doc (:obj:`SpacyDoc`):
                The spaCy document.
            remove_tokens (:obj:`Optional[List[int]]`):
                The indexes of the tokens to remove from the document, if any.
            add_tokens (:obj:`Optional[Dict[int, Token]]`):
                The tokens to add to the document, if any. These are specified
                as a dictionary whose keys are the indexes in which to insert
                the new tokens, which are the respective values.

        Returns:
            :obj:`str`: The resulting, de-tokenized string including the
            removal/addition of tokens, if any.
        """
        if remove_tokens is None:
            remove_tokens = []
        if add_tokens is None:
            add_tokens = {}
        else:
            add_tokens = dict(sorted(add_tokens.items()))  # sort by index
        tokens = [Token(tk.text, tk._.has_space_after) for tk in doc]
        for i in remove_tokens:
            tokens[i] = Token(text="", has_space_after=False)
        for count, item in enumerate(add_tokens.items()):
            i, tk = item
            tokens.insert(i + count, tk)
        return self._capitalize_first_letter(
            self._remove_extra_whitespaces(
                "".join([f"{tk.text}{' '*int(tk.has_space_after)}" for tk in tokens])
            )
        )
