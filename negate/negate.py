"""Negation tools."""

import logging
import importlib
import os
import sys
import spacy
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union
from lemminflect import getInflection, getLemma
from spacy.symbols import AUX, neg, VERB
from spacy.tokens import Doc as SpacyDoc
from spacy.tokens import Token as SpacyToken
from .tokens import Token


class Negator:
    """Negator for the English language."""

    def __init__(
        self,
        use_transformers: Optional[bool] = None,
        use_gpu: Optional[bool] = None,
        fail_on_unsupported: Optional[bool] = None,
        log_level: Optional[int] = None
    ):
        """Instanciate a :obj:`Negator`.

        Args:
            use_transformers (:obj:`Optional[bool]`, defaults to :obj:`False`):
                Whether to use a Transformer model for POS tagging and
                dependency parsing.

                .. note::

                   When set to :obj:`True` the model <en_core_web_trf
                   `https://spacy.io/models/en#en_core_web_trf`>__ is used.
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
        if use_transformers is None:
            use_transformers = False
        if use_gpu is None:
            use_gpu = False
        if fail_on_unsupported is None:
            fail_on_unsupported = False
        if log_level is None:
            log_level = logging.INFO

        # Set up logger.
        logging.basicConfig(
            format="%(name)s - %(levelname)s: %(message)s",
            level=log_level
        )
        self.logger = logging.getLogger(__class__.__name__)
        self.fail_on_unsupported = fail_on_unsupported
        # Load spaCy model. If not available locally, the model will be first
        # installed.
        if use_transformers and use_gpu:
            spacy.require_gpu()
        self.spacy_model = self._initialize_spacy_model(use_transformers)
        # Initialize AUX negation dictionary.
        self._initialize_aux_negations()
        # Store whether tokens have a whitespace after them. This is used later
        # on for detokenization.
        SpacyToken.set_extension("has_space_after", default=True, force=True)

    def negate_sentence(
        self,
        sentence: str,
        prefer_contractions: bool = True
    ) -> str:
        if not sentence:
            return ""
        doc = self._parse(sentence)
        root = self._get_entry_point(doc)

        if not root or not self._is_sentence_supported(doc):
            self._handle_unsupported()
            if not root:  # Don't even bother trying :)
                return sentence

        # Any negations we can remove? (e.g.: "I don't know.", "They won't
        # complain.", "He has not done it.", etc.).
        negation = self._get_negated_child(root)
        # Handle sentences such as "Not to mention that they weren't there."
        # We want to prevent taking the first "Not" as the negation, since this
        # complicates things.
        first_aux_or_verb = self._get_first_aux_or_verb(doc)
        while (negation and first_aux_or_verb
                   and negation.i < first_aux_or_verb.i):
            # Search for another negation, if any.
            negation = self._get_negated_child(root, min_index=negation.i+1)
        if negation:
            aux_child = self._get_aux_child(root)
            # General verbs -> Remove negation and conjugate verb.
            # If there is an AUX child, we need to "unnegate" the AUX instead.
            if (not self._is_aux(root) and root.tag_ != "VBN"
                    and not aux_child and not self._is_verb_to_do(root)
                    and not self._is_verb_to_be(root)):
                remove = [root.i, negation.i]
                add = {root.i: Token(
                    text=self.conjugate_verb(root.text, root.tag_),
                    has_space_after=negation._.has_space_after
                )}
            # Special case AUX "won't" -> Remove negation and replace
            # "wo" -> "will".
            elif root.text.lower() == "wo":
                remove = [root.i, negation.i]
                add = {root.i: Token(
                    text=" will",
                    has_space_after=negation._.has_space_after
                )}
            elif aux_child and aux_child.text.lower() == "wo":
                remove = [aux_child.i, negation.i]
                add = {aux_child.i: Token(
                    text=" will",
                    has_space_after=negation._.has_space_after
                )}
            # Special case AUX "can't" -> Remove negation and replace
            # "ca" -> "can".
            elif root.text.lower() == "ca":
                remove = [root.i, negation.i]
                add = {root.i: Token(
                    text=" can",
                    has_space_after=negation._.has_space_after
                )}
            elif aux_child and aux_child.text.lower() == "ca":
                remove = [aux_child.i, negation.i]
                add = {aux_child.i: Token(
                    text=" can",
                    has_space_after=negation._.has_space_after
                )}
            # Any other AUX or verb in past participle -> Remove negation.
            else:
                remove = [root.i, negation.i]
                # Correctly handle space in e.g., "He hasn't been doing great."
                if negation.i < root.i and negation.i > 0:
                    doc[negation.i-1]._.has_space_after = negation._.has_space_after
                # Correctly handle space in e.g., "I'm not doing great." vs.
                # "I am not doing great."
                space_before = " " * int(root.i > 0
                                        and doc[root.i-1]._.has_space_after)
                # Negation can come before ("She will not ever go.") or after
                # the root ("She will not."). Space after is different in each
                # case.
                space_after = (negation._.has_space_after if negation.i > root.i
                               else root._.has_space_after)
                add = {root.i: Token(
                    text=f"{space_before}{root.text}",
                    has_space_after=space_after
                )}
                # Special case "do" -> Also remove "do" and conjugate verb.
                # E.g.: "He doesn't like it." -> "He likes it".
                if aux_child and self._is_verb_to_do(aux_child):
                    remove.append(aux_child.i)
                    add[root.i] = Token(
                        text=f"{space_before}"
                             f"{self.conjugate_verb(root.text.lower(), aux_child.tag_)}",
                        has_space_after=root._.has_space_after
                    )
            return self._compile_sentence(
                doc,
                remove_tokens=remove,
                add_tokens=add
            )

        # AUX as ROOT (e.g.: "I'm excited.") or ROOT children e.g.,
        # "I do think...".
        if (self._is_aux(root) or self._is_verb_to_be(root)
                or any(self._is_aux(child) for child in root.children)):
            # If the AUX has AUX children, negate them instead. E.g.: "I will
            # be there." or "They have been moody lately."
            aux_child = self._get_aux_child(root)
            return self._negate_aux_in_doc(
                aux=root if not aux_child else aux_child,
                doc=doc,
                prefer_contractions=prefer_contractions
            )

        # General verb non-negated.
        negated_aux = self.negate_aux(self.conjugate_verb('do', root.tag_),
                                      prefer_contractions)
        return self._compile_sentence(
            doc,
            remove_tokens=[root.i],
            add_tokens={root.i: Token(
                text=f"{negated_aux if negated_aux is not None else ''} "
                     f"{self.get_base_verb(root.text.lower())}",
                has_space_after=root._.has_space_after
            )}
        )

    def negate_verb(
        self,
        verb: str,
        tag: str = None,
        prefer_contractions: bool = True
    ) -> str:
        if not verb:
            return ""
        negated_aux = self.negate_aux(verb, prefer_contractions)
        if negated_aux:
            return negated_aux
        if tag is None:  # infer tag
            tag = self._parse(verb)[0].tag_
        return (
            f"{self.negate_aux(self.conjugate_verb('do', tag), prefer_contractions)} "
            f"{self.get_base_verb(verb.lower())}"
        )

    def conjugate_verb(self, verb: str, tag: str) -> str:
        conjugated_verb: Tuple[str] = getInflection(verb, tag)
        return conjugated_verb[0] if conjugated_verb else verb

    def get_base_verb(self, verb: str) -> str:
        base_verb: Tuple[str] = getLemma(verb, upos="VERB")
        return base_verb[0] if base_verb else verb

    def negate_aux(
        self,
        auxiliary_verb: str,
        prefer_contractions: bool = True
    ) -> Optional[str]:
        negated_aux = self._aux_negations[prefer_contractions].get(
            auxiliary_verb
        )
        if negated_aux is None:
            self._handle_unsupported()
        return negated_aux

    def _negate_aux_in_doc(
        self,
        aux: Union[Token, SpacyToken],
        doc: SpacyDoc,
        prefer_contractions: bool = True
    ) -> str:
        negation = self._get_negated_child(aux)
        # If AUX negated -> Remove negation.
        if negation:
            # Special case AUX "won't" -> Remove negation and replace
            # "wo" -> "will".
            if aux.text.lower() == "wo":
                replace_aux = Token(
                    text=" will",
                    has_space_after=negation._.has_space_after
                )
            # Special case AUX "can't" -> Remove negation and replace
            # "ca" -> "can".
            elif aux.text.lower() == "ca":
                replace_aux = Token(
                    text=" can",
                    has_space_after=negation._.has_space_after
                )
            # Nothing to do.
            else:
                replace_aux = Token(
                    text=aux.text,
                    has_space_after=negation._.has_space_after
                )
            return self._compile_sentence(
                doc,
                remove_tokens=[aux.i, negation.i],
                add_tokens={
                    aux.i: replace_aux
                }
            )

        # If AUX not negated -> Replace AUX with its negated.
        # Disambiguate "to be" and "to have" when AUX is "'s".
        aux_text = aux.text
        if aux.text.lower() == "'s":
            parent = self._get_parent(aux, doc)
            if parent and (parent.tag_ == "VBN"
                    or any(child.tag_ == "VBN" for child in parent.children)):
                # "'s" is "to have"
                aux_text = f"{aux.text}_"
        remove = [aux.i]
        add = {
            aux.i: Token(
                text=self.negate_aux(aux_text, prefer_contractions),
                has_space_after=aux._.has_space_after
            )
        }
        # Handle e.g., "should've" -> "shouldn't have"
        if aux.i+1 < len(doc) and doc[aux.i+1].text.lower() == "'ve":
            remove.append(aux.i+1)
            add[aux.i+1] = Token(
                text=" have",
                has_space_after=doc[aux.i+1]._.has_space_after
            )
        return self._compile_sentence(
            doc,
            remove_tokens=remove,
            add_tokens=add
        )

    def _parse(self, string_: str) -> SpacyDoc:
        # Remove extra whitespaces and other non-printable chars.
        string_ = self._remove_extra_whitespaces(string_)
        # Tokenize.
        doc = self.spacy_model(string_)
        i = 0  # Used to determine whitespaces.
        for tk in doc:
            has_space_after: bool = (
                i+len(tk) < len(string_) and (string_[i+len(tk)] == " ")
            )
            tk._.has_space_after = has_space_after
            i += len(tk) + int(has_space_after)
        return doc

    def _get_entry_point(self, doc: SpacyDoc) -> Optional[SpacyToken]:
        root = self._get_root(doc)
        if root is None:  # nothing we can do
            return None
        # If the root token is not an AUX or a VERB, look for an AUX or VERB in
        # its children.
        if not (self._is_aux(root) or self._is_verb(root)):
            entry_point = None
            if root.children:
                entry_point = [tk for tk in root.children
                               if self._is_aux(tk) or self._is_verb(tk)]
            # No AUX or VERB found in the root children -> Take the first AUX
            # or VERB in the sentence, if any.
            if not entry_point:
                entry_point = [tk for tk in doc
                               if self._is_aux(tk) or self._is_verb(tk)]
            return entry_point[0] if entry_point else None
        return root

    def _get_root(self, doc: SpacyDoc) -> Optional[SpacyToken]:
        root = [tk for tk in doc if tk.dep_ == "ROOT"]
        return root[0] if root else None

    def _get_negated_child(
        self,
        token: SpacyToken,
        min_index: int = 0
    ) -> Optional[SpacyToken]:
        if not token:
            return None
        min_index = max(0, min_index)  # prevent negative values
        child = [child for child in token.children if child.dep == neg
                                                      and child.i >= min_index]
        return child[0] if child else None

    def _get_aux_child(
        self,
        token: SpacyToken,
        min_index: int = 0
    ) -> Optional[SpacyToken]:
        if not token:
            return None
        min_index = max(0, min_index)  # prevent negative values
        child = [child for child in token.children if self._is_aux(child)
                                                      and child.i >= min_index]
        return child[0] if child else None

    def _get_first_aux_or_verb(
        self,
        doc: SpacyDoc
    ) -> Optional[SpacyToken]:
        aux = [tk for tk in doc if self._is_aux(tk) or self._is_verb(tk)]
        return aux[0] if aux else None

    def _get_parent(
        self,
        token: SpacyToken,
        doc: SpacyDoc
    ) -> Optional[SpacyToken]:
        if not token:
            return None
        parent = [
            potential_parent
            for potential_parent in doc
            if token in potential_parent.children
        ]
        return parent[0] if parent else None

    def _is_aux(self, token: SpacyToken) -> bool:
        if not token:
            return False
        return token.pos == AUX

    def _is_verb(self, token: SpacyToken) -> bool:
        if not token:
            return False
        return token.pos == VERB

    def _is_verb_to_do(self, verb: SpacyToken) -> bool:
        if not verb:
            return False
        return getLemma(verb.text.lower(), "VERB")[0] == "do"

    def _is_verb_to_be(self, verb: SpacyToken) -> bool:
        if not verb:
            return False
        return getLemma(verb.text.lower(), "VERB")[0] == "be"

    def _is_sentence_supported(self, doc: SpacyDoc) -> bool:
        return any(self._is_aux(tk) or self._is_verb(tk) for tk in doc)

    def _compile_sentence(
        self,
        doc: SpacyDoc,
        remove_tokens: List[int] = None,  # indexes to remove
        add_tokens: Dict[int, Token] = None  # idx, string to add
    ) -> str:
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
            tokens.insert(i+count, tk)
        return self._capitalize_first_letter(
            self._remove_extra_whitespaces(
                "".join([f"{tk.text}{' '*int(tk.has_space_after)}"
                         for tk in tokens])
                )
            )

    def _capitalize_first_letter(self, string_: str) -> str:
        if not string_:
            return ""
        return f"{string_[0].upper()}{string_[1:]}"

    def _remove_extra_whitespaces(self, string_: str) -> str:
        if not string_:
            return ""
        return " ".join(string_.split())

    def _initialize_spacy_model(
        self,
        use_transformers: bool,
        **kwargs
    ) -> spacy.language.Language:
        """Initialize the spaCy model to be used by the Negator.

        Heavily inspired by `https://github.com/BramVanroy/spacy_download`__.

        Args:
            use_transformers (:obj:`Optional[bool]`, defaults to :obj:`False`):
                Whether to use a Transformer model for POS tagging and
                dependency parsing.

                .. note::

                   When set to :obj:`True` the model <en_core_web_trf
                   `https://spacy.io/models/en#en_core_web_trf`>__ is used.
            **kwargs:
                Additional arguments passed to :func:`spacy.load`.

        Returns:
            :obj:`spacy.language.Language`: The loaded spaCy model, ready to
            use.
        """
        # See https://stackoverflow.com/a/25061573/14683209
        # We don't want the messages coming from pip to pollute the stdout.
        @contextmanager
        def suppress_stdout():
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    yield
                finally:
                    sys.stdout = old_stdout

        model_name = "en_core_web_trf" if use_transformers else "en_core_web_md"
        try:  # Model installed?
            model_module = importlib.import_module(model_name)
        except ModuleNotFoundError:  # Download and install model.
            self.logger.info("Downloading model. This only needs to happen "
                             "once. Please, be patient...")
            with suppress_stdout():
                spacy.cli.download(model_name, False, False, "-q")
            model_module = importlib.import_module(model_name)
        return model_module.load(**kwargs)

    def _handle_unsupported(self):
        """Handle behavior upon unsupported sentences.

        Raises:
            :obj:`RuntimeError`: If :arg:`fail_on_unsupported` is set to
            :obj:`True`.
        """
        if self.fail_on_unsupported:
            raise RuntimeError("Sentence not supported.")
        else:
            self.logger.warning("Sentence not supported. Output might be "
                                "arbitrary.")

    def _initialize_aux_negations(self):
        self._aux_negations = {
            True: {
                "'m": "'m not",
                "am": "am not",
                "'re": " aren't",
                "are": "aren't",
                "'s": " isn't",
                "is": "isn't",
                "was": "wasn't",
                "were": "weren't",
                "do": "don't",
                "does": "doesn't",
                "did": "didn't",
                "'ve": " haven't",
                "have": "haven't",
                "'s_": " hasn't",  # underscore to distinguish from verb to be.
                "has": "hasn't",
                "'d": " hadn't",
                "had": "hadn't",
                "can": "can't",
                "could": "couldn't",
                "must": "mustn't",
                "might": "mightn't",
                "may": "may not",
                "should": "shouldn't",
                "ought": "oughtn't",
                "'ll": " won't",
                "will": "won't",
                "would": "wouldn't"
            },
            False: {
                "'m": " am not",
                "am": "am not",
                "'re": " are not",
                "are": "are not",
                "'s": " is not",
                "is": "is not",
                "was": "was not",
                "were": "were not",
                "do": "do not",
                "does": "does not",
                "did": "did not",
                "'ve": " have not",
                "have": "have not",
                "'s_": " has not",  # underscore to distinguish from verb to be.
                "has": "has not",
                "'d": " had not",
                "had": "had not",
                "can": "cannot",
                "could": "could not",
                "must": "must not",
                "might": "might not",
                "may": "may not",
                "should": "should not",
                "ought": "ought not",
                "'ll": " will not",
                "will": "will not",
                "would": "would not"
            }
        }
