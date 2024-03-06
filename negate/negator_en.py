"""Negation tools."""
import importlib
import os
import sys
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import spacy
from lemminflect import getInflection, getLemma
from spacy.tokens import Doc as SpacyDoc
from spacy.tokens import Token as SpacyToken

from negate.negator import Negator_ABC
from negate.tokens import Token
from negate.version import EN_CORE_WEB_MD_VERSION, EN_CORE_WEB_TRF_VERSION


class Negator_EN(Negator_ABC):
    """Negator for the English language."""

    def __init__(self, use_transformers: Optional[bool] = None, use_gpu: Optional[bool] = None,
                 fail_on_unsupported: Optional[bool] = None, log_level: Optional[int] = None):
        """Instanciate a :obj:`Negator`.

        Args:
            use_transformers (:obj:`Optional[bool]`, defaults to :obj:`False`):
                Whether to use a Transformer model for POS tagging and
                dependency parsing.

                .. note::

                   When set to :obj:`True` the model `en_core_web_trf
                   <https://spacy.io/models/en#en_core_web_trf>`__ is used.
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
        super().__init__(use_transformers, use_gpu, fail_on_unsupported, log_level)
        # Initialize AUX negation dictionary.
        self._initialize_aux_negations()

    def negate_sentence(
            self,
            sentence: str,
            prefer_contractions: Optional[bool] = None
    ) -> str:
        """Negate a sentence.

        Affirmative sentences will be turned into negative ones and vice versa.

        .. note::

            Currently, only sentences that contain at least one verb are
            supported. The output of non-supported sentences might be arbitrary.

        Args:
            sentence (:obj:`str`):
                The sentence to negate.
            prefer_contractions (:obj:`Optional[bool]`, defaults to :obj:`True`):
                Whether, in case the negated part of the sentence is an
                auxiliary verb, get its contracted form (e.g., ``"isn't"``,
                ``"haven't"``, ``"wouldn't"``, etc.).

        Returns:
            :obj:`str`: The negated sentence.
        """
        if not sentence:
            return ""
        if prefer_contractions is None:
            prefer_contractions = True

        doc = self._parse(sentence)
        contains_inversion = self._contains_inversion(doc)
        root = self._get_entry_point(doc, contains_inversion)
        if not root or not self._is_sentence_supported(doc):
            self._handle_unsupported()
            if not root:  # Don't even bother trying :)
                return sentence
        # Any negations we can remove? (e.g.: "I don't know.", "They won't
        # complain.", "He has not done it.", etc.).
        negation = self._get_first_negation_particle(doc)
        # Handle sentences such as "Not to mention that they weren't there."
        # We want to prevent taking the first "Not" as the negation, since this
        # complicates things.
        first_aux_or_verb = self._get_first_aux_or_verb(doc)
        while (negation and first_aux_or_verb
               and first_aux_or_verb.tag_ not in ("VB", "VBG")
               and negation.i < first_aux_or_verb.i):
            # Search for another negation, if any.
            negation = self._get_negated_child(root, min_index=negation.i + 1)
        aux_child = self._get_aux_child(root)
        if negation:
            remove, add = self._handle_ca_wo(root, aux_child, negation=negation)
            # General verbs -> Remove negation and conjugate verb.
            # If there is an AUX child, we need to "unnegate" the AUX instead.
            if (not self._is_aux(root) and root.tag_ not in ("VBN", "VBG")
                    and not aux_child and not self._is_verb_to_do(root)
                    and not self._is_verb_to_be(root)):
                remove = [root.i, negation.i]
                add = {root.i: Token(
                    text=self.conjugate_verb(root.text, root.tag_),
                    has_space_after=negation._.has_space_after
                )}
            # Any other AUX or verb in past participle -> Remove negation.
            elif not remove and not add:
                if contains_inversion:  # E.g.: "Does she not know about it?"
                    remove = [negation.i]
                    add = {negation.i: Token(text="")}  # Add whitespace.
                else:
                    remove = [root.i, negation.i]
                    # Correctly handle space in e.g., "He hasn't been doing great."
                    if negation.i < root.i and negation.i > 0:
                        doc[negation.i - 1]._.has_space_after = negation._.has_space_after
                    # Correctly handle space in e.g., "I'm not doing great." vs.
                    # "I am not doing great."
                    space_before = " " * int(root.i > 0
                                             and doc[root.i - 1]._.has_space_after)
                    # Negation can come before ("She will not ever go.") or after
                    # the root ("She will not."). Space after is different in each
                    # case.
                    space_after = (negation._.has_space_after
                                   if negation.i > root.i
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
            try:
                return self._negate_aux_in_doc(
                    aux=root if not aux_child else aux_child,
                    doc=doc,
                    contains_inversion=contains_inversion,
                    prefer_contractions=prefer_contractions,
                    fail_on_unsupported=True
                )
            except RuntimeError:
                # In cases such as "I expect everything to be ready.", continue
                # trying to negate the root verb instead.
                pass

        # General verb non-negated.
        if (any(child.tag_ == "TO" for child in root.children)
                or root.tag_ == "VBG"):
            # E.g.: "A Python module negating sentences." or "A Python module
            # to negate sentences."
            add = f"not {root.text}"
        else:
            negated_aux = self.negate_aux(self.conjugate_verb('do', root.tag_),
                                          prefer_contractions)
            add = f"{negated_aux} {self.get_base_verb(root.text.lower())}"
        return self._compile_sentence(
            doc,
            remove_tokens=[root.i],
            add_tokens={root.i: Token(
                text=add,
                has_space_after=root._.has_space_after
            )}
        )

    @staticmethod
    def conjugate_verb(verb: str, tag: str) -> str:
        """Conjugate a verb to a tense.

        Args:
            verb (:obj:`str`):
                The verb to conjugate.
            tag (:obj:`str`):
                The target tense, specified as a `Penn Treebank POS tag
                <https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html>`__.

        Returns:
            :obj:`str`: The conjugated verb. If the verb could not be
            conjugated, it is returned unchanged.
        """
        conjugated_verb: Tuple[str] = getInflection(verb, tag)
        return conjugated_verb[0] if conjugated_verb else verb

    @staticmethod
    def get_base_verb(verb: str) -> str:
        """Get the base form (infinitive) of a verb.

        Args:
            verb (:obj:`str`):
                The verb to get the base form of.

        Returns:
            :obj:`str`: The base form of the verb. If the base form could not
            be inferred, the verb is returned unchanged.
        """
        base_verb: Tuple[str] = getLemma(verb, upos="VERB")
        return base_verb[0] if base_verb else verb

    def negate_aux(
            self,
            auxiliary_verb: str,
            prefer_contractions: Optional[bool] = None,
            fail_on_unsupported: Optional[bool] = None
    ) -> Optional[str]:
        """Get the negated form of an auxiliary verb.

        . note::

           This method negates unidirectionally from affirmative to negative.
           In other words, :param:`auxiliary_verb` must be a non-negated
           auxiliary, e.g., ``"is"``, ``"can"``, ``"must"``, etc.

        Args:
            auxiliary_verb (:obj:`str`):
                The auxiliary verb to get the negation form of.
            prefer_contractions (:obj:`Optional[bool]`, defaults to :obj:`True`):
                Whether to get the contracted form of the auxiliary (e.g.,
                ``"isn't"``, ``"haven't"``, ``"wouldn't"``, etc.).
            fail_on_unsupported (:obj:`Optional[bool]`, defaults to :attr:`fail_on_unsupported`):
                Whether to fail in case the passed auxiliary is not supported.
                Since the set of auxiliary verbs is finite, this method will
                only fail if the :param:`auxiliary_verb` is not an auxiliary, or
                the negated form of the auxiliary has been passed.

        Returns:
            :obj:`Optional[str]`: The negated form of the auxiliary, or
            :obj:`None` in case :param:`auxiliary_verb` is not supported.
        """
        if prefer_contractions is None:
            prefer_contractions = True
        if fail_on_unsupported is None:
            fail_on_unsupported = self.fail_on_unsupported

        negated_aux = self._aux_negations[prefer_contractions].get(
            auxiliary_verb.lower()
        )
        if negated_aux is None:
            self._handle_unsupported(fail_on_unsupported)
        return negated_aux

    def _negate_aux_in_doc(
            self,
            aux: Union[Token, SpacyToken],
            doc: SpacyDoc,
            contains_inversion: bool,
            prefer_contractions: Optional[bool] = None,
            fail_on_unsupported: Optional[bool] = None
    ) -> str:
        """Negate an auxiliary within a sentence.

        . note::

           This method, differently from :meth:`Negator.negate_aux`, is
           bidirectional. That means that the passed auxiliary can be in its
           affirmative or negative form.

        Args:
            aux (:obj:`Union[Token, SpacyToken]`):
                The auxiliary verb to negate
            doc (:obj:`SpacyDoc`):
                The spaCy doc containing the auxiliary.
            contains_inversion (:obj:`bool`):
                Whether the sentence contains an inversion.
            prefer_contractions (:obj:`Optional[bool]`, defaults to :obj:`True`):
                Whether to get the contracted form of the auxiliary (e.g.,
                ``"isn't"``, ``"haven't"``, ``"wouldn't"``, etc.).
            fail_on_unsupported (:obj:`Optional[bool]`, defaults to :attr:`fail_on_unsupported`):
                Whether to fail in case the passed auxiliary is not supported.

        Returns:
            :obj:`str`: The resulting sentence, with one of its auxiliary verbs
            (if any) negated.
        """
        if prefer_contractions is None:
            prefer_contractions = True
        if fail_on_unsupported is None:
            fail_on_unsupported = self.fail_on_unsupported

        negation = self._get_negated_child(aux)
        # If AUX negated -> Remove negation.
        if negation:
            remove, add = self._handle_ca_wo(aux, negation=negation)
            if not remove and not add:
                remove = [aux.i, negation.i]
                add = Token(
                    text=aux.text,
                    has_space_after=negation._.has_space_after
                )
            return self._compile_sentence(
                doc,
                remove_tokens=remove,
                add_tokens={aux.i: add}
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
        remove = []
        # Non-contracted inversion or "am", which does not have a contracted
        # version.
        if (contains_inversion and (aux.text.lower() == "am"
                                    or not prefer_contractions)):
            # Find the closest pronoun to the right of the aux and add the negation
            # particle after it.
            pronoun = None
            for tk in doc[aux.i + 1:]:
                if self._is_pronoun(tk):
                    pronoun = tk
                    break
            if pronoun is None:
                self._handle_unsupported(fail=fail_on_unsupported)
            add = {pronoun.i + 1: Token(text="not")}
        else:  # No inversion or contracted inversion.
            remove.append(aux.i)
            add = {
                aux.i: Token(
                    text=self.negate_aux(
                        aux_text,
                        prefer_contractions,
                        fail_on_unsupported=fail_on_unsupported
                    ),
                    has_space_after=aux._.has_space_after
                )
            }
            # Handle e.g., "should've" -> "shouldn't have"
            if aux.i + 1 < len(doc) and doc[aux.i + 1].text.lower() == "'ve":
                remove.append(aux.i + 1)
                add[aux.i + 1] = Token(
                    text=" have",
                    has_space_after=doc[aux.i + 1]._.has_space_after
                )
        return self._compile_sentence(
            doc,
            remove_tokens=remove,
            add_tokens=add
        )

    @staticmethod
    def _handle_ca_wo(
            *aux_tokens: Optional[SpacyToken],
            negation: SpacyToken
    ) -> Tuple[Optional[List[int]], Optional[Dict[int, Token]]]:
        """Handle special cases ``"won't"`` and ``"can't"``.

        These auxiliary verbs are split into ``"wo"`` (AUX) and ``"n't"`` (neg),
        and ``"ca"`` (AUX) / ``"n't"`` (neg), respectively. If we simply removed
        the negation as with other negated auxiliaries (e.g., ``"cannot"`` →
        ``"can"`` (AUX) / ``"not"`` (neg), we remove ``"not"`` and keep
        ``"can"``), we would end up with ``"wo"`` and ``"ca"``, which are not
        correct words. Therefore, we need to take extra steps to replace these
        words by ``"will"`` and ``"can"``, respectively.

        Args:
            *aux_tokens (:obj:`Optional[SpacyToken]`):
                The tokens representing auxiliary verbs to consider. Of all the
                tokens passed, only the first one that actually corresponds to
                this special case will be processed.

                :obj:`None` values can be passed and will be skipped.
            negation (:obj:`SpacyToken`):
                The negation particle that accompanies the auxiliary verbs in
                :param:`*aux_tokens`.

        Returns:
            :obj:`Tuple[Optional[List[int]], Optional[Dict[int, Token]]]`: A
            tuple containing the following values:

               * A list with the indexes of the tokens to be removed. These will
                 be either ``"wo"`` or ``"ca"``, and the negation particle
                 ``"n't"``.
               * A dictionary containing the tokens to be added. These will
                 be either replacing ``"wo"`` → ``"will"``, or ``"ca"`` →
                 ``"can"``.
            If the auxiliary tokens passed do not correspond with this special
            case, ``(None, None)`` is returned.
        """
        remove = []
        add = {}
        for aux in aux_tokens:
            if not aux:
                continue
            # Case AUX "won't" -> Remove negation and replace
            # "wo" -> "will".
            if aux.text.lower() == "wo":
                remove.append(aux.i)
                add.update({aux.i: Token(
                    text=" will",
                    has_space_after=negation._.has_space_after
                )})
            # Case AUX "can't" -> Remove negation and replace
            # "ca" -> "can".
            elif aux.text.lower() == "ca":
                remove.append(aux.i)
                add.update({aux.i: Token(
                    text=" can",
                    has_space_after=negation._.has_space_after
                )})
            if remove and add:
                remove.append(negation.i)
                return remove, add
        return None, None

    @staticmethod
    def _is_verb_to_do(verb: SpacyToken) -> bool:
        """Determine whether a verb is the verb "to do" (in any tense).

        Args:
            verb (:obj:`SpacyToken`):
                The verb to determine whether it is the verb "to do".

        Returns:
            :obj:`bool`: :obj:`True` if the verb is the verb "to do", otherwise
            :obj:`False`.
        """
        if not verb:
            return False
        return getLemma(verb.text.lower(), "VERB")[0] == "do"

    @staticmethod
    def _is_verb_to_be(verb: SpacyToken) -> bool:
        """Determine whether a verb is the verb "to be" (in any tense).

        Args:
            verb (:obj:`SpacyToken`):
                The verb to determine whether it is the verb "to be".

        Returns:
            :obj:`bool`: :obj:`True` if the verb is the verb "to be", otherwise
            :obj:`False`.
        """
        if not verb:
            return False
        return getLemma(verb.text.lower(), "VERB")[0] == "be"

    def _is_sentence_supported(self, doc: SpacyDoc) -> bool:
        """Determine whether a sentence is supported.

        Currently, only sentences that contain at least a verb are supported.

        .. note::

            The output of non-supported sentences might be arbitrary.

        Args:
            doc (:obj:`SpacyDoc`):
                The spaCy document representing the sentence to check for
                support.

        Returns:
            :obj:`bool`: :obj:`True` if the token is an auxiliary verb,
            otherwise :obj:`False`.
        """
        return any(self._is_aux(tk) or self._is_verb(tk) for tk in doc)

    def _contains_inversion(self, doc: SpacyDoc) -> bool:
        """Determine whether a sentence contains an inversion.

        `What is an inversion?
        <https://dictionary.cambridge.org/es-LA/grammar/british-grammar/inversion>`__.

        Args:
            doc (:obj:`SpacyDoc`):
                The spaCy document in which to look for inversions.

        Returns:
            :obj:`bool`: :obj:`True` if the sentence contains an inversion,
            otherwise :obj:`False`.
        """
        aux = None
        pronoun = None
        for tk in doc:
            if self._is_aux(tk):
                aux = tk
            # Only attend to pronouns that don't refer to a noun (i.e., those
            # which could act as subjects).
            if (self._is_pronoun(tk)
                    and not self._is_noun(self._get_parent(tk, doc))):
                pronoun = tk
            if aux and pronoun:
                break
        else:
            return False
        return aux.i < pronoun.i

    def _initialize_spacy_model(
            self,
            use_transformers: bool,
            **kwargs
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
            use_transformers (:obj:`Optional[bool]`, defaults to :obj:`False`):
                Whether to use a Transformer model for POS tagging and
                dependency parsing.

                .. note::

                   When set to :obj:`True` the model `en_core_web_trf
                   <https://spacy.io/models/en#en_core_web_trf>`__ is used.
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
        if use_transformers:
            try:
                importlib.import_module("spacy_transformers")
            except ModuleNotFoundError:
                self.logger.error("Dependencies for Transformers missing. "
                                  "Install them with:\n\n"
                                  ' pip install "negate[transformers]"\n')
                sys.exit(1)
        model_name = (
            f"en_core_web_trf-{EN_CORE_WEB_TRF_VERSION}" if use_transformers
            else f"en_core_web_md-{EN_CORE_WEB_MD_VERSION}"
        )
        module_name = model_name.split("-")[0]
        try:  # Model installed?
            model_module = importlib.import_module(module_name)
        except ModuleNotFoundError:  # Download and install model.
            self.logger.info("Downloading model. This only needs to happen "
                             "once. Please, be patient...")
            with suppress_stdout():
                spacy.cli.download(model_name, True, False, "-q")
            model_module = importlib.import_module(module_name)
        return model_module.load(**kwargs)

    def _initialize_aux_negations(self) -> None:
        """Define the auxiliary verbs and their negated form.

        The negations are defined as a dictionary with two keys: :obj:`True` for
        the contracted auxiliaries, and :obj:`False` for its non-contracted
        counterparts.

        These keys contain in turn another dictionary whose keys are the
        non-negated forms of the auxiliaries, and its values the negated ones.
        """
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
                "'s_": " hasn't",  # underscore to distinguish from verb to be
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
                "'s_": " has not",  # underscore to distinguish from verb to be
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
