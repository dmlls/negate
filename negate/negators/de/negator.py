"""German Negation."""

import importlib
import logging
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set, Union

import spacy
from DERBI.derbi import DERBI
from negate.base import BaseNegator
from negate.utils.tokens import Token
from spacy.lang.de import German
from spacy.symbols import AUX, VERB
from spacy.tokens import Doc as SpacyDoc
from spacy.tokens import Token as SpacyToken


class Negator(BaseNegator):
    """Negator for the German language."""

    def __init__(
        self,
        use_transformers: Optional[bool] = None,
        use_gpu: Optional[bool] = None,
        fail_on_unsupported: Optional[bool] = None,
        log_level: Optional[int] = None,
        **kwargs,
    ):
        """Instanciate a German Negator.

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
            format="%(levelname)s: %(message)s",
            level=log_level
        )
        self.logger = logging.getLogger(__class__.__name__)
        self.fail_on_unsupported = fail_on_unsupported
        # Load spaCy model. If not available locally, the model will be first
        # installed.
        if use_transformers and use_gpu:
            spacy.require_gpu()
        else:
            spacy.require_cpu()
        self.spacy_model = self._initialize_spacy_model(use_transformers)
        # Store whether tokens have a whitespace after them. This is used later
        # on for de-tokenization.
        SpacyToken.set_extension("has_space_after", default=True, force=True)

        self.derbi = DERBI(self.spacy_model)
        self._de_initialize_nicht_replace_table()
        self._de_initialize_kein_table()
        self._de_initialize_contraction_table()
        self._de_initialize_reverse_contraction_table()
        self._de_initialize_phrase_table()
        self._de_initialize_reverse_phrase_table()
        self._de_initialize_two_part_phrase_table()
        self._de_initialize_reverse_two_part_phrase_table()

    def negate_sentence(
        self,
        sentence: str,
        **kwargs: Dict[str, Any],
    ) -> List[str]:
        """Negate a sentence.

        Affirmative sentences will be turned into negative ones and vice versa.

        .. note::

            Currently, only sentences that contain at least one verb are
            supported. The output of non-supported sentences might be arbitrary.

        Args:
            sentence (:obj:`str`):
                The sentence to negate.
            strategy (:obj:`Optional[List[str]]`, defaults to ``["kein", "nicht", "phrase"]``):
                The negation strategy to use, i.e., whether to negate by adding
                the "kein" or "nicht" negation particles, or perform a phrase
                negation.

        Returns:
            :obj:`List[str]`: The negated sentence(s).
        """
        results = set()
        if not sentence:
            return []

        strategy = kwargs.get("strategy")
        if strategy is None:
            strategy = ["kein", "nicht", "phrase"]

        for contraction in self._contraction_table:
            sentence = sentence.replace(contraction, self._contraction_table[contraction])

        doc = self._parse(sentence)
        root = self._get_entry_point(doc, False)

        if not root:
            self._handle_unsupported()
            return []

        un_negated = self._un_negate_sentence(sentence, doc)
        if un_negated:
            return list(un_negated)

        # Edge case "weder noch"
        exception_list = [" weder ", "Weder ", " sowohl ", "Sowohl"]
        weder_noch = any(x in sentence for x in exception_list)
        if "kein" in strategy and not weder_noch:
            results.update(self._negate_kein(doc))
        if "nicht" in strategy and not weder_noch:
            results.update(self._negate_nicht(doc, root))
        if "phrase" in strategy:
            results.update(self._negate_phrases(sentence))

        if not results:
            self._handle_unsupported(self.fail_on_unsupported)

        final_results = set()
        for result in results:
            final_result = result
            for contraction in self._reverse_contraction_table:
                final_result = final_result.replace(contraction, self._reverse_contraction_table[contraction])
            final_results.add(final_result)

        return list(final_results)

    def _parse(self, string_: str) -> SpacyDoc:
        """Parse a string.

        This method cleans up the string and tokenizes it. The resulting
        :obj:`SpacyDoc` object, also includes information about whitespaces to
        facilitate de-tokenization later on.

        Args:
            string_ (:obj:`str`):
                The string to parse.

        Returns:
            :obj:`SpacyDoc`: The string tokenized into a spaCy document.
        """
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

    def _get_entry_point(
        self,
        doc: SpacyDoc,
        contains_inversion: bool
    ) -> Optional[SpacyToken]:
        """Choose a suitable verb to attempt negating first, if any.

        Args:
            doc (:obj:`SpacyDoc`):
                The spaCy document in which to find the entry point.
            contains_inversion (:obj:`bool`):
                Whether the sentence contains an inversion or not.

        Returns:
            :obj:`Optional[SpacyToken]`: The chosen entry point (verb), or
            :obj:`None` if the sentence has no root, or contains no verbs.
        """
        if contains_inversion:
            entry_point = [tk for tk in doc
                           if self._is_aux(tk) or self._is_verb(tk)]
            if entry_point:
                return entry_point[0]
        root = self._get_root(doc)
        if root is None:  # nothing we can do
            return None
        # If the root token is not an AUX or a VERB, look for an AUX or
        # VERB in its children.
        if not (self._is_aux(root) or self._is_verb(root)):
            entry_point = None
            if root.children:
                entry_point = [tk for tk in root.children
                                if self._is_aux(tk) or self._is_verb(tk)]
            # No AUX or VERB found in the root children -> Take the first
            # AUX or VERB in the sentence, if any.
            if not entry_point:
                entry_point = [tk for tk in doc
                                if self._is_aux(tk) or self._is_verb(tk)]
            return entry_point[0] if entry_point else None
        return root

    def _get_root(self, doc: SpacyDoc) -> Optional[SpacyToken]:
        """Get the root token in a spaCy document, if any.

        Args:
            doc (:obj:`SpacyDoc`):
                The spaCy document to get the root from.

        Returns:
            :obj:`Optional[SpacyToken]`: The root token, or :obj:`None` if the
            sentence has no root.
        """
        root = [tk for tk in doc if tk.dep_ == "ROOT"]
        return root[0] if root else None

    def _capitalize_first_letter(self, string_: str) -> str:
        """Uppercase the first letter of a string.

        The capitalization of the rest of the string remains unchanged.

        Args:
            string_ (:obj:`str`):
                The string whose first letter to uppercase.

        Returns:
            :obj:`str`: The string with its first letter uppercased.
        """
        if not string_:
            return ""
        return f"{string_[0].upper()}{string_[1:]}"

    def _remove_extra_whitespaces(self, string_: str) -> str:
        """Remove any duplicated whitespaces in a string.

        Args:
            string_ (:obj:`str`):
                The string in which to remove any extra whitespaces.

        Returns:
            :obj:`str`: The string with one whitespace at most between words.
        """
        if not string_:
            return ""
        return " ".join(string_.split())

    def _is_aux(self, token: SpacyToken) -> bool:
        """Determine whether a token is an auxiliary verb.

        Args:
            token (:obj:`SpacyToken`):
                The spaCy token to determine whether it is auxiliary.

        Returns:
            :obj:`bool`: :obj:`True` if the token is an auxiliary verb,
            otherwise :obj:`False`.
        """
        if not token:
            return False
        return token.pos == AUX

    def _is_verb(self, token: SpacyToken) -> bool:
        """Determine whether a token is a non-auxiliary verb.

        .. note::

            If you want to check if a token is either an auxiliary *or* a verb,
            you can use this method in combination with :meth:`Negator._is_aux`.

        Args:
            token (:obj:`SpacyToken`):
                The spaCy token to determine whether it is a non-auxiliary verb.

        Returns:
            :obj:`bool`: :obj:`True` if the token is a non-auxiliary verb,
            otherwise :obj:`False`.
        """
        if not token:
            return False
        return token.pos == VERB

    @staticmethod
    def _is_full_sentence(root: SpacyToken) -> bool:
        """Check if it is a full sentence"""
        subject = [x for x in root.children if x.dep_ == "sb"]
        if not subject:
            return False
        return subject[0].i < root.i

    @staticmethod
    def _find_verb_form(token: SpacyToken) -> str:
        """find the verb form of token"""
        result = ""
        morph_dict = token.morph.to_dict()
        if "VerbForm" in morph_dict:
            result = morph_dict["VerbForm"]
        return result

    @staticmethod
    def _find_definite(token: SpacyToken) -> str:
        """find the definite type of token i.e. definite or indefinite"""
        result = ""
        morph_dict = token.morph.to_dict()
        if "Definite" in morph_dict:
            result = morph_dict["Definite"]
        return result

    @staticmethod
    def _find_number(token: SpacyToken) -> str:
        """find the number type of token i.e. plural or singular"""
        result = ""
        morph_dict = token.morph.to_dict()
        if "Number" in morph_dict:
            result = morph_dict["Number"]
        return result

    @staticmethod
    def _find_case(token: SpacyToken) -> str:
        """find the case type of token e.g. nominative, accusative, etc."""
        result = ""
        morph_dict = token.morph.to_dict()
        if "Case" in morph_dict:
            result = morph_dict["Case"]
        return result

    @staticmethod
    def _find_gender(token: SpacyToken) -> str:
        """find the gender type of token i.e. feminine, masculine, neutral."""
        result = ""
        morph_dict = token.morph.to_dict()
        if "Gender" in morph_dict:
            result = morph_dict["Gender"]
        return result

    @staticmethod
    def _find_last_word(doc: SpacyDoc) -> SpacyToken:
        """find the last word"""
        if doc[-1].pos_ == "PUNCT":
            return doc[-2]
        return doc[-1]

    def _compile_sentence(
        self,
        doc: SpacyDoc,
        remove_tokens: Optional[List[int]] = None,
        add_tokens: Optional[Dict[int, Token]] = None
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
            tokens.insert(i+count, tk)
        return self._capitalize_first_letter(
            self._remove_extra_whitespaces(
                "".join([f"{tk.text}{' '*int(tk.has_space_after)}"
                         for tk in tokens])
                )
            )

    def _un_negate_sentence(self, sentence: str, doc: SpacyDoc) -> Set[str]:
        # un negate phrases
        un_negated_sentences_phrases = self._reverse_negate_phrases(sentence)
        if un_negated_sentences_phrases:
            return un_negated_sentences_phrases

        # replace all "nicht"s with unnegated counterpart
        wip_sentence = sentence
        for key, value in self._nicht_replace_table.items():
            wip_sentence = wip_sentence.replace(key, value)

        # if the unnegated sentence is not equal to the original sentence, negations were present, return unnegated
        # sentence
        if not wip_sentence == sentence:
            return {wip_sentence[:1].capitalize() + wip_sentence[1:]}

        tokens_to_remove = []
        tokens_to_add = {}

        for token in doc:
            if str(token).lower() in ["kein", "keine", "keinen", "keiner", "keinem", "keines"]:
                tokens_to_remove.append(token.i)
                try:
                    parent = next(token.ancestors)
                except StopIteration:
                    parent = None

                number = self._find_number(token)
                if parent:
                    if parent.pos_ == "NOUN":
                        number = self._find_number(parent)
                if number == "Sing":
                    tokens_to_add[token.i] = Token(str(token).replace("k", "").replace("Ke", "E"))

        if tokens_to_remove:
            return {self._compile_sentence(
                doc,
                tokens_to_remove,
                tokens_to_add
            )}

    def _negate_kein(
            self,
            doc: SpacyDoc
    ) -> Set[str]:
        results = set()
        kein_noun_dicts = [self._generate_kein_dict(x) for x in doc if x.pos_ == "NOUN"]
        kein_noun_dicts = [x for x in kein_noun_dicts if x]
        if kein_noun_dicts:
            for noun_dict in kein_noun_dicts:
                results.add(self._negate_kein_sentence(
                    noun_dict["leftmost"],
                    noun_dict["kein_form"],
                    noun_dict["inflected_adj"],
                    doc,
                    remove_article=noun_dict["ind_article"]))

        return results

    def _negate_nicht(
            self,
            doc: SpacyDoc,
            root: Union[Token, SpacyToken]
    ) -> Set[str]:
        results = set()
        svps = any(x.dep_ == "svp" for x in doc)
        adps = any(x.pos_ == "ADP" for x in doc)
        advs = any(x.pos_ == "ADV" for x in doc)
        infs = self._find_verb_form(self._find_last_word(doc)) in ["Inf", "Part"]
        full_sentence = self._is_full_sentence(root=root)

        if svps:
            results.add(self._negate_verb_part(doc=doc))

        if advs:
            results.add(self._negate_adverb(doc=doc))

        if adps:
            results.update(self._negate_adposition(root=root, doc=doc, dont_negate_at_end=any([svps, advs])))

        if infs:
            results.add(self._negate_infinitive(doc=doc))

        if full_sentence and not any([svps, adps, infs]):
            results.add(self._negate_at_end(doc))

        return results

    def _negate_phrases(self, sentence: str) -> Set[str]:
        results = set()

        for phrase in self._phrase_table.keys():
            if phrase in sentence:
                results.add(sentence.replace(phrase, self._phrase_table[phrase]))

        for two_part_phrase in self._two_part_phrase_table.keys():
            if two_part_phrase[0] in sentence and two_part_phrase[1] in sentence:
                result_sentence = sentence
                result_sentence = result_sentence.replace(two_part_phrase[0],
                                                          self._two_part_phrase_table[two_part_phrase][0])
                result_sentence = result_sentence.replace(two_part_phrase[1],
                                                          self._two_part_phrase_table[two_part_phrase][1])
                results.add(result_sentence)

        return results

    def _reverse_negate_phrases(self, sentence: str) -> Set[str]:

        for phrase in self._reverse_phrase_table.keys():
            if phrase in sentence:
                return {sentence.replace(phrase, self._reverse_phrase_table[phrase])}

        for two_part_phrase in self._reverse_two_part_phrase_table.keys():
            if two_part_phrase[0] in sentence and two_part_phrase[1] in sentence:
                result_sentence = sentence
                result_sentence = result_sentence.replace(two_part_phrase[0],
                                                          self._reverse_two_part_phrase_table[two_part_phrase][0])
                result_sentence = result_sentence.replace(two_part_phrase[1],
                                                          self._reverse_two_part_phrase_table[two_part_phrase][1])
                return {result_sentence}

    def _negate_at_end(
            self,
            doc: SpacyDoc
    ) -> str:
        add = {self._find_last_word(doc).i + 1: Token(
            text=" nicht",
            has_space_after=False
        )}

        return self._compile_sentence(
            doc=doc,
            remove_tokens=[],
            add_tokens=add
        )

    def _negate_before_token(
            self,
            token: Union[Token, SpacyToken],
            doc: SpacyDoc
    ) -> str:

        add = {token.i: Token(
            text=" nicht",
            has_space_after=True
        )}

        return self._compile_sentence(
            doc=doc,
            remove_tokens=[],
            add_tokens=add
        )

    def _negate_kein_sentence(
            self,
            kein_before: Union[Token, SpacyToken],
            kein_form: Union[Token, SpacyToken],
            inflected_adj: Dict[int, str],
            doc: SpacyDoc,
            remove_article: bool,  # noqa FBT001
    ) -> str:

        add = {}
        remove_tokens = []
        if remove_article:
            remove_tokens.append(kein_before.i)

        for key, value in inflected_adj.items():
            remove_tokens.append(key)
            # As kein is added, position in sentence needs to be shifted by one
            add[key + 1] = Token(value)

        add[kein_before.i] = kein_form

        return self._compile_sentence(
            doc=doc,
            remove_tokens=remove_tokens,
            add_tokens=add
        )

    def _negate_adposition(
            self,
            root: Union[Token, SpacyToken],
            doc: SpacyDoc,
            dont_negate_at_end: bool,  # noqa FBT001
    ) -> Set[str]:
        adpositions = [x for x in doc if x.pos_ == "ADP"]
        results = set()
        for adposition in adpositions:
            if adposition.i < root.i and not dont_negate_at_end:
                results.add(self._negate_at_end(doc=doc))
            else:
                results.add(self._negate_before_token(adposition, doc))

        return results

    def _negate_adverb(
            self,
            doc: SpacyDoc
    ) -> str:
        adverb = [x for x in doc if x.pos_ == "ADV"][0]
        return self._negate_before_token(adverb, doc)

    def _negate_infinitive(
            self,
            doc: SpacyDoc
    ) -> str:
        last_word = self._find_last_word(doc)
        next_left = next(last_word.lefts)
        if next_left:
            if self._find_verb_form(next_left) in ["Inf", "Part"]:
                return self._negate_before_token(next_left, doc)
        return self._negate_before_token(last_word, doc)

    def _negate_verb_part(
            self,
            doc: SpacyDoc
    ) -> str:
        return self._negate_before_token(self._find_last_word(doc), doc)

    def _generate_kein_dict(self, noun_token: Union[Token, SpacyToken]) -> Union[bool, Dict]:
        children = list(noun_token.children)
        # No children to the right
        children = [child for child in children if child.i < noun_token.i]

        if children:
            leftmost_child = children[0]
        else:
            leftmost_child = noun_token

        if not children or children[0].pos_ == "ADJ":

            # inflect adjectives
            inflected_adj = {}
            for child in children:
                if child.pos_ == "ADJ":
                    # inflection using "magic" string :D
                    noun_number = self._find_number(noun_token)
                    noun_case = self._find_case(noun_token)
                    noun_gender = self._find_gender(noun_token)
                    derbi_string = (f"Case={noun_case if noun_case else 'Acc'}|Declination=Mixed|Degree=Pos|"
                                    f"Gender={noun_gender if noun_gender else 'Fem'}|"
                                    f"Number={noun_number if noun_number else 'Plur'}")
                    inflected_adj[child.i] = self.derbi.inflect(child, derbi_string)

            return {
                "ind_article": False,
                "leftmost": leftmost_child,
                "kein_form": self._de_match_kein_form(noun_token),
                "inflected_adj": inflected_adj
            }
        if children[0].pos_ == "DET" and self._find_definite(leftmost_child) == "Ind":
            return {
                "ind_article": True,
                "leftmost": leftmost_child,
                "kein_form": self._de_match_kein_form(noun_token),
                "inflected_adj": {}
            }
        # if it isn't a noun to be negated with "kein" return False
        return False

    def _initialize_spacy_model(
            self,
            use_transformers: bool,  # noqa FBT001
            **kwargs
    ) -> German:
        """
        Initialize the spaCy model to be used by the Negator.

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
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    yield
                finally:
                    sys.stdout = old_stdout

        model_name = "de_dep_news_trf" if use_transformers else "de_core_news_md"
        try:  # Model installed?
            model_module = importlib.import_module(model_name)
        except ModuleNotFoundError:  # Download and install model.
            self.logger.info("Downloading model. This only needs to happen "
                             "once. Please, be patient...")
            with suppress_stdout():
                spacy.cli.download(model_name, False, False, "-q")
            model_module = importlib.import_module(model_name)
        return model_module.load(**kwargs)

    def _de_initialize_kein_table(self):
        """Define the declination of the German word "kein".

        The declination is defined as a dictionary with keys for the gender and plural.
        The values contain dictionaries with the right declination for each case.
        """
        self._kein_table = {
            "Masc": {
                "Nom": "kein",
                "Acc": "keinen",
                "Dat": "keinem",
                "Gen": "keines"
            },
            "Fem": {
                "Nom": "keine",
                "Acc": "keine",
                "Dat": "keiner",
                "Gen": "keiner"
            },
            "Neut": {
                "Nom": "kein",
                "Acc": "kein",
                "Dat": "keinem",
                "Gen": "keines"
            },
            "Plur": {
                "Nom": "keine",
                "Acc": "keine",
                "Dat": "keinen",
                "Gen": "keiner"
            }
        }

    def _de_match_kein_form(self, object_token: Union[Token, SpacyToken]) -> Token:
        """Find the right Form for the work "kein".

        The morphology information is retrieved from the token and matched against the kein table.
        """
        morph_dict = object_token.morph.to_dict()
        case = morph_dict["Case"]
        number_gender = morph_dict["Number"]
        if number_gender != "Plur":
            number_gender = morph_dict["Gender"]

        return Token(
            text=self._kein_table[number_gender][case],
            has_space_after=True
        )

    def _de_initialize_contraction_table(self):
        """Define a dictionary with German contractions and how they can be expanded.
        """
        self._contraction_table = {
            " am ": " an dem ",
            " ans ": " an das ",
            " beim ": " bei dem ",
            " im ": " in dem ",
            " ins ": " in das ",
            " vom ": " von dem ",
            " zum ": " zu dem ",
            " zur ": " zu der ",
            "Am ": "An dem ",
            "Ans ": "An das ",
            "Beim ": "Bei dem ",
            "Im ": "In dem ",
            "Ins ": "In das ",
            "Vom ": "Von dem ",
            "Zum ": "Zu dem ",
            "Zur ": "Zu der "
        }

    def _de_initialize_reverse_contraction_table(self):
        """Define a dictionary with German contractions. The keys are the long forms,
        while the values are the contracted forms.

        The function _de_initialize_contraction_table is used and the returned dictionary reversed.
        """
        self._reverse_contraction_table = {k: v for (v, k) in self._contraction_table.items()}

    def _de_initialize_phrase_table(self):
        """Define German phrases and their negated form.

        The phrases are defined as a dictionary where the key is the affirmative version
        and the value is the negative version.
        """
        self._phrase_table = {
            "immer noch": "nicht mehr",
            "Immer noch": "Nicht mehr",
            "jemand": "niemand",
            "Jemand": "Niemand",
            "irgendwo": "nirgendwo",
            "Irgendwo": "Nirgendwo",
            "Schon immer": "Noch nie",
            "schon immer": "noch nie",
            "immer": "nie",
            "Immer": "Nie",
            "oft": "selten",
            "Oft": "Selten",
            "etwas": "nichts",
            "Etwas": "Nichts",
            "alles": "nichts",
            "Alles": "Nichts",
        }

    def _de_initialize_reverse_phrase_table(self):
        """Define German phrases and their negated form.

        The phrases are defined as a dictionary where the key is the negative version
        and the value is the affirmative version.

        The function _de_initialize_phrase_table is used and the dictionary is reverted.
        """
        self._reverse_phrase_table = {k: v for (v, k) in self._phrase_table.items()}

    def _de_initialize_two_part_phrase_table(self):
        """Define German two-part phrases and their negated form.

        The negations are defined as a dictionary where the key is the negative version
        and the value is the affirmative version.
        """
        self._two_part_phrase_table = {
            ("weder", "noch"): ("sowohl", "als auch"),
            ("Weder", "noch"): ("Sowohl", "als auch"),
        }

    def _de_initialize_reverse_two_part_phrase_table(self):
        """Define German two-part phrases and their negated form.

        The negations are defined as a dictionary where the key is the affirmative version
        and the value is the negative version.

        The function _de_initialize_two_part_phrase_table is used and the returned dictionary reversed.
        """
        self._reverse_two_part_phrase_table = {k: v for (v, k) in self._two_part_phrase_table.items()}

    def _de_initialize_nicht_replace_table(self):
        """Define how the word "nicht" should be removed.
        """
        self._nicht_replace_table = {
            " nicht ": " ",
            " nicht.": ".",
            "Nicht ": "",
        }
