from negate.negator import Negator_ABC
from typing import Dict, Optional, Union
import importlib
import os
import sys
from contextlib import contextmanager

import spacy
from DERBI.derbi import DERBI
from spacy.lang.de import German
from spacy.tokens import Doc as SpacyDoc
from spacy.tokens import Token as SpacyToken

from negate.tokens import Token


class Negator_DE(Negator_ABC):

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
            strategy: tuple = ("kein", "nicht", "phrase"),
    ) -> set[str]:
        results = set()
        if not sentence:
            return set()

        for contraction in self._contraction_table:
            sentence = sentence.replace(contraction, self._contraction_table[contraction])

        doc = self._parse(sentence)
        root = self._get_entry_point(doc, False)

        if not root:
            self._handle_unsupported()
            return set()

        un_negated = self._un_negate_sentence(sentence, doc)
        if un_negated:
            return un_negated

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

        return final_results

    def _un_negate_sentence(self, sentence: str, doc: SpacyDoc) -> set[str]:
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
    ) -> set[str]:
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
    ) -> set[str]:
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

    def _negate_phrases(self, sentence: str) -> set[str]:
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

    def _reverse_negate_phrases(self, sentence: str) -> set[str]:

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
    ) -> set[str]:
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

    def _generate_kein_dict(self, noun_token: Union[Token, SpacyToken]) -> dict | bool:
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
