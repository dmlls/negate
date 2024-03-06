from abc import ABC, abstractmethod

from negate.tokens import Token
import spacy
from spacy.symbols import AUX, NOUN, PRON, VERB, neg
from spacy.tokens import Doc as SpacyDoc
from spacy.tokens import Token as SpacyToken
from typing import Dict, List, Optional
import logging

class Negator_ABC(ABC):

    def __init__(
            self,
            use_transformers: Optional[bool] = None,
            use_gpu: Optional[bool] = None,
            fail_on_unsupported: Optional[bool] = None,
            log_level: Optional[int] = None
    ):
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

    @abstractmethod
    def negate_sentence(
            self,
            sentence: str,
            prefer_contractions: Optional[bool] = None
    ) -> str:
        pass

    @abstractmethod
    def _initialize_spacy_model(
            self,
            use_transformers: bool,
            **kwargs
    ) -> spacy.language.Language:
        pass

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
                    i + len(tk) < len(string_) and (string_[i + len(tk)] == " ")
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

    @staticmethod
    def _get_root(doc: SpacyDoc) -> Optional[SpacyToken]:
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

    @staticmethod
    def _get_first_negation_particle(
            doc: SpacyDoc
    ) -> Optional[SpacyToken]:
        """Get the first negation particle in a document.

        Args:
            doc (:obj:`SpacyDoc`):
                The spaCy document containing the token.
        Returns:
            :obj:`Optional[SpacyToken]`: The first negation particle in the
            sentence, or :obj:`None` if no such particle exists.
        """
        negation = [tk for tk in doc if tk.dep == neg]
        return negation[0] if negation else None

    @staticmethod
    def _get_negated_child(
            token: SpacyToken,
            min_index: int = 0
    ) -> Optional[SpacyToken]:
        """Get the negated child of a token, if any.

        Only the first negated child with an index equal or greater than
        :param:`min_index` is returned.

        Args:
            token (:obj:`SpacyToken`):
                The spaCy token to get the negated child from.
            min_index (:obj:`int`, defaults to ``0``):
                The minimum index (inclusive) the negated child must have in
                order to be returned. Useful to consider children on the left
                or the right of the passed token.

        Returns:
            :obj:`Optional[SpacyToken]`: The negated child of :param:`token`, or
            :obj:`None` if no negated child was found.
        """
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
        """Get the child of a token that is an auxiliary verb, if any.

        Only the first child that is an auxiliary with an index equal or greater
        than :param:`min_index` is returned.

        Args:
            token (:obj:`SpacyToken`):
                The spaCy token to get the auxiliary children from.
            min_index (:obj:`int`, defaults to ``0``):
                The minimum index (inclusive) the auxiliary child must have in
                order to be returned. Useful to consider children on the left
                or the right of the passed token.

        Returns:
            :obj:`Optional[SpacyToken]`: The auxiliary child of :param:`token`,
            or :obj:`None` if no auxiliary child was found.
        """
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
        """Get the first verb in a spaCy document.

        The verb can be an auxiliary or not.

        Args:
            doc (:obj:`SpacyDoc`):
                The spaCy document to get the first verb from.

        Returns:
            :obj:`Optional[SpacyToken]`: The first verb in the document or
            :obj:`None` if no verb was found.
        """
        aux = [tk for tk in doc if self._is_aux(tk) or self._is_verb(tk)]
        return aux[0] if aux else None

    @staticmethod
    def _get_parent(
            token: SpacyToken,
            doc: SpacyDoc
    ) -> Optional[SpacyToken]:
        """Get the parent of a given token, if any.

        Args:
            token (:obj:`SpacyToken`):
                The spaCy token to get the parent of.
            doc (:obj:`SpacyDoc`):
                The spaCy document in which to find for the parent.

        Returns:
            :obj:`Optional[SpacyToken]`: The parent of the token, or :obj:`None`
            if the token has no parent.
        """
        if not token:
            return None
        parent = [
            potential_parent
            for potential_parent in doc
            if token in potential_parent.children
        ]
        return parent[0] if parent else None

    @staticmethod
    def _is_aux(token: SpacyToken) -> bool:
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

    @staticmethod
    def _is_pronoun(token: SpacyToken) -> bool:
        """Determine whether a token is a pronoun.

        Args:
            token (:obj:`SpacyToken`):
                The spaCy token to determine whether it is a pronoun.

        Returns:
            :obj:`bool`: :obj:`True` if the token is a pronoun,
            otherwise :obj:`False`.
        """
        if not token:
            return False
        return token.pos == PRON

    @staticmethod
    def _is_noun(token: SpacyToken) -> bool:
        """Determine whether a token is a noun.

        Args:
            token (:obj:`SpacyToken`):
                The spaCy token to determine whether it is a noun.

        Returns:
            :obj:`bool`: :obj:`True` if the token is a noun,
            otherwise :obj:`False`.
        """
        if not token:
            return False
        return token.pos == NOUN

    @staticmethod
    def _is_verb(token: SpacyToken) -> bool:
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
    def _capitalize_first_letter(string_: str) -> str:
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

    @staticmethod
    def _remove_extra_whitespaces(string_: str) -> str:
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
            tokens.insert(i + count, tk)
        return self._capitalize_first_letter(
            self._remove_extra_whitespaces(
                "".join([f"{tk.text}{' ' * int(tk.has_space_after)}"
                         for tk in tokens])
            )
        )

    def _handle_unsupported(self, fail: Optional[bool] = None):
        """Handle behavior upon unsupported sentences.

        Args:
            fail (:obj:`Optional[bool]`):
                Whether to raise an exception with unsupported sentences or not.
        Raises:
            :obj:`RuntimeError`: If :arg:`fail_on_unsupported` is set to
            :obj:`True`.
        """
        if fail is None:
            fail = self.fail_on_unsupported
        if fail:
            raise RuntimeError("sentence not supported")
        else:
            self.logger.warning("Sentence not supported. Output might be arbitrary.")
