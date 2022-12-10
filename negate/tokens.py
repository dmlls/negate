"""Token-related utils."""

from dataclasses import dataclass


@dataclass
class Token:
    """A token, i.e., a sequence of characters.

    A :obj:`spacy.tokens.Token` is read-only. See
    `https://spacy.io/api/token`__. Furthermore, it cannot hold an empty string
    as :attr:`text`. This class is just a minimalist equivalent to a Spacy Token
    that is both writable and allows empty strings as text. It only contains the
    attributes relevant for our purposes.

    Attributes:
        text (:obj:`str`):
            The characters that form the token.
        has_space_after (:obj:`str`):
            Whether the token is followed by a whitespace (this whitespace is
            not part of the :attr:`text`). This information is used for
            detokenization.
    """

    text: str
    has_space_after: bool = True
