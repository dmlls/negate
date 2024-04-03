"""Languages supported by Negate."""

from pathlib import Path
from typing import List


class Language:
    """Currently available languages for negation.

    The values correspond to the ISO 639-1 language codes. See
    `https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes`__.
    """

    _languages: List[str] = [
        f.name for f in Path(__file__).parent.iterdir() if f.is_dir()
        if not f.name.startswith("__")
    ]

    @classmethod
    def is_supported(cls, lang: str) -> bool:
        """Determine whether a string represents a supported language.

        Args:
            lang (:obj:`str`):
                The language to check support for.

        Returns:
            :obj:`bool`: Whether the string represents a supported language or
            not.
        """
        return lang in cls._languages

    @classmethod
    def get_supported(cls) -> List[str]:
        """Get supported languages for negation.

        Returns:
            :obj:`List[SupportedLanguage]`: All the currently supported
            languages.
        """
        return cls._languages
