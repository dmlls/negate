from negate.negator_en import Negator_EN
from negate.negator_de import Negator_DE
from typing import Optional


class Negator:

    def __init__(
            self, language: str = "EN",
            use_transformers: Optional[bool] = None,
            use_gpu: Optional[bool] = None,
            fail_on_unsupported: Optional[bool] = None,
            log_level: Optional[int] = None
    ):
        self.language = language
        if language == "EN":
            self.negator = Negator_EN(use_transformers, use_gpu, fail_on_unsupported, log_level)
        elif language == "DE":
            self.negator = Negator_DE(use_transformers, use_gpu, fail_on_unsupported, log_level)
        else:
            raise ValueError("Language not supported, supported languages are EN and DE")

    def negate_sentence(
            self,
            sentence: str,
            *args, **kwargs
    ) -> set[str] | str:
        return self.negator.negate_sentence(sentence, *args, **kwargs)


