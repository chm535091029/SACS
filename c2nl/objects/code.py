import numpy as np
from c2nl.inputters.vocabulary import Vocabulary, BOS_WORD, EOS_WORD


class Code(object):
    """
    Code containing annotated text, original text, selection label and
    all the extractive spans that can be an answer for the associated question.
    """

    def __init__(self, _id=None):
        self._id = _id
        self._language = None
        self._text = None
        self._tokens = []
        self._src_text = None
        self._src_tokens = []
        self._type = []
        self._mask = []
        self._struc = None
        self._level = None
        self._map = None
        self._edge = None
        self.src_vocab = None  # required for Copy Attention

    @property
    def edge(self) -> dict:
        return self._edge

    @edge.setter
    def edge(self, param: dict) -> None:
        self._edge = param

    @property
    def map(self) -> dict:
        return self._map

    @map.setter
    def map(self, param: dict) -> None:
        self._map = param

    @property
    def id(self) -> str:
        return self._id

    @property
    def language(self) -> str:
        return self._language

    @language.setter
    def language(self, param: str) -> None:
        self._language = param

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, param: str) -> None:
        self._text = param

    @property
    def src_text(self) -> str:
        return self._src_text

    @src_text.setter
    def src_text(self, param: str) -> None:
        self._src_text = param

    @property
    def type(self) -> list:
        return self._type

    @type.setter
    def type(self, param: list) -> None:
        assert isinstance(param, list)
        self._type = param

    @property
    def mask(self) -> list:
        return self._mask

    @mask.setter
    def mask(self, param: list) -> None:
        assert isinstance(param, list)
        self._mask = param

    @property
    def tokens(self) -> list:
        return self._tokens

    @tokens.setter
    def tokens(self, param: list) -> None:
        assert isinstance(param, list)
        self._tokens = param
        self.form_src_vocab()

    @property
    def src_tokens(self) -> list:
        return self._src_tokens

    @src_tokens.setter
    def src_tokens(self, param: list) -> None:
        assert isinstance(param, list)
        self._src_tokens = param
        self.form_src_vocab()

    @property
    def level(self) -> np.ndarray:
        return self._level

    @level.setter
    def level(self, param: np.ndarray) -> None:
        assert isinstance(param, np.ndarray)
        self._level = param

    @property
    def struc(self) -> np.ndarray:
        return self._struc

    @struc.setter
    def struc(self, param: np.ndarray) -> None:
        assert isinstance(param, np.ndarray)
        self._struc = param

    def form_src_vocab(self) -> None:
        self.src_vocab = Vocabulary()
        assert self.src_vocab.remove(BOS_WORD)
        assert self.src_vocab.remove(EOS_WORD)
        self.src_vocab.add_tokens(self.tokens)

    def form_src_vocab1(self) -> None:
        self.src_vocab1 = Vocabulary()
        assert self.src_vocab1.remove(BOS_WORD)
        assert self.src_vocab1.remove(EOS_WORD)
        self.src_vocab1.add_tokens(self.src_tokens)

    def vectorize(self, word_dict, _type='word'):
        if _type == 'word':
            return [word_dict[w] for w in self.tokens] ,[word_dict[w] for w in self.src_tokens]
        elif _type == 'char':
            return [word_dict.word_to_char_ids(w).tolist() for w in self.tokens]
        else:
            assert False
