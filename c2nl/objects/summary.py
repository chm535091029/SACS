from c2nl.inputters.vocabulary import EOS_WORD, BOS_WORD


class Summary(object):
    """
    Summary containing annotated text, original text, a list of
    candidate documents, answers and well formed answers.
    """

    def __init__(self, _id=None):
        self._id = _id
        self._text = None
        self._init_text = None
        self._tokens = []
        self._init_tokens = []
        self._type = None  # summary, comment etc

    @property
    def id(self) -> str:
        return self._id

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, param: str) -> None:
        self._text = param

    @property
    def init_text(self) -> str:
        return self._init_text

    @init_text.setter
    def init_text(self, param: str) -> None:
        self._init_text = param

    @property
    def init_tokens(self) -> list:
        return self._init_tokens

    @init_tokens.setter
    def init_tokens(self, param: list) -> None:
        assert isinstance(param, list)
        self._init_tokens = param


    @property
    def tokens(self) -> list:
        return self._tokens

    @tokens.setter
    def tokens(self, param: list) -> None:
        assert isinstance(param, list)
        self._tokens = param

    def append_token(self, tok=EOS_WORD):
        assert isinstance(tok, str)
        self._tokens.append(tok)

    def prepend_token(self, tok=BOS_WORD):
        assert isinstance(tok, str)
        self._tokens.insert(0, tok)

    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, param: str) -> None:
        assert isinstance(param, str)
        self._type = param

    def vectorize(self, word_dict, _type='word'):
        if _type == 'word':
            return [word_dict[w] for w in self.tokens],[word_dict[w] for w in self.init_tokens]
        elif _type == 'char':
            return [word_dict.word_to_char_ids(w).tolist() for w in self.tokens]
        else:
            assert False
