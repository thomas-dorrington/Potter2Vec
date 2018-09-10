import regex


def version_to_class(v):
    """
    Map from pre-processor class version to the corresponding class.
    Allows us to load saved pre-processor classes with a uniform API.
    Could probably use a Factory class here, but would be a bit a bit over-kill.
    """

    return {
        'V1': PreprocessorV1
    }[v]


class PreprocessorV1(object):
    """
    Token pre-processor class that cleans lines of word tokens before yielding for training.
    Does nothing but lower-case tokens, and remove certain character class of words.

    This is the most basic pre-processor, but more complicated versions could be implemented. Examples:
      - replace digits with a D symbol
      - collapse multiple digits to a single D
      - expand digits to their written out form, '1' -> 'one'
      - handle out-of-vocabulary words by replacing with a "OOV" token

    """

    def __init__(self, clean_regex=r'[^\p{Alpha} ]'):
        """
        `clean_regex` specifies what characters to remove during pre-processing.
        The default removes anything that is not an alphabetical character or a space.
        """

        self.clean_regex = regex.compile(clean_regex, regex.U)

    @staticmethod
    def from_json(loaded_preprocessor):

        return PreprocessorV1(
            clean_regex=loaded_preprocessor['clean_regex']
        )

    def to_json(self):

        return {
            'version': 'V1',
            'clean_regex': self.clean_regex.pattern
        }

    def preprocess(self, tokens):

        return_tokens = []

        for tok in tokens:

            tok = tok.lower()
            tok = self.clean_regex.sub('', tok)
            tok = tok.strip()

            if tok:
                # If not empty string
                return_tokens.append(tok)

        return return_tokens
