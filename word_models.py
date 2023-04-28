class SimilarTokenGroup:
    '''
    This class is used to store the all the similar words for one given token.
    '''
    def __init__(self, token, similar_words):
        self.token = token  # instance of Token
        self.similar_words = similar_words  # all the similar words for this word

    def save_to_sv(self, file_path="similar_words_subset.txt"):
        '''
        This method is used to save the similar words to a file. The default file path is similar_words_subset.txt
        :param file_path: the file path to save the similar words
        :return: None
        '''
        print("[HierarchicalModel] writing similar words (subset of the original dataset) to file")
        with open(file_path, 'w') as f:
            f.write("{} {}\n".format(len(self.similar_words), len(self.similar_words[0].values)))
            for w in self.similar_words:
                f.write("{} {}\n".format(w.to_string(), " ".join([str(v) for v in w.values])))


class SimilarToken:
    '''
    This class is used to store the information of one similar word.
    '''
    def __init__(self, token, parent_token, distance, values):
        self.token = token  # instance of Token
        self.parent_token = parent_token
        self.distance = distance
        self.values = values

    def to_string(self):
        '''
        This method is used to convert the similar word to a string representation.
        :return: string representation of the similar word
        '''
        return "{}:{}:{}".format(self.token.word, self.token.line, self.token.position)


class Token:
    '''
    This class is used to store the information of one token.
    The token is defined as a word in a specific line and position.
    '''
    def __init__(self, word, line, position):
        self.word = word
        self.line = line
        self.position = position

    def to_string(self):
        '''
        This method is used to convert the token to a string representation.
        :return: string representation of the token
        '''
        return "{}:{}:{}".format(self.word, self.line, self.position)

    def from_string(token):
        '''
        This method is used to convert a string representation of a token to an instance of Token.
        :param token: string representation of the token
        :return: instance of Token
        '''
        token_split = token.split(":")
        # ideally, the token_split size should be 3
        # if the token_split size is 4, then it means the word contains ":" in it, so we need to join the first 2 elements
        # if the token_split size is neither 3 nor 4, then it means the token is invalid
        if len(token_split) == 3:
            word = token_split[0]
            line = token_split[1]
            position = token_split[2]
        elif len(token_split) == 4:
            word = ":".join(token_split[:-2])
            line = token_split[-2]
            position = token_split[-1]
        else:
            raise Exception("Invalid token format: {}".format(token))

        return Token(word, line, position)
