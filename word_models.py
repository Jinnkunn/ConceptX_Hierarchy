class SimilarTokenGroup:
    '''
    This class is used to store the all the similar words for one given token.
    '''
    def __init__(self, token, similar_words):
        self.token = token  # instance of Token
        self.similar_words = similar_words  # all the similar words for this word

    def save_to_sv(self, file_path="similar_words_subset.txt"):
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
        return "{}:{}:{}".format(self.word, self.line, self.position)
