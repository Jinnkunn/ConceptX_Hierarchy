# use word to vec to find the most similar words

import gensim
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

from support_models.embedding import Embedding
from support_models.words import SimilarToken, Token, SimilarTokenGroup


class HierarchicalModel:
    def __init__(self, model_path, bert_model_name='bert-base-cased'):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)
        print("[HierarchicalModel] word2vec model loaded")
        self.words = list(self.model.key_to_index.keys())
        self.bert_model = Embedding(bert_model_name)
        print("[HierarchicalModel] BERT model loaded")

    def get_dendgram(self):
        X = self.model[self.words]
        Z = linkage(X, 'ward')
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x-axis labels
            leaf_font_size=8.,  # font size for the x-axis labels
        )
        plt.show()

    def get_most_similar(self, word, line=-1, position=-1, n=10):
        if line == -1 and position == -1:
            # find all the word has the following format: word:d+:d+
            # where d is a digit. The first d is the line number and the second d is the position
            words_selected = [Token.from_string(w) for w in self.words if
                              w.startswith(word + ":")]
        else:
            # if the line and position is specified, then we can find the specific word
            specific_word = Token(word, line, position)
            words_selected = [specific_word]

        similar_words = []

        # For each token, they have their own similar words
        for one_token in words_selected:
            similar_words_raw = self.model.most_similar(one_token.to_string(), topn=n)
            similar_words_raw = [SimilarToken(
                # token=Token(word=token_split(w[0])[0], line=token_split(w[0])[1], position=token_split(w[0])[2]),
                token=Token.from_string(w[0]),
                distance=w[1],
                parent_token=one_token,
                # raw representation of the word
                values=self.model[w[0]]
            ) for w in similar_words_raw]

            # add the similar words to the list
            # SimilarWordsGroup object contains the token and its similar words
            similar_words.append(SimilarTokenGroup(
                token=Token(word=word, line=line, position=position),
                similar_words=similar_words_raw
            ))

        return similar_words

    def find_distance(self, word1, word2):
        return self.model.distance(word1, word2)

    def get_embedding(self, token):
        return self.model[token]

    def get_most_simiar_word_for_new_input(self, sentence, n=10):
        '''
        :param sentence: new input sentence
        :param n: number of similar words to return
        :return: a list of similar words for each token in the sentence

        The result has the following format:
        [
            {
                'token': 'He',
                'similar_words': [
                    {
                        'word': 'She',
                        'line': '0',
                        'position': '0',
                        'distance': 0.0
                    },
            }
        ]
        '''

        # embed the new word based on BERT
        embedding = self.bert_model.get_embedding(sentence)
        # find the most similar word in the word2vec model
        result = []
        for i in range(len(embedding)):
            similar_words = self.model.most_similar(positive=[embedding[i]['embedding']], topn=n)
            similar_words_for_one_token = []

            # one_similar_word_tuple has the following format: (word:d+:d+, distance)
            # where d is a digit. The first d is the line number and the second d is the position
            for one_similar_word_tuple in similar_words:
                # one_similar_word has the following format: [word, d+ ,d+]
                similar_words_for_one_token.append(SimilarToken(
                    token=Token.from_string(one_similar_word_tuple[0]),
                    distance=one_similar_word_tuple[1],
                    values=self.model[one_similar_word_tuple[0]]
                ))

            result.append({
                'token': embedding[i]['token'].replace(" ", ""),
                'similar_words': similar_words_for_one_token
            })
        return result
