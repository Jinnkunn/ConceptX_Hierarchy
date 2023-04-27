# use word to vec to find the most similar words

import gensim
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

from embedding import Embedding


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
        # find all the word has the following format: word:d+:d+
        # where d is a digit. The first d is the line number and the second d is the position
        if line == -1 and position == -1:
            words_selected = [w for w in self.words if w.startswith(word + ":")]
        else:
            specific_word = word + ":" + str(line) + ":" + str(position)
            words_selected = [specific_word]

        similar_words = []

        for w in words_selected:
            similar_words_raw = self.model.most_similar(w, topn=n)
            similar_words_raw = [SimilarWord(
                word=w[0].split(':')[0],
                line=w[0].split(':')[1],
                position=w[0].split(':')[2],
                distance=w[1],
                # raw representation of the word
                values=self.model[w[0]]
            ) for w in similar_words_raw]

            similar_words.append(SimilarWordsGroup(
                word=word,
                line=line,
                position=position,
                similar_words=similar_words_raw
            ))

        return similar_words

    def find_distance(self, word1, word2):
        return self.model.distance(word1, word2)

    def get_embedding(self, word):
        return self.model[word]

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
                one_similar_word = one_similar_word_tuple[0].split(':')
                similar_words_for_one_token.append(SimilarWord(
                    word=one_similar_word[0].replace("##", ""),
                    line=one_similar_word[1],
                    position=one_similar_word[2],
                    distance=one_similar_word_tuple[1],
                    values=self.model[one_similar_word_tuple[0]]
                ))

            result.append({
                'token': embedding[i]['token'].replace(" ", ""),
                'similar_words': similar_words_for_one_token
            })
        return result


class SimilarWordsGroup:
    def __init__(self, word, line, position, similar_words):
        self.word = word
        self.line = line
        self.position = position
        self.similar_words = similar_words # all the similar words for this word

    def save_to_sv(self, file_path="similar_words_subset.txt"):
        print("[HierarchicalModel] writing similar words (subset of the original dataset) to file")
        with open(file_path, 'w') as f:
            f.write("{} {}\n".format(len(self.similar_words), len(self.similar_words[0].values)))
            for w in self.similar_words:
                f.write("{}:{}:{} {}\n".format(w.word, w.line, w.position, " ".join([str(v) for v in w.values])))


class SimilarWord:
    def __init__(self, word, line, position, distance, values):
        self.word = word
        self.line = line
        self.position = position
        self.distance = distance
        self.values = values