from hierarchy import HierarchicalModel

if __name__ == '__main__':
    model = HierarchicalModel('files/output12.txt', 'bert-base-cased')

    # similar_words = model.get_most_similar('He', 0, 0, n=30)
    # print(similar_words)

    sentence = "I"
    similar_words_new_input = model.get_most_simiar_word_for_new_input(sentence)
    for w in similar_words_new_input:
        similar_words_for_token = []
        for sw in w['similar_words']:
            similar_words_for_token.append(sw['word'])
        print({
            'token': w['token'],
            'similar_words': similar_words_for_token
        })
