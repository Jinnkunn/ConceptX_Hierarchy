from hierarchy import HierarchicalModel, SimilarWords

if __name__ == '__main__':
    model = HierarchicalModel('files/output2.txt', 'bert-base-cased')
    similar_words = model.get_most_similar('He', 0, 0, n=500)[0]
    similar_words.save_to_sv("./files/similar_words_subset.txt")

    subset_model = HierarchicalModel("./files/similar_words_subset.txt", "bert-base-cased")
    subset_model.get_dendgram()


    # sentence = "I"
    # similar_words_new_input = model.get_most_simiar_word_for_new_input(sentence)
    # for w in similar_words_new_input:
    #     similar_words_for_token = []
    #     for sw in w['similar_words']:
    #         similar_words_for_token.append(sw['word'])
    #     print({
    #         'token': w['token'],
    #         'similar_words': similar_words_for_token
    #     })
