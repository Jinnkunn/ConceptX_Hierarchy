# ConceptX_Hierarchy

The purpose of this code is to utilize the output files generated by the [ConceptX_Converter](https://github.com/Jinnkunn/ConceptX_Converter) to construct a hierarchical model with Gensim. This enables the matching of new input sentences to the high-dimensional space produced by [ConceptX](https://github.com/hsajjad/ConceptX), allowing users to gain a deeper comprehension of its output.

## Usage
The main.py provides an example of how to use the code. When you run the code, we are assuming you already using the ConceptX to generate the activations, and using the ConceptX_Converter to convert the activations to the input format of word2Vec.

### Model Initialization
The following code will be used to initialize the model
```{python}
model = HierarchicalModel('files/output12.txt', 'bert-base-cased')
```
where the first parameter is the path to the w2c format file generated by the ConceptX_Converter, and the second parameter is the name of the BERT model you want to use. The default value is 'bert-base-cased'. The model will take a few minutes to load, depending on the size of the output file.

### Get Most Similar Words for Existing Words
The following code will be used to get the top n most similar words to the input word
```{python}
model.model.get_most_simiar_words(word, line, position, n)
```
The code takes in two parameters. The first parameter represents the input word to be analyzed, while the second parameter specifies the number of most similar words (from the original dataset used to train ConceptX) to be retrieved. By default, the code retrieves the top 10 most similar words.
<br><br>
When the line and position number are not specified, the code retrieves the most similar words for all occurrences of the input word in the file that was used to create the HierarchicalModel. For example, if the input word is 'He', the code will find all the similar words for every instance of 'He' in the file that was used to generate the HierarchicalModel.
<br><br>
However, if the user wants to retrieve the most similar words for a specific word, they will need to specify the line and position number. This allows for a more targeted analysis, honing in on specific areas of interest within the text.

### Get Most Similar Words for New Input
The following code will be used to get the top n most similar sentences to the input sentence
```{python}
model.model.get_most_simiar_word_for_new_input(sentence, n)
```
The code takes in two parameters. The first parameter represents the input sentence to be analyzed, while the second parameter specifies the number of most similar words (from the original dataset used to train ConceptX) to be retrieved. By default, the code retrieves the top 10 most similar words.