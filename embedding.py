# embedding the words based on BERT

from transformers import BertModel, BertTokenizer
import torch

class Embedding:
    def __init__(self, model_name):
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def get_embedding(self, sentence):
        input_ids = torch.tensor([self.tokenizer.encode(sentence, add_special_tokens=True)])
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]

        # map the embedding vector back to each token
        mapped_result = []
        for i in range(len(input_ids[0])):
            mapped_result.append({
                'token': self.tokenizer.decode(input_ids[0][i].item()),
                'embedding': last_hidden_states[0][i].numpy()
            })

        return mapped_result
