import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from tqdm import tqdm

class BERTSimilarity:
    def __init__(self, model_name="bert-base-uncased", vocab_size=5000, batch_size=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        raw_vocab = list(self.tokenizer.get_vocab().keys())
        self.vocab = self._filter_vocab(raw_vocab)[:vocab_size]
        self.batch_size = batch_size

        self.token_embeddings = self._compute_vocab_embeddings()

    def _filter_vocab(self, vocab):
        """
        Removes special tokens, subwords, non-alpha tokens, short words.
        """
        filtered = [
            token for token in vocab
            if token.isalpha() and not token.startswith("##") and len(token) > 2
        ]
        return filtered

    def _compute_vocab_embeddings(self):
        """
        Compute embeddings for filtered vocab using batching.
        """
        embeddings = []
        for i in tqdm(range(0, len(self.vocab), self.batch_size), desc="Computing embeddings"):
            batch_tokens = self.vocab[i:i+self.batch_size]
            inputs = self.tokenizer(batch_tokens, return_tensors="pt", padding=True, truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeds = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeds)

        return torch.cat(embeddings, dim=0)


    def get_similar_words(self, input_word, top_n=10):
        """
        Given an input word, return top N similar words.
        """
        inputs = self.tokenizer(input_word, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            input_embedding = outputs.last_hidden_state.mean(dim=1)

        similarities = F.cosine_similarity(input_embedding, self.token_embeddings, dim=1)
        top_indices = torch.topk(similarities, top_n).indices
        similar_words = [(self.vocab[i], float(similarities[i])) for i in top_indices]
        return similar_words