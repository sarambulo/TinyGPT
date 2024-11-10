import torch
import torch.nn as nn
from torch.nn import functional as F

def encode():
    """
    [@input]: input sequence of tokens (characters)
    [@return]: sequence of embeddings (integers)
    """
    pass

def decode():
    """
    [@input]: sequence of embeddings (integers)
    [@return]: input sequence of tokens (characters)
    """
    pass

class Tiny():
    def get_batch(self, dataset):
        """
        [@return]: matrix with batch rows, n features for colums, c channels
        """
        pass
    
    def forward(self):
        """
        [input]: batch
        [output]: logits = embedding
        """
        pass

    def generate(self, idx, max_new_tokens):
        """
        [input]: request
        [output]: respons, sequence of characters
        """
        for i in range(max_new_tokens):
            # get the predictions
            logits, _ = self(idx)
            # care about the last time step only
            logits = logits[:, -1, :]
            # use a softmax to generate the list of probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the probability distribution
            next = torch.multinomial(probs, num_samples=1)
            # add index to the running sequence
            idx = torch.cat((idx, next), dim=-1)
        return idx

