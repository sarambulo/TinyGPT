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

class Tiny(nn.Module):
    """
    Bigram Model

    Attributes
    - `token_embedding_table`: Table with the probability of ocurrence 
    of each token (columns) given the previous token (rows)
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def get_batch(self, dataset):
        """
        [@return]: matrix with batch rows, n features for colums, c channels
        """
        pass
    
    def forward(self, X_train: torch.Tensor, y_train_true: torch.Tensor): 
        """
        Calculates the prediction (`y_train_hat`) for a given batch of tokens
        and the Categorical Cross Entropy Loss for that prediction (`loss`)

        :param X_train: (B, T) Tensor with B batches and T **encoded** tokens
        :return y_train_hat: (B, T, C) Tensor with B batches, T tokens and C channels
        :return loss: (1) Tensor with the Categorical Cross Entropy
        """
        y_train_hat = self.token_embedding_table(X_train) # (B, T, C)
        # Reshape Tensors to comform to the cross_entropy requirements
        # The channel must be the second dimension
        B, T, C = y_train_hat.shape
        y_train_hat = y_train_hat.view(B * T, C)
        y_train_true = y_train_true.view(B * T)
        loss = F.cross_entropy(y_train_hat, y_train_true)
        return y_train_hat, loss

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
