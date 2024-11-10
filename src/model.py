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

    def predict(self):
        """
        [input]: request
        [output]: respons, sequence of characters
        """
        pass

