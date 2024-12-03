class Tokenizer:
    """
    Tokenizer class to encode and decode text data.
    """
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_index = {ch: i for i, ch in enumerate(self.chars)}
        self.index_to_char = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, s):
        """
        Encode a string into a list of integers.
        """
        return [self.char_to_index[c] for c in s]
    
    def decode(self, indices):
        """
        Decode a list of integers back into a string.
        """
        return ''.join([self.index_to_char[i] for i in indices])
