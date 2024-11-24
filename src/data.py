import os
import torch
from torch.utils.data import Dataset

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

class TextDataset(Dataset):
    """
    Dataset class to handle text data and batching.
    """
    def __init__(self, text, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = self.tokenizer.encode(text)
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        """
        Returns a single input-target pair for training.

        Args:
        - idx (int): The starting index for the sequence in the dataset.

        Returns:
        - x (torch.Tensor): A tensor of size `block_size` containing a sequence of tokens.
        - y (torch.Tensor): A tensor of size `block_size` containing the target sequence,
        which is offset by 1 token relative to `x`.

        Explanation:
        - `block_size`: The length of the sequence used as input to the model.
        - `y` is shifted by one position relative to `x` because the model is trained
        to predict the next token in the sequence for each position in `x`.
        """
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def get_tiny_shakespeare_data(block_size):
    """
    Load and tokenize the Tiny Shakespeare dataset.
    Returns the dataset and tokenizer.
    """
    TINY_SHAKESPEARE_PATH = os.path.join("data", "tinyshakespeare", "input.txt")
    with open(TINY_SHAKESPEARE_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer = Tokenizer(text)
    dataset = TextDataset(text, tokenizer, block_size)
    return dataset, tokenizer

if __name__ == "__main__":
    block_size = 128  # Example block size
    dataset, tokenizer = get_tiny_shakespeare_data(block_size)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Dataset size: {len(dataset)}")
