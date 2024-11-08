import os
import torch

"""
Tiny Shakespeare Dataset
Retrieved from: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
"""
TINY_SHAKESPEARE_PATH = os.path.join("data", "tinyshakespeare", "input.txt")

def get_tinyshakespeare():
    """
    Load and tokenize the Tiny Shakespeare dataset
    
    Outputs:
     * vocab_size = size of vocabulary
     * encode = encoder function: str -> character embedding
     * decode = decoder function: character embedding -> str
    """
    with open(TINY_SHAKESPEARE_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    char_to_index = {ch:i for (i,ch) in enumerate(chars)}
    index_to_char = {i:ch for (i,ch) in enumerate(chars)}

    def encode(s):
        return [char_to_index[c] for c in s]
    
    def decode(l):
        chars_array = [index_to_char[i] for i in l]
        return ''.join(chars_array)

    return vocab_size, encode, decode

if __name__ == "__main__":
    get_tinyshakespeare()
