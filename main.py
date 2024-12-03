from data import get_tinyshakespeare
from src.interface import *


if __name__ == "__main__":
    env_mode, env_weight_out, env_epochs, env_max_iterations, env_learning_rate = parse_args()

    vocab_size, encode, decode = get_tinyshakespeare()
    print("Loaded Vocab Size:", vocab_size)
    
    test_str = "Hello, World!"
    test_encoding = encode(test_str)
    test_decoding = decode(test_encoding)
    
    print(f'Test encoding/decoding of \"{test_str}\" is {test_encoding}/{test_decoding}')