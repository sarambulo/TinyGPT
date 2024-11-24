from typing import Tuple
from data import get_tinyshakespeare

import argparse

def parse_args() -> Tuple[str, str, int, int, float]:
    """
    Adapted from the code provided in 10-601 at CMU

    Parses all args and returns them. Returns:

    (1) mode : "train", "test", or "interactive"
    (2) weight_out : The output path of the file containing your weights
    (3) epochs : An integer indicating the number of episodes to train for
    (4) max_iterations : An integer representing the max number of iterations for training
    (5) lr : A float representing the learning rate

    Example usage: 
    $ python main.py -m unittest
    $ python main.py -m train -w ./result.pkl -e 3 -i 1000 -lr 0.001
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--mode", type=str, choices=["train", "eval", "interactive", "unittest"])
    parser.add_argument('-w', "--weight_out", type=str, required=False)
    parser.add_argument('-e', "--epochs", type=int, required=False)
    parser.add_argument('-i', "--max_iterations", type=int, required=False)
    parser.add_argument('-lr', "--learning_rate", type=float, required=False)

    args = parser.parse_args()

    return (args.mode, args.weight_out, args.epochs, args.max_iterations, args.learning_rate)

if __name__ == "__main__":
    env_mode, env_weight_out, env_epochs, env_max_iterations, env_learning_rate = parse_args()
    print(env_mode)
    print(env_weight_out)
    print(env_epochs)
    print(env_max_iterations)
    print(env_learning_rate)

    vocab_size, encode, decode = get_tinyshakespeare()
    print("Loaded Vocab Size:", vocab_size)
    
    test_str = "Hello, World!"
    test_encoding = encode(test_str)
    test_decoding = decode(test_encoding)
    
    print(f'Test encoding/decoding of \"{test_str}\" is {test_encoding}/{test_decoding}')