from typing import Tuple
import argparse

def parse_args() -> Tuple[str, str, int, int, float]:
    """
    Adapted from the code provided in 10-601 at CMU

    Parses all args and returns them. Returns:

    (1) mode : "train", "test", or "interactive"
    (2) weight_out : The output path of the file containing your weights
    (3) losses_out: The output path for the losses during training
    (4) epochs : An integer indicating the number of episodes to train for
    (5) batch_size : An integer indicating the number of observations per batch
    (6) max_iterations : An integer representing the max number of iterations for training
    (7) lr : A float representing the learning rate

    Example usage: 
    $ python main.py -m unittest
    $ python main.py -m train -w ./result.pkl -b 32 -e 50 -i 1000 -lr 0.001
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--mode", type=str, choices=["train", "eval", "interactive", "unittest"])
    parser.add_argument('-w', "--weight_out", type=str, required=False, default="weights.txt")
    parser.add_argument('-l', "--losses_out", type=str, required=False, default="losses.csv")
    parser.add_argument('-e', "--epochs", type=int, required=False, default=10)
    parser.add_argument('-b', "--batch_size", type=int, required=False, default=256)
    parser.add_argument('-i', "--max_iterations", type=int, required=False)
    parser.add_argument('-lr', "--learning_rate", type=float, required=False, default=0.001)

    args = parser.parse_args()

    return (
        args.mode, args.weight_out, args.losses_out, args.epochs,
        args.batch_size, args.max_iterations, args.learning_rate
    )