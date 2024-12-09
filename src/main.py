import torch
from torch import nn
from torch.utils.data import DataLoader
from data import get_tiny_shakespeare_data
from interface import *
from train import *
from model import TinyBigram
from pathlib import Path


if __name__ == "__main__":
    env_mode, env_weight_out, env_losses_out, env_epochs, env_batch_size, env_max_iterations, env_learning_rate = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device.upper()}")

    train_dataset, val_dataset, tokenizer = get_tiny_shakespeare_data(block_size=64, device=device)
    vocab_size = tokenizer.vocab_size
    print("Loaded Vocab Size:", vocab_size)

    model = TinyBigram(vocab_size, device)
    model = model.to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=env_batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    
    # test_str = "Hello, World!"
    # test_encoding = encode(test_str)
    # test_decoding = decode(test_encoding)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=env_learning_rate)

    # print(f'Test encoding/decoding of \"{test_str}\" is {test_encoding}/{test_decoding}')
    losses = train(model, train_dataloader, optimizer, env_epochs)
    # Store weigths
    with Path(env_losses_out).open('w') as f:
        f.write(f"epoch,loss\n")
        for epoch, loss in enumerate(losses):
            f.write(f"{epoch},{loss}\n")
    print(losses)

