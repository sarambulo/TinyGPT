from torch import nn
from data import get_tinyshakespeare

# TODO: Create a train/validate/test class to modularize this better
#       for any potential model that we implement/improve upon.
def train(model, dataloader, optimizer, num_epochs):
    model.train()
    max_grad_norm = 1.0

    for epoch in range(num_epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            # To avoid exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

if __name__ == "__main__":
    vocab_size, encode, decode = get_tinyshakespeare()
    print("Loaded Vocab Size:", vocab_size)
    
    test_str = "Hello, World!"
    test_encoding = encode(test_str)
    test_decoding = decode(test_encoding)
    
    print(f'Test encoding/decoding of \"{test_str}\" is {test_encoding}/{test_decoding}')