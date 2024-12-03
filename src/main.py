from torch import nn
from torch.utils.data import DataLoader
from data import get_tinyshakespeare
from src.interface import *
from src.train import *
from src.model import Tiny


if __name__ == "__main__":
    env_mode, env_weight_out, env_epochs, env_max_iterations, env_learning_rate = parse_args()

    dataset, tokenizer = get_tinyshakespeare()
    vocab_size = tokenizer.vocab_size
    print("Loaded Vocab Size:", vocab_size)

    model = Tiny(vocab_size)

    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    
    # test_str = "Hello, World!"
    # test_encoding = encode(test_str)
    # test_decoding = decode(test_encoding)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=env_learning_rate)

    # print(f'Test encoding/decoding of \"{test_str}\" is {test_encoding}/{test_decoding}')
    print(train(model, train_dataloader, optimizer, env_epochs))

