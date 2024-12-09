from torch import nn
from time import time

# TODO: Create a train/validate/test class to modularize this better
#       for any potential model that we implement/improve upon.
def train(model, dataloader, optimizer, num_epochs):
    model.train()
    max_grad_norm = 1.0
    losses = []

    for epoch in range(num_epochs):
        time_start = time()
        for x, y in dataloader:
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            losses.append(loss.item())
            # To avoid exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        time_end = time()
        print(f"Epoch {epoch}: Loss {loss.item():.4f} Time (min) {(time_end - time_start) / 60}")
        losses.append(loss.item())
            
    return losses