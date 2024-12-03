from torch import nn

# TODO: Create a train/validate/test class to modularize this better
#       for any potential model that we implement/improve upon.
def train(model, dataloader, optimizer, num_epochs):
    model.train()
    max_grad_norm = 1.0
    losses = []

    for epoch in range(num_epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            # To avoid exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        print(loss.item())
        losses.append(loss.item())
            
    return losses