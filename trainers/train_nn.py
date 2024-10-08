import torch
import torch.nn as nn
from tqdm import tqdm

def L_infinity_loss(output, target):
    return torch.max(torch.abs(output - target))

def train_model(model, train_loader, num_epochs, optimizer, scheduler):
    # criterion = nn.MSELoss()
    criterion = L_infinity_loss
    
    model.train()
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        # for x_batch, y_batch in tqdm(train_loader, desc="Training Batch"):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch.float())
            loss = criterion(outputs, y_batch.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    return model