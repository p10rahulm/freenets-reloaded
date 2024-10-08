import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_optimizer_and_scheduler(model, optimizer_name, learning_rate=0.001):
    if optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    return optimizer, scheduler