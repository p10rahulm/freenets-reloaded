import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def test_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch.float())
            y_true.extend(y_batch.numpy())
            y_pred.extend(outputs.numpy())
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"MSE: {mse}, MAE: {mae}, R2 Score: {r2}")
