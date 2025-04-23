from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def train_loop(model, device, optimizer, criterion, train_loader, dev_loader, dropout, epochs=100):
    lossi = []
    devlossi = []

    best_dev_loss = float('inf')
    best_model = None

    for epoch in tqdm(range(epochs)):
        model.train()
        loss_total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device) 

            logits = model(X_batch)
            loss = criterion(logits, y_batch.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        lossi.append(loss_total / len(train_loader))

        model.eval()
        with torch.no_grad():
            dev_loss_total = 0
            for X_dev, y_dev in dev_loader:
                X_dev, y_dev = X_dev.to(device), y_dev.to(device) 

                dev_logits = model(X_dev)
                dev_loss = criterion(dev_logits, y_dev.float())
                dev_loss_total += dev_loss.item()

        current_dev_loss = dev_loss_total / len(dev_loader)
        devlossi.append(current_dev_loss)
        
        if current_dev_loss < best_dev_loss:
            best_dev_loss = current_dev_loss
            best_model = type(model)(dropout=dropout)
            best_model.load_state_dict(model.state_dict())

    print(f"best dev loss is {best_dev_loss:.4f}")
    plt.plot(lossi)
    plt.plot(devlossi)
    devlossi_min = round(min(devlossi), ndigits=4)
    plt.hlines([devlossi_min], xmin=0, xmax=len(devlossi), colors='red',linestyles='dashdot', label=devlossi_min)
    plt.legend()
    plt.show()
    
    return best_model

def eval_loop(best_model, X, y):
    best_model.eval()
    pred_logits = best_model(X.cpu())

    preds = pred_logits.cpu()
    preds = (preds > 0.5).float()

    print(classification_report(y.cpu().numpy(), preds))
    ConfusionMatrixDisplay.from_predictions(y.cpu().numpy(), preds, normalize='true')
    plt.title("Normalize on True")
    plt.show()

    ConfusionMatrixDisplay.from_predictions(y.cpu().numpy(), preds, normalize='pred')
    plt.title("Normalize on Pred")
    plt.show()