import torch
import numpy as np
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs=25, patience=3):

    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(total=len(train_loader.dataset), desc=f"Epoch {epoch+1}/{num_epochs}", unit="sample") as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                
                optimizer.zero_grad()
                
                # Forward-Pass
                outputs, _ = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                pbar.update(inputs.size(0))
        
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
    
        val_loss /= len(val_loader)
    
        print(f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
        # Early Stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early Stopping is triggered. End model training!")
            break

    print('Training complete')

def extract_feature_vectors(model, device, data_loader):
    model.eval()
    feature_vectors = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            _, features = model(inputs)  # Extract feature vector
            feature_vectors.append(features.cpu().numpy())
    
    feature_vectors = np.concatenate(feature_vectors, axis=0)
    return feature_vectors

# One-hot encoding for the PTB XL labels
def ptb_xl_ecg_labeling(y_train):

    y_train = list(y_train)

    y_train_ = [[0,0] for i in list(y_train)]
    for index, e in enumerate(y_train):
        if e == 0:
            y_train_[index][0] = 1
        elif e == 1:
            y_train_[index][1] = 1

    y_train_ = np.array(y_train_)
    return y_train_