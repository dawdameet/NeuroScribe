import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder
def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
def load_dataset(config):
    if config['dataset']['type'] == 'csv':
        df = pd.read_csv(config['dataset']['path'])
        assert set(config['dataset']['features'] + [config['dataset']['target_column']]).issubset(df.columns), \
            "Some specified features are missing from the dataset"
        categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = LabelEncoder().fit_transform(df[col])
        df = df.fillna(0)
        features = df[config['dataset']['features']].values.astype('float32')
        labels = df[config['dataset']['target_column']].values.astype('float32')
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Ensure correct shape
        dataset = TensorDataset(features, labels)
        train_size = int((1 - config['training']['validation_split']) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
        return train_loader, val_loader
    else:
        raise ValueError("Unsupported dataset type")
def build_model(config):
    layers = []
    for layer in config['model']['layers']:
        if layer['type'] == 'Linear':
            layers.append(nn.Linear(layer['input_size'], layer['output_size']))
            if 'activation' in layer:
                activation = getattr(nn, layer['activation'])()
                layers.append(activation)
        elif layer['type'] == 'Dropout':
            layers.append(nn.Dropout(layer['p']))
    return nn.Sequential(*layers)
def train_model(model, train_loader, val_loader, config):
    optimizer = getattr(optim, config['model']['optimizer']['type'])(
        model.parameters(), lr=config['model']['optimizer']['learning_rate']
    )
    criterion = getattr(nn, config['model']['loss'])()
    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss/len(val_loader)}")
def main(yaml_path):
    config = load_config(yaml_path)
    model = build_model(config)
    train_loader, val_loader = load_dataset(config)
    train_model(model, train_loader, val_loader, config)
if __name__ == "__main__":
    main("selfsuff.yaml")
