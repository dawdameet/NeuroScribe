import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Load YAML configuration
def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Define the model based on YAML
def build_model(config):
    layers = []
    for layer in config['model']['layers']:
        if layer['type'] == 'Linear':
            layers.append(nn.Linear(layer['input_size'], layer['output_size']))
            if layer.get('activation') == 'ReLU':
                layers.append(nn.ReLU())
            elif layer.get('activation') == 'Softmax':
                layers.append(nn.Softmax(dim=1))
        elif layer['type'] == 'Dropout':
            layers.append(nn.Dropout(layer['p']))
    return nn.Sequential(*layers)

# Load dataset (example: MNIST)
def load_dataset(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Training loop
def train_model(model, train_loader, val_loader, config):
    optimizer = getattr(optim, config['model']['optimizer']['type'])(
        model.parameters(), lr=config['model']['optimizer']['learning_rate']
    )
    criterion = getattr(nn, config['model']['loss'])()
    
    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.view(inputs.size(0), -1) 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.view(inputs.size(0), -1)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Validation Accuracy: {100 * correct / total}%")

# Main function
def main(yaml_path):
    config = load_config(yaml_path)
    model = build_model(config)
    train_loader, val_loader = load_dataset(config['training']['batch_size'])
    train_model(model, train_loader, val_loader, config)

if __name__ == "__main__":
    main("config.yaml")