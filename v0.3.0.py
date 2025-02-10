import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Load YAML configuration
def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

# Xavier/He weight initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Build model dynamically
def build_model(config):
    layers = []
    for layer in config['model']['layers']:
        if layer['type'] == 'Linear':
            layers.append(nn.Linear(layer['input_size'], layer['output_size']))
            if layer.get('activation') == 'ReLU':
                layers.append(nn.ReLU())
            elif layer.get('activation') == 'LeakyReLU':
                layers.append(nn.LeakyReLU(negative_slope=0.01))
        elif layer['type'] == 'BatchNorm':
            layers.append(nn.BatchNorm1d(layer['size']))
        elif layer['type'] == 'Dropout':
            layers.append(nn.Dropout(layer['p']))
    model = nn.Sequential(*layers)
    model.apply(init_weights)  # Apply weight initialization
    return model

# Load dataset dynamically
def load_dataset(config):
    batch_size = config['training']['batch_size']
    
    if config['training']['dataset'] == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif config['training']['dataset'] == "CIFAR10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    train_size = int((1 - config['training']['validation_split']) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Training function with adaptive learning rate, early stopping & model saving
def train_model(model, train_loader, val_loader, config):
    optimizer = getattr(optim, config['model']['optimizer']['type'])(
        model.parameters(), 
        lr=config['model']['optimizer']['learning_rate'],
        weight_decay=config['model']['optimizer'].get('weight_decay', 0)
    )
    criterion = getattr(nn, config['model']['loss'])()
    
    # Learning rate scheduler (ReduceLROnPlateau)
    scheduler = None
    if config['training'].get('reduce_lr_on_plateau', False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=config['training'].get('lr_patience', 2), factor=0.5
        )

    best_val_loss = float('inf')
    patience_counter = 0
    patience = config['training'].get('patience', 5)

    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten for fully connected layer
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient Clipping
            if config['training'].get('gradient_clipping', None):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clipping'])

            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_train_loss}")

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.view(inputs.size(0), -1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {avg_val_loss}, Accuracy: {val_accuracy}%")

        # Reduce learning rate on plateau
        if scheduler:
            scheduler.step(avg_val_loss)

        # Early stopping
        if config['training'].get('early_stopping', False):
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if config['training'].get('save_model', False):
                    torch.save(model.state_dict(), config['training']['model_path'])
                    print("Model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    break

# Main function
def main(yaml_path):
    config = load_config(yaml_path)
    model = build_model(config)
    train_loader, val_loader = load_dataset(config)
    train_model(model, train_loader, val_loader, config)

if __name__ == "__main__":
    main("config0.3.0.yaml")
