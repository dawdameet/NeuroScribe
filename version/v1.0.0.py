import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class AILangInterpreter:
    def __init__(self, script_path):
        self.config = self.load_script(script_path)
        self.model = self.build_model()
        self.optimizer, self.criterion = self.setup_training()
        self.train_loader, self.val_loader = self.load_data()
    
    def load_script(self, script_path):
        with open(script_path, 'r') as file:
            return yaml.safe_load(file)
    
    def build_model(self):
        layers = []
        for layer in self.config['model']['layers']:
            if layer['type'] == 'Linear':
                layers.append(nn.Linear(layer['input_size'], layer['output_size']))
                if layer.get('activation') == 'ReLU':
                    layers.append(nn.ReLU())
                elif layer.get('activation') == 'Softmax':
                    layers.append(nn.Softmax(dim=1))
            elif layer['type'] == 'Dropout':
                layers.append(nn.Dropout(layer['p']))
        return nn.Sequential(*layers)
    
    def setup_training(self):
        optimizer = getattr(optim, self.config['model']['optimizer']['type'])(
            self.model.parameters(), lr=self.config['model']['optimizer']['learning_rate']
        )
        criterion = getattr(nn, self.config['model']['loss'])()
        return optimizer, criterion
    
    def load_data(self):
        batch_size = self.config['training']['batch_size']
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    
    def train(self):
        for epoch in range(self.config['training']['epochs']):
            self.model.train()
            total_loss = 0
            for inputs, labels in self.train_loader:
                inputs = inputs.view(inputs.size(0), -1)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(self.train_loader)}")

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.view(inputs.size(0), -1)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Validation Accuracy: {100 * correct / total}%")

if __name__ == "__main__":
    ai_interpreter = AILangInterpreter("config.yaml")
    ai_interpreter.train()
    ai_interpreter.evaluate()
