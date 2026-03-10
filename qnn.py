import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import brevitas.nn as qnn


# 2. Improved CNN Architecture (VGG-style)
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.network = nn.Sequential(
            # Quantize the first input
            qnn.QuantIdentity(bit_width=4, return_quant_tensor=True),
            # Block 1: Input 32x32
            qnn.QuantConv2d(3, 32, kernel_size=3, padding=1, weight_bit_width=4),
            nn.BatchNorm2d(32),
            qnn.QuantReLU(bit_width=4, return_quant_tensor=True),
            qnn.QuantConv2d(32, 64, kernel_size=3, stride=1, padding=1, weight_bit_width=4),
            qnn.QuantReLU(bit_width=4, return_quant_tensor=True),
            nn.MaxPool2d(2, 2), # Output: 64 x 16 x 16

            # Block 2: 
            qnn.QuantConv2d(64, 128, kernel_size=3, stride=1, padding=1, weight_bit_width=4),
            nn.BatchNorm2d(128),
            qnn.QuantReLU(bit_width=4, return_quant_tensor=True),
            qnn.QuantConv2d(128, 128, kernel_size=3, stride=1, padding=1, weight_bit_width=4),
            qnn.QuantReLU(bit_width=4, return_quant_tensor=True),
            nn.MaxPool2d(2, 2), # Output: 128 x 8 x 8

            # Block 3:
            qnn.QuantConv2d(128, 256, kernel_size=3, stride=1, padding=1, weight_bit_width=4),
            nn.BatchNorm2d(256),
            qnn.QuantReLU(bit_width=4, return_quant_tensor=True),
            qnn.QuantConv2d(256, 256, kernel_size=3, stride=1, padding=1, weight_bit_width=4),
            qnn.QuantReLU(bit_width=4, return_quant_tensor=True),
            nn.MaxPool2d(2, 2), # Output: 256 x 4 x 4

            # Classifier
            nn.Flatten(), 
            qnn.QuantLinear(256 * 4 * 4, 1024, weight_bit_width=4),
            qnn.QuantReLU(bit_width=4, return_quant_tensor=True),
            nn.Dropout(0.5),
            qnn.QuantLinear(1024, 512, weight_bit_width=4),
            qnn.QuantReLU(bit_width=4, return_quant_tensor=True),
            nn.Dropout(0.2),
            qnn.QuantLinear(512, 10, weight_bit_width=4)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    # 1. Improved Data Augmentation
    # Augmentation is key to preventing overfitting on small images
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # CIFAR-10 means/stds
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    # 3. Training Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    model = ImprovedCNN().to(device)
    
    # Using Adam Optimizer and a Learning Rate Scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    # 4. Training Loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Simple Validation Accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_loss = running_loss / len(trainloader)
        scheduler.step(avg_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Test Acc: {val_acc:.2f}%')
    
    # Save the trained model
    torch.save(model.state_dict(), "quantized_cifar10.pth")
    print("Training Complete!")
