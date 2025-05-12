import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.utils import prune
import os

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据项处理
transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载数据集
train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transforms_train)
test_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transforms_test)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)


# 定义CNN模型
class SimpleCNN(nn.Module):
    def __int__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1.64 * 8 * 8)
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


model = SimpleCNN().to(device)


# 训练函数
def train(model, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch[{epoch + 1 / epochs}],Step[{i + 1}/{len(train_loader)}],Loss:{running_loss / 100:.4f}')
                running_loss = 0.0


# 测试函数
def test(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy:{accuracy:.2f}%')
    return accuracy


# 原始模型训练
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Training original model:")
train(model, criterion, optimizer, epochs=20)
original_acc = test(model)


# 模型剪枝函数
def prune_model(model, prune_rate=0.2):
    parameters_to_prune = (
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.fc1, 'weight'),
        (model.fc2, 'weight'),
    )

    for module, param in parameters_to_prune:
        prune.l1_unstructured(module, name=param, amount=prune_rate)
    return model


# 应用剪枝
print("\n Pruning model...")
pruned_model = prune_model(model)
pruned_acc = test(pruned_model)

# 剪枝后微调
print("\nFine-tuning pruned model:")
optimizer = optim.Adam(pruned_model.parameters(), lr=0.0001)
train(pruned_model, criterion, optimizer, epochs=5)
pruned_finetuned_acc = test(pruned_model)


# 模型量化
def quantize_model(model):
    quantized_model = torch.quantization.quantize_dynamic(
        model,  # 原始模型
        {nn.Linear, nn.Conv2d},  # 要量化的模块类型
        dtype=torch.qint8  # 量化类型
    )
    return quantized_model


# 应用量化
print("\nQuantizing model...")
quantized_model = quantize_model(pruned_model)
quantized_acc = test(quantized_model)


# 保存模型并比较大小
def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    size = os.path.getsize(filename) / 1024
    print(f"{filename} size:{size:.2f} KB")


save_model(model, 'original_model.pth')
save_model(pruned_model, 'pruned_model.pth')
save_model(quantized_model, 'quantized_model.pth')

# 结果比较
print("\nResults Comparison:")
print(f"Original Model Accuracy: {original_acc:.2f}%")
print(f"Pruned Model Accuravy: {pruned_finetuned_acc:.2f}%")
print(f"Quantized Model Accuracy: {quantized_acc:.2f}%")
