import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


# 一个简单的全连接神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)
        self.fc3 = nn.Linear(1, 20)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


x = torch.linspace(-5, 5, 100).view(-1, 1)
y = torch.randn_like(x) * 0.1  # y = 2x + 1，带有噪声

# 数据加载器
dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
# 初始化网络和损失函数
model = SimpleNN()
criterion = nn.MSELoss()

# 定义优化算法
optimizers = {
    # 'SGD': optim.SGD(model.parameters(), lr=0.0001),
    'SGD with Momentum': optim.SGD(model.parameters(), lr=0.6, momentum=0.9),
    # 'Adam': optim.Adam(model.parameters(), lr=0.0001),
    # 'AdamW': optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1),
    # 'RMSProp': optim.RMSprop(model.parameters(), lr=0.0001)
}


if __name__ == '__main__':
    # 训练过程
    num_epochs = 100
    losses = {key: [] for key in optimizers.keys()}

    for optimizer_name, optimizer in optimizers.items():
        # 重新初始化网络
        model = SimpleNN()
        optimizer = optimizer

        # 训练模型
        for epoch in range(num_epochs):
            epoch_loss = 0
            for inputs, targets in train_loader:
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # 记录每个优化器的损失
            losses[optimizer_name].append(epoch_loss / len(train_loader))

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    for optimizer_name, loss_list in losses.items():
        plt.plot(range(num_epochs), loss_list, label=optimizer_name)

    plt.title('Comparison of Optimizers')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
