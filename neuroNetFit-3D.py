import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from mpl_toolkits.mplot3d import Axes3D

# 加载数据
file_path = "/Users/zyd/Downloads/Gaussian/outputExcel.xlsx"  # 替换为您的文件路径
data = pd.read_excel(file_path)

# 准备数据
X = torch.tensor(data.iloc[:, :2].values, dtype=torch.float32)
y = torch.tensor(data.iloc[:, 2].values, dtype=torch.float32).view(-1, 1)

# 数据分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建数据加载器
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)


# 构建神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 128)  # 更多的神经元
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


model = Net()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(100):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


# 绘制等高线图
def plot_3d(model, X, y):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 真实的数据点
    ax.scatter(X[:, 0], X[:, 1], y.view(-1).numpy(), color="red", label="Real data")

    with torch.no_grad():
        # 创建网格
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        # 预测网格点的z值
        Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32))
        Z = Z.view(xx.shape)

        # 绘制曲面
        ax.plot_surface(xx, yy, Z.numpy(), alpha=0.5, cmap="winter")

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_zlabel("Z value")
    ax.legend()
    plt.show()


plot_3d(model, X, y)
