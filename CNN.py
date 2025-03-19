import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同

# 超参数
BATCH_SIZE = 50  # 每批次的样本数
LR = 0.001  # 学习率
# DOWNLOAD_MNIST = True  # 表示是否下载数据集
DOWNLOAD_MNIST = False


# 定义CNN模型
class cnn_Module(nn.Module):
    def __init__(self):
        super(cnn_Module, self).__init__()

        # 第一个卷积层 -> 激活函数(ReLU) -> 最大池化(MaxPooling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # 输入图像为1通道(28x28)
                in_channels=1,  # 输入图片的通道数
                out_channels=16,  # 输出特征图的通道数
                kernel_size=5,  # 卷积核大小 5x5
                stride=1,  # 步长为1
                padding=2,  # 填充操作保持图片大小不变 输出尺寸=（输入尺寸-卷积核尺寸+2*padding）/步长 +1
            ),
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool2d(2),  # 最大池化层，2x2
            # 输出: (16, 14, 14)
        )
        # 第二个卷积层 -> 激活函数(ReLU) -> 最大池化(MaxPooling)
        self.conv2 = nn.Sequential(
            nn.Conv2d(  # 输入图像为(16,14,14)
                in_channels=16,
                out_channels=32,  # 输出32个特征图
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 最大池化，2x2
            # 输出: (32, 7, 7)
        )
        # 第三个卷积层 -> 激活函数(ReLU) -> 最大池化(MaxPooling)
        self.conv3 = nn.Sequential(
            nn.Conv2d(  # 输入图像为(32, 7, 7)
                in_channels=32,
                out_channels=64,  # 输出64个特征图
                kernel_size=3,  # 卷积核大小为3x3
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 最大池化，2x2
            # 输出: (64, 3, 3)
        )
        # 全连接层
        self.out = nn.Linear(64 * 3 * 3, 10)  # 输入是64个特征图，每个特征图3x3大小，输出10个类别

    def forward(self, x):
        x = self.conv1(x)  # 通过第一个卷积层
        x = self.conv2(x)  # 通过第二个卷积层
        x = self.conv3(x)  # 通过第三个卷积层
        x = x.view(x.size(0), -1)  # 将每个批次的特征图展平成一个一维向量 这一步输出：（B,64*3*3),其中B为批次大小
        output = self.out(x)  # 通过全连接层得到输出
        return output


class training_model:
    def __init__(self, cnn, epochs):
        # 记录损失和准确率
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.epochs = epochs
        self.cnn = cnn
        self.optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
        self.loss_func = nn.CrossEntropyLoss()

        # 打印网络结构
        print(self.cnn)

    def compute_loss_and_accuracy(self, output, labels):
        """
        计算损失和预测准确率
        """
        loss = self.loss_func(output, labels)
        _, predicted = torch.max(output, 1)
        num_correct = (predicted == labels).sum().item()
        accuracy = num_correct / len(labels)
        return loss, accuracy

    def training(self):
        # 下载MNIST手写数据集
        train_data = torchvision.datasets.MNIST(
            root='./data/',  # 保存数据集的路径
            train=True,  # 训练数据集
            transform=torchvision.transforms.ToTensor(),  # 转换为Tensor格式
            download=DOWNLOAD_MNIST,  # 如果为True则下载数据集
        )

        test_data = torchvision.datasets.MNIST(
            root='./data/',
            train=False,  # 测试数据集
            transform=torchvision.transforms.ToTensor()  # 转换为Tensor格式
        )

        # 数据加载器，用于从 train_data 数据集中按批次加载训练数据，每批次大小为包含batch_size个样本
        train_loader = Data.DataLoader(
            dataset=train_data,
            batch_size=BATCH_SIZE,
            shuffle=True  # 打乱数据
        )

        # 测试集的处理（为节省时间，只使用前2000个测试数据）
        test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255.0  # 归一化
        test_y = test_data.targets[:2000]

        # 训练过程 epochs:训练轮数
        for epoch in range(self.epochs):
            train_loss = 0
            train_accuracy = 0
            for step, (b_x, b_y) in enumerate(train_loader):
                # step:批次索引 b_x:当前批次的输入数据（50张图像），形状为[50,1,28,28] b_y:当前批次的标签（50个标签），形状为[50]
                output = self.cnn(b_x)  # 将数据输入模型，得到预测结果
                loss, accuracy = self.compute_loss_and_accuracy(output, b_y)  # 计算该批次的损失和预测准确率

                self.optimizer.zero_grad()  # 清除之前的梯度
                loss.backward()  # 反向传播计算梯度
                self.optimizer.step()  # 更新参数

                train_loss += loss.item()  # 累加所有批次的损失
                train_accuracy += accuracy  # 累加所有批次的预测准确率

            # 打印每轮训练过程的平均损失和准确率
            print(f"Epoch {epoch+1}: Train Loss: {train_loss/ len(train_loader):.4f}, Train Accuracy: {train_accuracy/ len(train_loader):.2f}")
            print

            # 记录每轮训练过程的平均损失和准确率
            self.train_losses.append(train_loss / len(train_loader))  # len(train_loader)得到的是批次数量
            self.train_accuracies.append(train_accuracy / len(train_loader))

            # 每个epoch结束后在测试集上评估模型
            self.evaluate(test_x, test_y, epoch)

        # 保存模型
        torch.save(self.cnn.state_dict(), 'cnn.pkl')

    def evaluate(self, test_x, test_y, epoch):
        self.cnn.eval()
        with torch.no_grad():
            test_output = self.cnn(test_x)
            test_loss, test_accuracy = self.compute_loss_and_accuracy(test_output, test_y)

            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_accuracy)

            print(f"Epoch {epoch+1}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}")

    def plot_func(self):
        # 绘制损失和准确率曲线
        plt.figure(figsize=(12, 5))

        # 绘制训练和测试损失
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs-CNN')
        plt.legend()

        # 绘制训练和测试准确率
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs-CNN')
        plt.legend()

        # 保存绘图结果
        plt.savefig('loss_accuracy_curves-CNN.png')
        plt.show()

