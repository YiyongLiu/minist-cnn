import torch
import torchvision
import numpy as np
import cv2
import CNN


def load_and_train_model(cnn, epochs):
    """
    This function is used to train the model.
    """
    train = CNN.training_model(cnn, epochs)
    train.training()
    train.plot_func()


def test_model(cnn, test_x, test_y):
    """
    This function is used to test the model performance.
    """
    cnn.eval()

    # 随机选择 32 个测试样本
    indices = np.random.choice(len(test_x), 32, replace=False)
    inputs = test_x[indices]
    corrects = np.array(test_y[indices])
    test_output = cnn(inputs)
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    print('prediction number', pred_y)
    print('   correct number', corrects)

    # 可视化
    img = torchvision.utils.make_grid(inputs).numpy().transpose(1, 2, 0)
    cv2.namedWindow('win', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('win', 800, 600)
    cv2.imshow('win', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 加载测试数据
    test_data = torchvision.datasets.MNIST(
        root='./data/',
        train=False,
        transform=torchvision.transforms.ToTensor()
    )
    test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255.0  # 归一化
    test_y = test_data.targets[:2000]
    # 初始化模型
    cnn = CNN.cnn_Module()
    # 训练轮数
    epochs = 5
    # 提问用户是否需要训练网络
    choice = input("Do you want to train the network? (yes/no): ").strip().lower()

    if choice == "yes":
        load_and_train_model(cnn, epochs)
    elif choice == "no":
        # 加载训练好的模型

        cnn.load_state_dict(torch.load('cnn.pkl'))  # 加载已保存的模型
        print("Model loaded successfully!")

    # 测试模型
    test_model(cnn, test_x, test_y)
