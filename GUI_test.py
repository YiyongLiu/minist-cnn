import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import tkinter as tk
import tkinter.messagebox  # 导入messagebox模块
import cv2
from PIL import Image, ImageDraw
import CNN


# 加载预训练的模型
cnn = CNN.cnn_Module()
cnn.load_state_dict(torch.load('cnn.pkl'))
cnn.eval()


# 创建tkinter界面
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("手写数字识别")
        self.geometry("400x500")
        self.canvas = tk.Canvas(self, bg="white", width=280, height=280)
        self.canvas.pack(pady=20)

        # 添加“识别”按钮
        self.predict_button = tk.Button(self, text="识别", command=self.predict, width=10, height=2)
        self.predict_button.place(x=50, y=400)

        # 添加“清除”按钮
        self.clear_button = tk.Button(self, text="清除", command=self.clear, width=10, height=2)
        self.clear_button.place(x=280, y=400)

        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 3), (event.y - 3)
        x2, y2 = (event.x + 3), (event.y + 3)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def predict(self):
        self.image = self.image.resize((28, 28))
        img_array = np.array(self.image)
        img_array = (255 - img_array) / 255.0  # 反色并归一化
        img_array = img_array[np.newaxis, np.newaxis, :, :].astype(np.float32)
        img_tensor = torch.from_numpy(img_array)

        output = cnn(img_tensor)
        pred_y = torch.max(output, 1)[1].item()

        # 弹出预测结果
        result = tkinter.messagebox.showinfo("预测结果", f"预测的数字是: {pred_y}")

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)


# 运行应用
if __name__ == "__main__":
    app = App()
    app.mainloop()
