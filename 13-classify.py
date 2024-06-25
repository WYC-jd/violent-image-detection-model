import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision import models

class ViolenceClass:
    def init(self):

        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

    def classify(self, output_list):
        """
        将张量传入并分类为预测列表
        参数：output_list (list): 列表张量
        返回：pred_list: 预测列表
        """
        # 获取每个图像的预测类别
  
        probs_list = [nn.Softmax(dim=0)(logits) for logits in output_list]  

        pred_list = [torch.argmax(probs).item() for probs in probs_list]

        return pred_list

    def images_to_tensor(self, image_list):
        """
        将图片转换为张量
        参数: image_list (list): 图片列表
        返回: standard_list: 转换并归一化后的张量
        """
        # 定义一个ToTensor转换

        to_tensor = transforms.ToTensor()

        # 定义一个归一化函数
        
        def normalize(tensor):
            min_val = torch.min(tensor)
            max_val = torch.max(tensor)
            normalized_tensor = (tensor - min_val) / (max_val - min_val)

            return normalized_tensor

        # 将列表中的每个图像转换为tensor，并存储在一个新的列表中

        tensor_list = [to_tensor(image) for image in image_list]
        
        # 将列表中每个tensor归一化， 并存储在一个新的列表中

        standard_list = [normalize(tensor) for tensor in tensor_list]

        # 为每个tensor添加批次维度

        input_list = [tensor.unsqueeze(0) for tensor in standard_list]

        return input_list

    def load_images_from_path(self, path):
        """
        从指定路径加载图像并转换为RGB格式
        参数: path (str): 图像文件夹的路径
        返回: list: 包含所有图像的列表，每个图像都是PIL.Image对象
        """
        images = []
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img = Image.open(file_path).convert('RGB')
                images.append(img)
        return images


    def label_true(self, split):
        """
        获得数据集真实标签。
        参数：split：数据集类型名 + '/'
        返回：label：真实标签列表
        """

        #获得文件路径

        assert split in ["train/", "val/", "test/"]
        #data_root = "/your/path/to/violence_224/"
        data_root = "C:/Users/wyc/Desktop/violent_test/violence_224/"
        data_list = [os.path.join(data_root, split, i) for i in os.listdir(data_root + split)]

        label = [int(data.split("/")[-1][0]) for data in data_list]  # 获取标签值，0表示正常，1表示暴力

        return label

    def accuracy_score(self, y_true, y_pred):
        """
        计算模型预测的准确率。
        参数:
        y_true (list or numpy array): 真实标签。
        y_pred (list or numpy array): 预测标签。
        返回:float: 准确率，范围在0到1之间。
        """
        # 确保输入是可迭代对象
        if len(y_true) != len(y_pred):
            raise ValueError("真实标签和预测标签的长度必须相同")

        # 计算匹配的样本数
        correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    
        # 计算准确率
        accuracy = correct / len(y_true)
    
        return accuracy
