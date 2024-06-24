from model import ViolenceClassifier
from dataset import CustomDataModule
from dataset import CustomDataSet



class ViolenceClass:
    def init(self):

        #调用ViolenceClassifier的初始化函数

        ViolenceClassifier.__init__(self, num_classes=2, learning_rate=1e-3)

    def load_checkpoint(checkpoint_path, device='gpu'):
        """
        加载checkpoint文件并返回模型

        Args:
            ViolenceClassifier: 模型类
            checkpoint_path: checkpoint文件路径
            device: 设备类型，例如 'gpu' 或 'cuda'

        Returns:
            model: 加载后的模型
            epoch: 训练的epoch数
            loss: 训练的损失值
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'No checkpoint found at {checkpoint_path}')

        # 加载checkpoint

        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 实例化模型

        model = ViolenceClassifier(learning_rate=lr)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 获取其他信息
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        print(f'Checkpoint loaded from {checkpoint_path}')
        return model, epoch, loss

    def classify(standard_list):
        """
        将张量传入并分类为预测列表
        参数：tensor_list (list): 列表张量
        返回：pred_list: 预测列表
        """
        output_list = self.model(standard_list)

        # 获取每个图像的预测类别

        pred_list = torch.argmax(output_list, dim=1)  

        return pred_list

    def images_to_tensor(image_list):
        """
        将图片转换为张量
        参数: image_list (list): 图片liebiao
        返回: standard_list: 转换并归一化后的张量
        """
        # 定义一个ToTensor转换

        to_tensor = transforms.ToTensor()

        # 定义一个ToStandard标准化

        to_standard = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  

        # 将列表中的每个图像转换为tensor，并存储在一个新的列表中

        tensor_list = [to_tensor(image) for image in image_list]
        
        # 将列表中每个tensor归一化， 并存储在一个新的列表中

        standard_list = [to_standard(tensor) for tensor in tensor_list]

        return standard_list

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


    def accuracy_score(y_true, y_pred):
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
