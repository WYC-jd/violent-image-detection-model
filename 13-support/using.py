import torch
import os
from classify import ViolenceClass
from model import ViolenceClassifier

if __name__ == "__main__":

    # 实例化模型和接口

    model = ViolenceClassifier(learning_rate=3e-4)
    using_example = ViolenceClass() 
    
    # checkpoint_path = 'path/to/your/checkpoint.pth'
    checkpoint_path = 'C:/Users/wyc/Desktop/violent_test/train_logs/resnet18_pretrain_test/version_0/checkpoints/resnet18_pretrain_test-epoch=08-val_loss=0.05.ckpt'
        
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'No checkpoint found at {checkpoint_path}')

    # 加载checkpoint

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # 加载模型
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  

    # 从指定路径批量加载图片,返回一个图片列表，图片格式为RGB

    # img_example_list = using_example.load_images_from_path('/path/to/your/image') 
    img_example_list = using_example.load_images_from_path('C:/Users/wyc/Desktop/violent_test/violence_224/test')

    # 转换图片列表到tensor格式

    imgs_example = using_example.images_to_tensor(img_example_list)

    # 使用模型进行推理
    with torch.no_grad():
        output_list = [model(img) for img in imgs_example]

    # 调用模型获得真实和预测标签
    label_true = using_example.label_true("test/")
    label_predict = using_example.classify(output_list)

    # 计算正确率并打印

    examples_accurancy = using_example.accuracy_score(label_true, label_predict)
    print(examples_accurancy)

    # 获得预测列表并打印
    print(label_predict)



