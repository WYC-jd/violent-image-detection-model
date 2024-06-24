from classify import ViolenceClass


if __name__ == "__main__":

    # 实例化该接口类

    using_example = ViolenceClass()

    # 加载checkpoint并返回模型
    
    checkpoint_path = 'path/to/your/checkpoint.pth'
    model, epoch, loss = using_example.load_checkpoint(checkpoint_path, device='gpu')

    # 打印轮次，损失值

    print(f'Model loaded. Epoch: {epoch}, Loss: {loss}')

    # 从指定路径批量加载图片,返回一个图片列表，图片格式为RGB

    img_example_list = using_example.load_images_from_path('/path/to/your/image') //"C:/Users/wyc/Desktop/暴力图片检测模型/output/output"

    # 转换图片列表到tensor格式

    imgs_example = using_example.image_to_tensor(img_example_list)

    # 调用模型分类

    predicted_output = using_example.classify(imgs_example)

    # 计算正确率并打印

    examples_accurancy = using_example.accuracy_score(using_example.label, predicted_output)
    print(examples_accurancy)

    # 获得预测列表并打印
    examples_pred = using_example.classify(imgs_example)
    print(examples_pred)

