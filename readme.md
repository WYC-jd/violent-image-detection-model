## 接口调用实例说明



```python
from classify import ViolenceClass


if __name__ == "__main__":

    # 实例化该接口类

    using_example = ViolenceClass()

    # 从指定路径批量加载图片,返回一个图片列表，图片格式为RGB

    img_example_list = using_example.load_images_from_path('/path/to/your/image')

    # 转换图片列表到tensor格式

    imgs_example = using_example.image_to_tensor(img_example_list)

    # 调用模型

    predicted_output = using_example.classify(imgs_example)

    # 计算正确率并打印

    examples_accurancy = using_example.accuracy_score(using_example.label, predicted_output)
    print(examples_accurancy)

    # 获得预测列表并打印
    examples_pred = using_example.classify(imgs_example)
    print(examples_pred)
```