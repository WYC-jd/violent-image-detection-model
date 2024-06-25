from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import ViolenceClassifier
from dataset import CustomDataModule
import multiprocessing

if __name__ == '__main__':
    # 在 Windows 上使用 multiprocessing 时需要加上这行
    multiprocessing.freeze_support()

    # 参数设置
    # gpu_id=[1]
    lr = 3e-4
    batch_size = 128
    log_name = "resnet18_pretrain_test"
    print("{} cpu, batch size: {}, lr: {}".format(log_name, batch_size, lr))

    # 创建数据模块
    data_module = CustomDataModule(batch_size=batch_size)

    # 设置模型检查点回调
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=log_name + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    # 设置 TensorBoard 日志记录器
    logger = TensorBoardLogger("train_logs", name=log_name)

    # 创建 Trainer 实例
    trainer = Trainer(
        max_epochs=40,
        accelerator='cpu',  # 使用CPU
        # devices=gpu_id,   # 如果使用GPU，需要配置合适的设备ID
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    # 创建模型
    model = ViolenceClassifier(learning_rate=lr)

    # 开始训练
    trainer.fit(model, data_module)




