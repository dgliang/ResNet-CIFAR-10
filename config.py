import torch
import os

# 创建检查点目录
os.makedirs('checkpoint', exist_ok=True)

# 训练配置
class Config:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 128
    N_EPOCHS = 300
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    N_CLASSES = 10
    DATA_PATH = 'dataset'
    CHECKPOINT_PATH = 'checkpoint/resnet18_cifar10.pt'
    LOG_PATH = 'logs/training.log'
    TENSORBOARD_PATH = 'runs/resnet18_cifar10'