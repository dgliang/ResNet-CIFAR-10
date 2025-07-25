import torch
import torch.nn as nn
from data_loader import read_dataset
from model.ResNet import ResNet18
from config import Config

# set device
device = Config.DEVICE
n_class = Config.N_CLASSES
batch_size = Config.BATCH_SIZE
train_loader, valid_loader, test_loader = read_dataset(batch_size=Config.BATCH_SIZE, pic_path=Config.DATA_PATH)

model = ResNet18() # 得到预训练模型
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, Config.N_CLASSES) # 将最后的全连接层修改
# 载入权重

model.load_state_dict(torch.load('checkpoint/resnet18_cifar10.pt'))
model = model.to(Config.DEVICE)

total_sample = 0
right_sample = 0
model.eval()  # 验证模型
for data, target in test_loader:
    data = data.to(Config.DEVICE)
    target = target.to(Config.DEVICE)
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data).to(Config.DEVICE)
    # convert output probabilities to predicted class(将输出概率转换为预测类)
    _, pred = torch.max(output, 1)    
    # compare predictions to true label(将预测与真实标签进行比较)
    correct_tensor = pred.eq(target.data.view_as(pred))
    # correct = np.squeeze(correct_tensor.to(device).numpy())
    total_sample += Config.BATCH_SIZE
    for i in correct_tensor:
        if i:
            right_sample += 1
print("Accuracy:",100*right_sample/total_sample,"%")
