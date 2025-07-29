import torch
import torch.nn as nn
from data_loader import read_dataset
# from model.ResNet import ResNet18
from model.WRN import WideResNet_28_10
from config import Config


def load_model():
    # model = ResNet18()
    # model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    # model.fc = nn.Linear(512, Config.N_CLASSES)
    model = WideResNet_28_10(num_classes=Config.N_CLASSES)
    model.load_state_dict(torch.load('checkpoint/v1.pt'))
    return model.to(Config.DEVICE)


def evaluate_model(model, test_loader):
    model.eval()
    total_sample = 0
    right_sample = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
            output = model(data)
            _, pred = torch.max(output, 1)
            correct_tensor = pred.eq(target)
            total_sample += target.size(0)
            right_sample += correct_tensor.sum().item()
    accuracy = 100 * right_sample / total_sample
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    test_loader = read_dataset(batch_size=Config.BATCH_SIZE, pic_path=Config.DATA_PATH)[2]
    model = load_model()
    evaluate_model(model, test_loader)
