import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optimf
from torch.utils.tensorboard import SummaryWriter
from config import Config
import logging
from data_loader import read_dataset
# from model.ResNet import ResNet18
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.ResNet import ResNet50


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_PATH),
        logging.StreamHandler()
    ]
)
writer = SummaryWriter(Config.TENSORBOARD_PATH)


def create_model():
    # model = ResNet18()
    model = ResNet50(num_classes=Config.N_CLASSES)
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    # 将最后的全连接层改掉
    # model.fc = torch.nn.Linear(512, Config.N_CLASSES)
    model.fc = torch.nn.Linear(2048, Config.N_CLASSES)
    return model.to(Config.DEVICE)


def train_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0.0
    
    for data, target in train_loader:
        data = data.to(Config.DEVICE)
        target = target.to(Config.DEVICE)

        optimizer.zero_grad()
        output = model.forward(data).to(Config.DEVICE)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*data.size(0)

    # 计算平均损失
    train_loss = train_loss / len(train_loader.sampler)
    
    # 记录到TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    logging.info(f'Train Loss: {train_loss:.6f}')
    
    return train_loss


def validate_model(model: nn.Module, valid_loader, criterion, epoch: int):
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in valid_loader:
            data = data.to(Config.DEVICE)
            target = target.to(Config.DEVICE)
            
            output = model.forward(data).to(Config.DEVICE)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    # 计算指标
    valid_loss = valid_loss / len(valid_loader.sampler)
    accuracy = 100.0 * correct / total
    
    # 记录到TensorBoard
    writer.add_scalar('Loss/valid', valid_loss, epoch)
    writer.add_scalar('Accuracy/valid', accuracy, epoch)
    logging.info(f'Valid Loss: {valid_loss:.6f} | Valid Accuracy: {accuracy:.2f}%')
    
    return valid_loss, accuracy


def adjust_learning_rate(optimizer, epoch, initial_lr=Config.LEARNING_RATE):
    """*手动调整* 学习率"""
    lr = initial_lr
    # 每10个epoch将学习率减半
    if epoch % 10 == 0 and epoch > 0:
        lr *= 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logging.info(f'Learning Rate adjusted to: {lr:.6f}')
    
    writer.add_scalar('Learning Rate', lr, epoch)
    return lr


def save_model(model, valid_loss, best_loss, path=Config.CHECKPOINT_PATH):
    if valid_loss <= best_loss:
        logging.info(f'Validation loss decreased ({best_loss:.6f} --> {valid_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        return valid_loss
    return best_loss


def main():
    # 创建模型
    model = create_model()
    criterion = nn.CrossEntropyLoss().to(Config.DEVICE)
    optimizer = optimf.SGD(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        momentum=Config.MOMENTUM, 
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.N_EPOCHS) # CosineAnnealingLR 学习率调度器 
    
    # 加载数据
    train_loader, valid_loader, test_loader = read_dataset(
        batch_size=Config.BATCH_SIZE, 
        pic_path=Config.DATA_PATH
    )
    
    # 训练前的准备
    best_loss = np.Inf
    accuracy_history = []
    
    # 开始训练
    logging.info(f'Starting training on {Config.DEVICE} for {Config.N_EPOCHS} epochs')
    for epoch in tqdm(range(1, Config.N_EPOCHS + 1)):
        print(f"Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        logging.info(f'Epoch: {epoch}/{Config.N_EPOCHS}')
        
        # 训练和验证
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch)
        valid_loss, accuracy = validate_model(model, valid_loader, criterion, epoch)
        accuracy_history.append(accuracy)
        
        # 保存最佳模型
        best_loss = save_model(model, valid_loss, best_loss)

        # 调整学习率
        # current_lr = adjust_learning_rate(optimizer, epoch)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)
    
    # 训练结束
    writer.close()
    logging.info('Training completed!')
    
    # 测试最终模型
    logging.info('Evaluating model on test set...')
    model.load_state_dict(torch.load(Config.CHECKPOINT_PATH))
    _, test_accuracy = validate_model(model, test_loader, criterion, Config.N_EPOCHS + 1)
    logging.info(f'Final Test Accuracy: {test_accuracy:.2f}%')


if __name__ == "__main__":
    main()    