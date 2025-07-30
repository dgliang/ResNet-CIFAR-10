import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optimf
from torch.utils.tensorboard import SummaryWriter
from config import Config
import logging
from data_loader import read_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# from model.WRN import WideResNet_28_10
from model.wrn_highperf import WideResNet_28_10


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
    model = WideResNet_28_10(
        num_classes=Config.N_CLASSES,
        droprate=0.3,
        use_bn=True,
        use_fixup=False
    )
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


def save_model(model, valid_loss, best_loss, path=Config.CHECKPOINT_PATH):
    if valid_loss <= best_loss:
        logging.info(f'Validation loss decreased ({best_loss:.6f} --> {valid_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        return valid_loss
    return best_loss


def adjust_learning_rate(optimizer, epoch):
    """ WRN 分段式 step schedule：在 50/100/140/180 时降低 LR """
    lr = Config.LEARNING_RATE
    if epoch >= 180:
        lr *= 0.004
    elif epoch >= 140:
        lr *= 0.008    # 0.05 * 0.008 = 0.0004
    elif epoch >= 100:
        lr *= 0.02
    elif epoch >= 50:
        lr *= 0.2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    writer.add_scalar('Learning Rate', lr, epoch)
    logging.info(f"Adjusted learning rate: {lr:.6f}")


def main():
    # 创建模型
    model = create_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(Config.DEVICE)
    optimizer = optimf.SGD(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        momentum=Config.MOMENTUM, 
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=Config.T_0, T_mult=Config.T_mult, eta_min=Config.ETA_MIN)
    
    # 加载数据
    train_loader, valid_loader, test_loader = read_dataset(
        batch_size=Config.BATCH_SIZE, 
        pic_path=Config.DATA_PATH
    )
    
    # 训练前的准备
    best_loss = np.inf
    accuracy_history = []
    
    # 开始训练
    logging.info(f'Starting training on {Config.DEVICE} for {Config.N_EPOCHS} epochs')
    for epoch in tqdm(range(1, Config.N_EPOCHS + 1)):
        # 调整学习率
        adjust_learning_rate(optimizer, epoch)

        print(f"Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        logging.info(f'Epoch: {epoch}/{Config.N_EPOCHS}')
        
        # 训练和验证
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch)
        valid_loss, accuracy = validate_model(model, valid_loader, criterion, epoch)
        accuracy_history.append(accuracy)
        
        # 保存最佳模型
        best_loss = save_model(model, valid_loss, best_loss)

        # scheduler.step()
        # current_lr = optimizer.param_groups[0]['lr']
        # writer.add_scalar('Learning Rate', current_lr, epoch)
    
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