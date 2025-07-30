import logging
from config import Config


def adjust_learning_rate(optimizer, epoch, writer=None):
    """ Sets the learning rate to the initial LR divided by 5 at 50th, 100th , 150th and 200th epochs """
    lr = Config.LEARNING_RATE
    
    if epoch >= 50 and epoch < 100:
        lr *= 0.2
    elif epoch >= 100 and epoch < 150:
        lr *= 0.04
    elif epoch >= 150 and epoch < 200:
        lr *= 0.008
    elif epoch >= 200:
        lr *= 0.004
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    writer.add_scalar('Learning Rate', lr, epoch)
    logging.info(f"Adjusted learning rate: {lr:.6f}")
    return lr