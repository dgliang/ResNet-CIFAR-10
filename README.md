# ResNet_CIFAR10

本项目旨在使用残差神经网络（ResNet）在 CIFAR-10 数据集上进行图像分类训练，最终在测试集上获得超过 **96% 的准确率**。最终模型选择为 `WideResNet-28-10`，配合多种训练技巧与数据增强方法提升了模型性能。

---

## 1. CIFAR-10 数据集简介

CIFAR-10 是一个标准的图像分类数据集，包含 10 个类别，共计 60000 张 `32x32` 像素的 RGB 彩色图像：

* 训练集：50000 张
* 测试集：10000 张
* 每张图像维度为：`32 × 32 × 3`

---

## 2. 模型选择过程

本项目并非一开始就使用 `WideResNet-28-10`，而是经过了以下模型尝试与比较：

* **ResNet-18**
  模型较浅，训练速度快，但准确率不足，仅达到 **88.78%**

* **ResNet-50**
  深度较大但容易过拟合，准确率反而下降至 **84.62%**

* **WideResNet-28-10**
  深度适中，通道更宽，表达能力强，是 ResNet 的一种并行扩容版本
  最终选择此模型，首轮训练准确率即达到 **94.05%**

---

## 3. 数据增强策略

为了提高模型的泛化能力，我们在训练集上采用了多种数据增强方法：

### （1）归一化

使用 CIFAR-10 标准的均值和标准差。

```python
transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
```

### （2）随机水平翻转

```python
transforms.RandomHorizontalFlip()
```

### （3）填充后随机裁剪

将图像从 `32x32` 填充到 `40x40`，再裁剪回 `32x32`：

```python
transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4,4,4,4), mode="reflect").squeeze()),
transforms.ToPILImage(),
transforms.RandomCrop(32),
```

### （4）Cutout 遮挡

随机遮挡图像的一小块区域来抑制过拟合：

```python
Cutout(n_holes=1, length=5)  # 遮挡一个 5x5 的区域
```

### （5）AutoAugment 策略

引入 AutoAugment for CIFAR-10，自动搜索得到的 25 种最优组合，包括旋转、颜色扰动、平移等。

```python
AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
```

---

## 4. 训练策略

* 总训练轮数：**250 epochs**
* Batch size：**128**
* 优化器：**SGD + Momentum + Nesterov**

  * Momentum：加速收敛，平滑震荡
  * 权重衰减（L2）：防止过拟合
* 损失函数：**交叉熵 + Label Smoothing**

  * Label Smoothing：避免过度拟合 One-Hot 标签

---

## 5. 学习率调度

采用了手动分段策略进行：

* **手动分段调整**：在固定 epoch 手动降低学习率

---

## 6. 最终结果

使用 `WideResNet-28-10`，配合 Cutout、AutoAugment、Label Smoothing、CosineAnnealing 学习率调度等策略，在 CIFAR-10 测试集上取得如下结果，见 `results` 目录：

* **Loss**: 0.5898
* **Accuracy**: **96.64%**

---

## 7. 运行方式

### （1）安装依赖

- Python 版本 >= 3.11
- 使用 pip 安装项目所需的依赖：

```bash
pip install -r requirements.txt
```

### （2）准备数据集

代码会自动下载 CIFAR-10 数据集并缓存在本地（默认在 `dataset` 目录下）。如果网络不好可以手动下载：https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz。

### （3）运行训练主程序

```bash
python train.py
```

你也可以在 `config.py` 中修改相关训练参数（如 `epoch`, `batch_size`, `learning_rate` 等）。

### （4）TensorBoard 可视化（可选）

```bash
tensorboard --logdir=runs
```

---

## 参考

* ResNet 原始论文：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
* WideResNet 原始论文：[Wide Residual Networks (Zagoruyko & Komodakis)](https://arxiv.org/abs/1605.07146)
* Cutout: [https://github.com/uoguelph-mlrg/Cutout](https://github.com/uoguelph-mlrg/Cutout)
* AutoAugment: [https://arxiv.org/abs/1805.09501](https://arxiv.org/abs/1805.09501)

