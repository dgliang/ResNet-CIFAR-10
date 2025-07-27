from ResNet import ResNet, BasicBlock


def WideResNet_28_10(num_classes=10):
    widen_factor = 10
    depth = 28
    assert (depth - 4) % 6 == 0, "Depth should be 6n + 4 for WRN"
    n = (depth - 4) // 6
    layers = [n, n, n] # 3 stages, each with n blocks

    return ResNet(
        block=BasicBlock,
        layers=layers,
        num_classes=num_classes,
        width_per_group=16 * widen_factor,
        # Important: CIFAR-10 don't need stride dilation
        replace_stride_with_dilation=[False, False, False]
    )
