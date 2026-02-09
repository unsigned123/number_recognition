from network import *

if __name__ == '__main__':
    batch_size = 16
    learning_rate = 0.01

    model = Network(batch_size)

    # 创建优化器
    optimizer = utils.SGDOptimizer(learning_rate) 

    # 1. 卷积层1: 16个5×5滤波器，padding=2
    conv1 = ConvolutionLayer(
        input_channel=1,
        output_channel=16,
        input_size=28,
        output_size=28,  # 因为padding=2，尺寸不变
        kernel_size=5,
        stride=1,
        need_padding=True,
        padding_size=2,
        activation='relu',
        initialization='kaiming_normal',
        optimizer=optimizer,
        batch_size=batch_size
    )

    # 2. 最大池化2×2
    pool1 = PoolingLayer(
        channel=16,
        input_size=28,
        window_size=2,
        rule='maximum',
        batch_size=batch_size
    )

    # 3. 卷积层2: 32个3×3滤波器，padding=1
    conv2 = ConvolutionLayer(
        input_channel=16,
        output_channel=32,
        input_size=14,  # 池化后尺寸减半
        output_size=14,  # 因为padding=1，尺寸不变
        kernel_size=3,
        stride=1,
        need_padding=True,
        padding_size=1,
        activation='relu',
        initialization='kaiming_normal',
        optimizer=optimizer,
        batch_size=batch_size
    )

    # 4. 最大池化2×2
    pool2 = PoolingLayer(
        channel=32,
        input_size=14,
        window_size=2,
        rule='maximum',
        batch_size=batch_size
    )

    # 5. 展平层
    flatten = FlattenLayer(
        input_dimension=32,
        input_size=7,  # 第二次池化后：14/2 = 7
        batch_size=batch_size
    )

    # 6. 全连接层1: 128个神经元 + ReLU
    fc1 = HiddenLayer(
        input_dimension=32*7*7,  # 1568
        output_dimension=128,
        activation='relu',
        initialization='kaiming_normal',
        optimizer=optimizer,
        batch_size=batch_size
    )

    # 7. Dropout(0.5)
    dropout = DropoutLayer(dropout_probability=0.5)

    # 8. 全连接层2: 10个神经元 (无激活函数，因为后面接Softmax)
    fc2 = HiddenLayer(
        input_dimension=128,
        output_dimension=10,
        activation='none',  # 无激活函数
        initialization='kaiming_normal',
        optimizer=optimizer,
        batch_size=batch_size
    )

    # 9. Softmax层
    softmax = SoftmaxLayer(
        input_dimension=10,
        batch_size=batch_size
    )

    # 10. 损失层
    loss_layer = LossLayer(
        input_dimension=10,
        batch_size=batch_size,
        loss_type='cross_entropy',
        reduction='average'
    )

    # 组装模型
    model.layers = [conv1, pool1, conv2, pool2, flatten, fc1, dropout, fc2, softmax]