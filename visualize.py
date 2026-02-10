import numpy as np
from network import *
import matplotlib.pyplot as plt

import pickle
import random

model: Network = None
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

for i, layer in enumerate(model.layers):
    print(f'第{i + 1}层:{layer.__class__}')

index = int(input('输入需要查看的层序号:')) - 1

layer = model.layers[index]
if isinstance(layer, ConvolutionLayer):
    input_channel = layer.kernel_shape[0]
    output_channel = layer.kernel_shape[1]

    grid_rows, grid_cols = input_channel, output_channel
    img_height = img_width = layer.kernel_size

    matrix = layer.kernel

    if matrix.max() > 1 or matrix.min() < 0:
        matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())

    # 创建一个大的画布
    fig, axes = plt.subplots(grid_rows, grid_cols, 
                            figsize=(grid_cols*2, grid_rows*2),
                            squeeze=False)

    # 遍历每个位置并显示灰度图
    for i in range(grid_rows):
        for j in range(grid_cols):
            axes[i, j].imshow(matrix[i, j], cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
            axes[i, j].set_title(f'({i},{j})')

    plt.tight_layout()
    plt.show()

elif isinstance(layer, HiddenLayer):
    input_dimension = layer.weight_shape[0]
    output_dimension = layer.weight_shape[1]

    matrix = layer.weights

    if matrix.max() > 1 or matrix.min() < 0:
        matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='gray', vmin=0, vmax=1)  # vmin和vmax确保正确映射
    plt.colorbar()  # 显示颜色条
    plt.axis('off')  # 可选：关闭坐标轴
    plt.show()