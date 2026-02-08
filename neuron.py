import numpy as np
from typing import *
import utils
import math

class HiddenLayer:
    def __init__(self, input_dimension: int, output_dimension: int, activation: Literal['sigmoid','relu','tanh','none'], 
                 initialization: Literal['kaiming_normal','xavier_normal'],
                 optimizer: utils.Optimizer,
                 batch_size: int):
        #基本信息
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.batch_size = batch_size

        #神经元数量是output_dimension
        self.number_of_neurons = output_dimension

        #权重矩阵的大小是input_dimension x output_dimension
        #偏置向量的大小是output_dimension x 1
        #输入向量的大小是input_dimension x 1 -> 注意输入和输出向量大小不同!
        self.weight_shape = (input_dimension, output_dimension, )
        self.biases_shape = (output_dimension, 1, )
        self.input_shape = (input_dimension, batch_size, )

        #激活函数及其导函数
        activation_choices = {'sigmoid':utils.sigmoid_vector,
                              'relu':utils.relu_vector,
                              'tanh':utils.tanh_vector,
                              'none':lambda input_vector:input_vector}
        activation_derivative_choices = {'sigmoid':utils.sigmoid_derivative_vector,
                              'relu':utils.relu_derivative_vector,
                              'tanh':utils.tanh_derivative_vector,
                              'none':lambda input_vector, output_vector:np.ones_like(input_vector)}
        self.activation = activation_choices[activation]
        self.activation_derivative = activation_derivative_choices[activation]

        #初始化
        #初始化权重
        if initialization == 'kaiming_normal':
            self.weights = np.random.normal(0, math.sqrt(2 / input_dimension), self.weight_shape)
        elif initialization == 'xavier_normal':
            self.weights = np.random.normal(0, math.sqrt(2 / (input_dimension + output_dimension)), self.weight_shape)
            xavier_adjust = {'sigmoid':1,
                             'relu':math.sqrt(2),
                             'tanh':4}
            self.weights *= xavier_adjust[activation]
        #初始化偏置项
        self.biases = np.zeros(self.biases_shape)

        #状态标志位与状态存储
        #是否已传播与更新
        self.forwarded = False
        self.backwarded = False
        self.updated = False
        #前向传播状态
        self.input_vector = None
        self.output_vector = None
        self.unactivated_output_vector = None
        #反向传播状态
        #包含所有batch_size个梯度向量
        self.upstream_loss_gradient = None
        self.input_vector_loss_gradient = None
        #只包含平均梯度
        #矩阵
        self.weights_loss_gradient = None
        #梯度向量
        self.biases_loss_gradient = None

        #优化器
        self.optimizer = optimizer
        

    #前向传播
    def forward(self, input_vector: np.typing.NDArray):
        if input_vector.shape != self.input_shape:
            raise TypeError("输入向量大小有误！")
        
        x = input_vector
        W = self.weights
        b = self.biases
        f = self.activation

        z = W.T @ x + b

        a = f(z)

        self.input_vector = x
        self.output_vector = a
        self.unactivated_output_vector = z

        #更新标志
        self.forwarded = True

        return a
    
    #反向传播
    def backward(self, upstream_loss_gradient: np.typing.NDArray):
        #不考虑batch时，dL_da大小是(output_dim, )
        #考虑batch时，dL_da大小是(output_dim, batch_size)
        dL_da = upstream_loss_gradient
        da_dz = self.activation_derivative(self.input_vector, self.output_vector)
        x = self.input_vector
        W = self.weights

        #传播过激活函数
        dL_dz = da_dz * dL_da

        #传播过线性函数
        #对偏置项梯度
        #dL_db是(output_dim, batch_size)大小的矩阵
        #这里对其在batch_size方向（行方向）取平均值得到平均梯度

        #dL_db = dL_dz
        dL_db = np.sum(dL_dz, axis=1) / self.batch_size
        #dL_dW = dL_dz @ x.T，如果采用传统定义

        #对权重梯度
        #dL/dW是一个和W同型的矩阵（如果不考虑batch） -> 这是一个秩1矩阵uv^T，是向量的外积

        #然而这里我们的batch实际上不为1，而x和dL_dz(与z同型)的尺寸都与batch大小有关，这意味着这是两个矩阵相乘，而不是向量外积
        #但是这里注意：x的实际大小是(input_dim, batch_size)，dL_dz实际大小是(output_dim, batch_size)，dL_dz则是(batch_size, output_dim)
        #因此**由矩阵乘法定义，这里的结果应该是batch中每一个秩1矩阵uv^T的和**
        #因此，结果**必须**除以batch_size

        #dL_dW = x @ dL_dz.T
        dL_dW = (x @ dL_dz.T) / self.batch_size

        #计算dL_dx，用于下一级的反向传播
        #求导有求转置的神奇功能
        #不考虑batch时，W是(input_dim, output_dim)，dL_dz是(output_dim, )，结果是向量(input_dim, )
        #考虑batch时，W仍是(input_dim, output_dim)，dL_dz变为(output_dim, batch_size)，结果变成了一个矩阵(input_dim, batch_size)，它的每一列都是一个样本自己的dL_dx
        #但是，参照上面dL_da和dL_dz的尺寸要求，这里dL_dx必须原样保存下来，传给前面一级
        #因此有无batch的实现相同
        dL_dx = W @ dL_dz

        self.upstream_loss_gradient = upstream_loss_gradient
        self.weights_loss_gradient = dL_dW
        self.biases_loss_gradient = dL_db
        self.input_vector_loss_gradient = dL_dx

        #更新标志
        self.backwarded = True

        #返回dL_dx，它是整个batch每个样本各自的dL_dx梯度向量捆在一起，尺寸(input_dim, batch_size)，为便于下一级反向传播
        return dL_dx
    
    #更新权重与偏置项
    def update(self):
        self.weights = self.optimizer.get_weights_delta(self.weights, self.weights_loss_gradient)
        self.biases = self.optimizer.get_biases_delta(self.weights, self.weights_loss_gradient)

        self.forwarded = False
        self.backwarded = False