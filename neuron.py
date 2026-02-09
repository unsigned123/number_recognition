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
        
        self.updated = False
        
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
        #不考虑batch时，dL_da大小是(output_dim, 1)
        #考虑batch时，dL_da大小是(output_dim, batch_size)
        #da_dz是(output_dim, batch_size)
        dL_da = upstream_loss_gradient
        #注意第一个不是input_vector
        da_dz = self.activation_derivative(self.unactivated_output_vector, self.output_vector)
        x = self.input_vector
        W = self.weights

        #传播过激活函数
        #注意这里是标量乘法，挨个相乘，并不是矩阵乘法；如果一定要视为矩阵乘法，这个da/dz是对角阵，所以可以交换
        #所以使用“*”
        dL_dz = dL_da * da_dz

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
        self.updated = True

        self.weights = self.optimizer.get_new_weights(self.weights, self.weights_loss_gradient)
        self.biases = self.optimizer.get_new_biases(self.biases, self.biases_loss_gradient)

        self.forwarded = False
        self.backwarded = False

class ConvolutionLayer:
    def __init__(self, input_channel: int, output_channel: int,
                 input_size: int, output_size: int, 
                 kernel_size :int, stride: int, need_padding: bool,
                 padding_size :int,
                 activation: Literal['sigmoid','relu','tanh','none'], 
                 initialization: Literal['kaiming_normal','xavier_normal'],
                 optimizer: utils.Optimizer,
                 batch_size: int):
        #基本信息
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.output_size = output_size

        self.stride = stride
        self.need_padding = need_padding
        self.padding_size = padding_size

        #卷积核的形状是(input_channel, output_channel, kernel_size, kernel_size)
        #偏置的形状是(output_channel, 1)
        #输入的形状是(input_channel, input_size, input_size, batch_size)
        self.kernel_shape = (input_channel, output_channel, kernel_size, kernel_size)
        self.biases_shape = (output_channel, 1)
        self.input_shape = (input_channel, input_size, input_size, batch_size)

        #激活函数及其导函数
        activation_choices = {'sigmoid':utils.sigmoid_vector,
                              'relu':utils.relu_vector,
                              'tanh':utils.tanh_vector,
                              'none':lambda input_tensor:input_tensor}
        activation_derivative_choices = {'sigmoid':utils.sigmoid_derivative_vector,
                              'relu':utils.relu_derivative_vector,
                              'tanh':utils.tanh_derivative_vector,
                              'none':lambda input_tensor, output_tensor:np.ones_like(input_tensor)}
        self.activation = activation_choices[activation]
        self.activation_derivative = activation_derivative_choices[activation]

        #初始化
        #初始化卷积核
        if initialization == 'kaiming_normal':
            self.kernel = np.random.normal(0, math.sqrt(2 / input_channel), self.kernel_shape)
        elif initialization == 'xavier_normal':
            self.kernel = np.random.normal(0, math.sqrt(2 / (input_channel + output_channel)), self.kernel_shape)
            xavier_adjust = {'sigmoid':1,
                             'relu':math.sqrt(2),
                             'tanh':4}
            self.kernel *= xavier_adjust[activation]
        #初始化偏置项
        self.biases = np.zeros(self.biases_shape)

        #状态标志位与状态存储
        #是否已传播与更新
        self.forwarded = False
        self.backwarded = False
        self.updated = False
        #前向传播状态
        self.input_tensor = None
        self.padded_input_tensor = None
        self.output_tensor = None
        self.unactivated_output_tensor = None
        #加速卷积计算
        self.convolution_windows = None
        #反向传播状态
        #包含所有batch_size个output_size维梯度向量
        self.upstream_loss_gradient = None
        self.input_tensor_loss_gradient = None
        #只包含平均梯度
        #张量
        self.kernel_loss_gradient = None
        #梯度张量
        self.biases_loss_gradient = None

        #优化器
        self.optimizer = optimizer

    #前向传播
    def forward(self, input_tensor: np.typing.NDArray):
        padded_input_tensor = None
        if self.need_padding:
            padded_input_tensor = np.pad(input_tensor, ((0, 0), (self.padding_size, self.padding_size), (self.padding_size, self.padding_size)) ,mode='constant', constant_values=0)
        
        self.input_tensor = input_tensor

        self.updated = False

        #input_tensord的形状为(input_channel, padded_input_size, padded_input_size, batch_size)，因此我们选择轴1, 2
        windows = np.lib.stride_tricks.sliding_window_view(padded_input_tensor
                                                           if self.need_padding
                                                           else input_tensor, (self.kernel_size, self.kernel_size), axis=(1, 2))
        #进行stride切片，显然我们只想要在轴1, 2上以self.stride为步长进行卷积；windows多了2个维度加在最后
        #现在，windows的轴1和轴2被修剪为output_size，output_size（即窗口个数），后面加了两个窗口维度，它们的大小为(kernel_size, kernel_size)
        windows = windows[:, ::self.stride, ::self.stride, :, :]

        self.convolution_windows = windows

        #计算卷积
        #windows的形状为(input_channel, output_size, output_size, batch_size, kernel_size, kernel_size)，kernel的形状为(input_channel, output_channel, kernel_size, kernel_size)
        # i: input_channel
        # h: output_height
        # w: output_width
        # b: batch_size
        # k: kernel_height
        # l: kernel_width
        # j: output_channel
        #输出形状为(output_channel, output_size, output_size, batch_size)
        convolution = np.einsum('ihwbkl,ijkl->jhwb', windows, self.kernel)

        self.unactivated_output_tensor = convolution + self.biases
        self.output_tensor = self.activation(self.unactivated_output_tensor)
        self.padded_input_tensor = padded_input_tensor

        self.forwarded = True

        return self.output_tensor
    
    #反向传播
    def backward(self, upstream_loss_gradient: np.typing.NDArray):
        #先计算对卷积核的梯度dL/dK
        
        #首先，上游的梯度是对于激活后输出Z的梯度dL/dZ，应当先把它转化为dL/dY，Y是激活前输出
        #dL/dZ,dL/dY的形状均为(output_channel, output_size, output_size, batch_size)，与Z,Y同形
        dL_dZ = upstream_loss_gradient
        dZ_dY = self.activation_derivative(self.unactivated_output_tensor, self.output_tensor)

        #传播过激活函数
        #dL/dY = dL/dZ * dZ/dY -> 注意链式法则右边顺序！
        #由于激活函数是逐个操作的，这里依旧可以把dZ/dY视作对角阵，两个张量的元素依旧可以对应相乘（而不是矩阵乘法）
        #dL/dY与dL/dZ同形
        dL_dY = dL_dZ * dZ_dY

        #传播过卷积层   
        # 直接方法计算dL/dX：对于每个输出位置的梯度，将其分配到对应的输入区域
        dL_dX = np.zeros(self.input_shape)
        
        if self.need_padding:
            padded_input_shape = (self.input_channel, 
                                self.input_size + 2*self.padding_size,
                                self.input_size + 2*self.padding_size,
                                self.batch_size)
            dL_dX_padded = np.zeros(padded_input_shape)
        else:
            dL_dX_padded = dL_dX
        
        # 遍历所有输出位置
        for h_out in range(self.output_size):
            for w_out in range(self.output_size):
                # 计算对应的输入区域
                h_start = h_out * self.stride
                w_start = w_out * self.stride
                h_end = h_start + self.kernel_size
                w_end = w_start + self.kernel_size
                
                # 获取当前输出位置对输入的梯度贡献
                # dL_dY[:, h_out, w_out, :] 形状: (output_channel, batch_size)
                # self.kernel 形状: (input_channel, output_channel, kernel_size, kernel_size)
                
                # 计算梯度贡献
                grad_contrib = np.einsum('jb,ijkl->iklb', 
                                    dL_dY[:, h_out, w_out, :],
                                    self.kernel)
                # 形状: (input_channel, kernel_size, kernel_size, batch_size)
                
                # 将贡献加到对应的输入区域
                dL_dX_padded[:, h_start:h_end, w_start:w_end, :] += grad_contrib
        
        # 如果前向传播有padding，裁剪掉padding部分
        if self.need_padding:
            dL_dX = dL_dX_padded[:, self.padding_size:-self.padding_size,
                            self.padding_size:-self.padding_size, :] / self.batch_size
        else:
            dL_dX = dL_dX_padded / self.batch_size
        dL_dK = np.einsum('abcd,ebcdfg->eafg', dL_dY, self.convolution_windows) / self.batch_size

        #最后计算dL_dB
        #dL/dY形状为(output_channel, output_size, output_size, batch_size)
        #我们需要对
        dL_dB = np.sum(dL_dY, axis=(1,2,3)) / self.batch_size

        self.biases_loss_gradient = dL_dB
        self.kernel_loss_gradient = dL_dK
        self.input_tensor_loss_gradient = dL_dX

        self.backwarded = True

        return dL_dX
    
    def update(self):
        self.updated = True

        self.kernel = self.optimizer.get_new_weights(self.kernel, self.kernel_loss_gradient)
        self.biases = self.optimizer.get_new_biases(self.biases, self.biases_loss_gradient)

        self.forwarded = False
        self.backwarded = False

