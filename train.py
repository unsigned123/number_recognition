from network import *
import pickle
from neuron import *
import gzip
import struct

batch_size = 16
learning_rate = 0.001

def new_network():
    # 创建优化器
    optimizer = utils.SGD(learning_rate) 

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
    layers = [conv1, pool1, conv2, pool2, flatten, fc1, dropout, fc2, softmax, loss_layer]

    model = Network(batch_size, layers)

    return model

def load_dataset(filename: str):
    data = None
    with gzip.open(filename, 'rb') as f:
        # 读取文件头
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]

        #读取图像
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(num_images, num_rows, num_cols)

    return data

def load_label(filename: str):
    with gzip.open(filename, 'rb') as f:
        # 读取文件头
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]
        
        # 读取所有标签数据
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
        
    return labels

model: Network = None
def train():

    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
    except:
        print('未检测到已训练模型，创建新模型')
        with open('model.pkl', 'wb') as f:
            model = new_network()
            pickle.dump(model, f)

    for each in model.layers:
        try:
            each.optimizer.learning_rate = learning_rate
            print('changed')
        except:
            pass
    model.layers[6].mode = 'training'
    images = load_dataset('mnist/train-images-idx3-ubyte.gz')
    labels = load_label('mnist/train-labels-idx1-ubyte.gz')

    images = images.reshape(-1, 1, 28, 28).transpose(1, 2, 3, 0) / 255

    one_hot_vectors = np.zeros((10, labels.shape[0]), dtype=int)
    one_hot_vectors[labels, np.arange(labels.shape[0])] = 1

    epoches = 10
    try:
        for epoch in range(epoches):
            for start_index in range(0, labels.shape[0] - batch_size, batch_size):
                batch_images = images[:, :, :, start_index:start_index + batch_size]
                batch_one_hot_vectors = one_hot_vectors[:, start_index:start_index + batch_size]

                model.load(batch_images, batch_one_hot_vectors)
                model.forward()
                model.backward()
                model.update()

                print(f'第{epoch + 1}轮的第{start_index // batch_size + 1}批，loss={model.loss}')
    except KeyboardInterrupt:
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)

        print('模型成功保存')
        exit(0)

    print(f'共{epoches}轮训练完毕!目前loss={model.loss}')

def reason():

    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
    except:
        print('未检测到已训练模型，创建新模型')
        with open('model.pkl', 'wb') as f:
            model = new_network()
            pickle.dump(model, f)
    model.layers[6].mode = 'reasoning'

    images = load_dataset('mnist/t10k-images-idx3-ubyte.gz')
    labels = load_label('mnist/t10k-labels-idx1-ubyte.gz')

    images = images.reshape(-1, 1, 28, 28).transpose(1, 2, 3, 0) / 255

    one_hot_vectors = np.zeros((10, labels.shape[0]), dtype=int)
    one_hot_vectors[labels, np.arange(labels.shape[0])] = 1

    batch_size = 16

    total_correct = 0
    total_checked = 0
    
    for start_index in range(0, labels.shape[0] - batch_size, batch_size):
        batch_images = images[:, :, :, start_index:start_index + batch_size]
        batch_one_hot_vectors = one_hot_vectors[:, start_index:start_index + batch_size]
        batch_labels = labels[start_index:start_index + batch_size]

        model.load(batch_images, batch_one_hot_vectors)

        result = model.forward()

        total_checked += batch_size
        total_correct += (result.argmax(axis=0) == batch_labels).sum()

        print(f'推理准确率{total_correct / total_checked}')


reason()
