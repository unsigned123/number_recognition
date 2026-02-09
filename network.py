from neuron import *

class Network:
    def __init__(self, batch_size: int, layers: List[Layer | LossLayer]):
        self.batch_size = batch_size
        self.layers = layers

        self.loaded = False
        self.forwarded = False
        self.backwarded = False

        self.sample_batch = None
        self.processing_batch = None
        self.one_hot_batch = None

        self.output_batch = None

        self.loss = None

    def load(self, sample_batch: np.typing.NDArray, one_hot_batch: np.typing.NDArray):
        self.sample_batch  = sample_batch
        self.processing_batch = sample_batch
        self.one_hot_batch = one_hot_batch

        self.loaded = True

    def forward(self):
        for layer in self.layers:
            if isinstance(layer, Layer):
                self.processing_batch = layer.forward(self.processing_batch)
            elif isinstance(layer, LossLayer):
                self.loss = layer.forward(self.processing_batch, self.sample_batch)
                break

        self.output_batch = self.processing_batch
        self.forwarded = True

        return self.output_batch
        
    def backward(self):
        for layer in reversed(self.layers):
            if isinstance(layer, Layer):
                self.processing_batch = layer.backward(self.processing_batch)
            elif isinstance(layer, LossLayer):
                self.processing_batch = layer.backward()
        
        self.backwarded = True

    def update(self):
        for layer in self.layers:
            layer.update()

        self.loaded = False
        self.forwarded = False
        self.backwarded = False
