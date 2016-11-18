import tensorflow as tf

class BasicModel:
    """
    Basic model on deep neural network
    """
    def __init__(self, input=None, labels=None, dims_in=None, dims_out=None):
        """
        :param input: input placeholder
        :param labels: label placeholder
        :param dims_in: input dimensions
        :param dims_out: output dimensions
        :return:
        """
        self.input = None
        self.labels = None
        self.dims_in = None
        self.dims_out = None
        self.model = None
        self.loss = None
        self.train = None
        self.evaluation = None

    def model(self):
        return self.model

    def loss(self):
        return self.loss

    def train(self):
        return self.train

    def evaluation(self):
        return self.evaluation
