import numpy as np

from cnn import *
from cnn_test import init_test
from tqdm import tqdm
from MaxPooling import *
from fc import *


class Model(object):
    def __init__(self, model1, pool1, model2, pool2, fc):
        self.model_1 = model1
        self.pool_1 = pool1
        self.model_2 = model2
        self.pool_2 = pool2
        self.fc = fc
        self.flatten = 0
        self.output1 = 0
        self.output2 = 0
        self.output3 = 0
        self.output4 = 0
        self.output = 0
        self.input = 0

    def forward(self, input):
        self.input = input
        self.model_1.forward(input)
        self.output1 = self.model_1.output_array
        self.pool_1.forward(self.output1)
        self.output2 = self.pool_1.output_array
        self.model_2.forward(self.output2)
        self.output3 = self.model_2.output_array
        self.pool_2.forward(self.output3)
        self.output4 = self.pool_2.output_array
        self.flatten = np.reshape(self.output4, (-1)).tolist()
        self.output = self.fc.forward(self.flatten)
        return self.output

    # 输入损失的偏导
    def backward(self, y_pre, y_real):
        self.fc.backpropagation(y_pre, y_real)
        sensitivity_array = np.dot(self.fc.weights[0].T, self.fc.delta[0])
        sensitivity_array.reshape(self.output4.shape)
        self.pool_2.backward(self.output3, sensitivity_array)
        model_2_sensitivity = self.pool_2.delta_array
        self.model_2.backward(self.output2, model_2_sensitivity, IdentityActivator)
        pool_1_sensitivity = self.model_2.delta_array
        self.pool_1.backward(self.output1, pool_1_sensitivity)
        model_1_sensitivity = self.pool_1.delta_array
        self.model_1.backward(self.input, model_1_sensitivity, IdentityActivator)

    def update(self, lr):
        conv_layers = [self.model_1, self.model_2]
        for i in range(len(conv_layers)):
            for filter in conv_layers[i].filters:
                filter.update(lr)


def my_cnn():
    model1 = ConvLayer(5, 5, 3, 3, 3, 3, 2, 1, IdentityActivator(), 0.001)
    pool1 = MaxPoolingLayer(5, 5, 3, 2, 2, 1)
    model2 = ConvLayer(4, 4, 3, 3, 3, 3, 2, 1, IdentityActivator(), 0.001)
    pool2 = MaxPoolingLayer(4, 4, 3, 2, 2, 1)
    fc = MyNet('relu', 27, 48, 48, 10)
    my_cnn = Model(model1, pool1, model2, pool2, fc)  # output shape:3,3,3
    return my_cnn


def loss_calculate(output, label_data):
    d = np.array(output) - label_data
    deri = output - label_data
    model_loss = d * d
    return model_loss, deri


def label_data(input):
    model = ConvLayer(5, 5, 3, 3, 3, 3, 0, 1, IdentityActivator(), 0.001)
    model.forward(input)
    return model.output_array


def label_data1():
    label = np.zeros((10, 1))
    label[0, :] = 1
    print()
    return label


if __name__ == '__main__':
    # test_code
    in_, _, _ = init_test()
    input = in_ / 255
    label = label_data1()
    my_net = my_cnn()
    epoch = 1000
    lr = 0.001

    for i in tqdm(range(epoch)):
        output = my_net.forward(input)
        loss, _ = loss_calculate(output, label)
        my_net.backward(output, label)
        my_net.update(lr)
        print(f'第{i}次迭代loss为{loss.sum()}')
