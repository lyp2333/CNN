# -*- coding: UTF-8 -*-
from cnn import *
from tqdm import tqdm


def init_test():
    a = np.array(
        [[[0, 1, 1, 0, 2],
          [2, 2, 2, 2, 1],
          [1, 0, 0, 2, 0],
          [0, 1, 1, 0, 0],
          [1, 2, 0, 0, 2]],
         [[1, 0, 2, 2, 0],
          [0, 0, 0, 2, 0],
          [1, 2, 1, 2, 1],
          [1, 0, 0, 0, 0],
          [1, 2, 1, 1, 1]],
         [[2, 1, 2, 0, 0],
          [1, 0, 0, 1, 0],
          [0, 2, 1, 0, 1],
          [0, 1, 2, 2, 2],
          [2, 1, 0, 0, 1]]]) / 2
    test_a = np.array([[2, 1, 2, 0, 0],
                       [1, 0, 0, 1, 0],
                       [0, 2, 1, 0, 1],
                       [0, 1, 2, 2, 2],
                       [2, 1, 0, 0, 1]])
    b = np.array(
        [[[0, 1, 1],
          [2, 2, 2],
          [1, 0, 0]],
         [[1, 0, 2],
          [0, 0, 0],
          [1, 2, 1]]])
    cl = ConvLayer(5, 5, 1, 3, 3, 1, 0, 1, IdentityActivator(), 0.01)
    # print(cl.filters[0].weights)
    # cl.filters[0].weights = np.random.rand(1, 3, 3) * 2 - 1
    # cl.filters[0].bias = 1
    # cl.filters[1].weights = np.random.rand(1, 3, 3) * 2 - 1
    return test_a, b, cl


def label_data(input):
    cl = ConvLayer(5, 5, 3, 3, 3, 2, 1, 2, IdentityActivator(), 0.001)
    cl.filters[0].weights = np.array(
        [[[-1, 0, -1],
          [-2, 0, -2],
          [-1, 0, -1]],
         [[-1, 0, -1],
          [-2, 0, -2],
          [-1, 0, -1]],
         [[-1, 0, -1],
          [-2, 0, -2],
          [-1, 0, -1]]], dtype=np.float64)
    cl.filters[0].bias = 1
    cl.filters[1].weights = np.array(
        [[[-1, 0, -1],
          [-2, 0, -2],
          [-1, 0, -1]],
         [[-1, 0, -1],
          [-2, 0, -2],
          [-1, 0, -1]],
         [[-1, 0, -1],
          [-2, 0, -2],
          [-1, 0, -1]]], dtype=np.float64)
    cl.forward(input)
    return cl.output_array


def label_data1(input):
    cl = ConvLayer(5, 5, 1, 3, 3, 1, 0, 1, IdentityActivator(), 0.0001)
    cl.filters[0].weights = np.array(
        [[[1, 0, 2],
          [2, 0, 4],
          [1, 0, 2]]], dtype=np.float64)
    cl.filters[0].bias = 1
    cl.forward(input)
    return cl.output_array


def forward_test():
    a, _, cl = init_test()
    cl.forward(a)
    print(cl.output_array)
def loss_calculate(output, label):
    loss_1 = output - label
    loss_all = (loss_1 * loss_1) / 2
    return loss_all.sum()


if __name__ == '__main__':
    input, _, model = init_test()  # 获得数据并初始化权值
    label = label_data1(input)
    epoch = 12000
    for i in tqdm(range(epoch)):
        model.forward(input)
        output = model.output_array
        loss = loss_calculate(output, label)
        sensitivity_array = np.ones(model.output_array.shape, dtype=np.float64) * (output - label)
        model.backward(input, sensitivity_array, IdentityActivator())
        model.update()
        print(f'第{i}次迭代loss为{loss}')
    print(model.filters[0])
