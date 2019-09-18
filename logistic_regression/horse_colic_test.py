"""
    预测病马死亡率
"""
from numpy import array

from logistic_regression.logistic_reg import random_grad_ascent, classify


def colic_test():
    lines = open('horseColicTraining.txt').readlines()
    data_set = []
    labels = []
    for line in lines:
        nums = line.strip().split('\t')
        data = []
        for num in nums[:-1]:
            data.append(float(num))
        data_set.append(data)
        labels.append(float(nums[-1]))
    data1_result, data2_result, data3_result, weights = random_grad_ascent(data_set, labels, 500)

    # 检测训练效果
    test_lines = open('horseColicTest.txt').readlines()
    error = 0
    for line in test_lines:
        nums = line.strip().split('\t')
        data = []
        for num in nums[:-1]:
            data.append(float(num))
        if int(classify(array(data), weights)) != int(nums[-1]):
            error += 1
    error_rate = float(error) / len(test_lines)
    print('error rate is ' + str(error_rate))
    return error_rate


def multi_test():
    error_rate = 0
    for i in range(10):
        error_rate += colic_test()
    print('average error rate is %f' % (error_rate / 10))


if __name__ == '__main__':
    multi_test()
