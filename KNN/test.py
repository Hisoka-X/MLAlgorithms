"""
    测试数据
"""
from numpy import shape

from KNN.kNN import creat_dataset, norm_dataset, classify


def dating_class_test():
    # 取百分之十的数据进行测试
    ho_ratio = .1
    data_set, labels = creat_dataset('datingTestSet.txt')
    data_set, ranges, min_data = norm_dataset(data_set)
    data_size = shape(data_set)[0]
    test_size = int(ho_ratio * data_size)
    error_size = 0
    for i in range(test_size):
        result = classify(data_set[i, :], data_set[test_size:data_size, :], labels[test_size:data_size], 5)
        if result != labels[i]:
            error_size += 1
            print('classify return ' + result + ' but should be ' + labels[i])
    print('the total error rate is ' + str(error_size / test_size))


if __name__ == '__main__':
    dating_class_test()
