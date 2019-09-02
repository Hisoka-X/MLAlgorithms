"""
    测试手写数字
"""
from os import listdir

from numpy import zeros

from KNN.kNN import classify


def img_2_vector(file_name):
    """
        将一个文件表示的数据存入向量中
    """
    fs = open(file_name)
    data = zeros((1, 1024))
    for i in range(32):
        line = fs.readline()
        for y in range(32):
            data[0, i * 32 + y] = int(line[y])
    return data


def test_for_number():
    """
        验证手写数字正确性
    """
    # 加载文件夹下所有文件
    files = listdir('trainingDigits')
    size = len(files)
    data_set = zeros((size, 1024))
    labels = []
    for i in range(size):
        file_name = files[i]
        num = file_name.split('_')[0]
        labels.append(int(num))
        data = img_2_vector('trainingDigits/' + file_name)
        data_set[i, :] = data
    test_files = listdir('testDigits')
    test_size = len(test_files)
    error_size = 0
    for i in range(test_size):
        file_name = test_files[i]
        num = int(file_name.split('_')[0])
        test_data = img_2_vector('testDigits/' + file_name)
        result = classify(list(test_data), data_set, labels, 3)
        if num != result:
            error_size += 1
            print('need return number is %d,but %d' % (num, result))
    print('error rate is %f' % (error_size / test_size))


if __name__ == '__main__':
    test_for_number()
