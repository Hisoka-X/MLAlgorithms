"""
    利用支持向量机实现手写数字的分类
"""
from os import listdir

from numpy import zeros, mat, nonzero, shape, multiply, sign, array

from support_vector_machine.smo import smo_complete, kernel_trans
from support_vector_machine.smo_simple import plot_scatter


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


def load_data_set(dir_name, test_num=1):
    """
        加载数据集 因为svm是一个二分类算法 所以需要提供一个识别的数字 然后结果判断是否为这个数字
    """
    files = listdir(dir_name)
    file_size = len(files)
    img_data = zeros((file_size, 1024))
    img_label = []
    for i in range(len(files)):
        img_data[i, :] = img_2_vector(dir_name + '/' + files[i])
        if test_num == int(files[i].split('_')[0]):
            img_label.append(1)
        else:
            img_label.append(-1)
    return img_data, img_label


def class_num_use_svm(reach=0.1, test_num=1):
    data_matrix, label_matrix = load_data_set('trainingDigits', test_num)
    data_matrix = mat(data_matrix)
    label_matrix = mat(label_matrix).T
    kernel_type = ('rbf', reach)
    b, alphas = smo_complete(data_matrix, label_matrix, 200, 0.0001, max_iter=10000, kernel_type=kernel_type)
    # alpha大于0的都是支持向量
    support_vector_index = nonzero(alphas.A > 0)[0]
    support_matrix = data_matrix[support_vector_index]
    support_label = label_matrix[support_vector_index]
    print('there are %d support vectors' % len(support_vector_index))

    # 遍历训练向量 计算出正确率
    m = shape(data_matrix)[0]
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(support_matrix, data_matrix[i, :], kernel_type)
        predict = kernel_eval.T * multiply(support_label, alphas[support_vector_index]) + b
        if sign(predict[0, 0]) != sign(label_matrix[i]):
            error_count += 1
    print('the training error rate is %f' % (float(error_count) / m))

    # 遍历测试向量 计算正确率
    data_matrix, label_matrix = load_data_set('testDigits', test_num)
    data_matrix = mat(data_matrix)
    m = shape(data_matrix)[0]
    label_matrix = mat(label_matrix).T
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(support_matrix, data_matrix[i, :], kernel_type)
        predict = kernel_eval.T * multiply(support_label, alphas[support_vector_index]) + b
        if sign(predict) != sign(label_matrix[i]):
            error_count += 1
    print('the test error rate is %f' % (float(error_count) / m))


def run():
    class_num_use_svm(reach=1)


if __name__ == '__main__':
    run()
