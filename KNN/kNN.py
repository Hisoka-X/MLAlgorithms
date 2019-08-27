"""
    k-近邻算法
"""
import operator

from numpy import array, tile, ndarray, matrix, zeros

from common.util import show_scatter


def creat_dataset(file_name):
    # group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # labels = ['A', 'A', 'B', 'B']
    fr = open(file_name)
    lines = fr.readlines()
    number_of_lines = len(lines)
    group = zeros((number_of_lines, 3))
    index = 0
    labels = []
    for line in lines:
        keys = line.strip().split('\t')
        group[index, :] = keys[0:3]
        labels.append(keys[-1])
        # 如果需要显示散点图 就需要对str进行int化
        # labels.append(int(keys[-1]))
        index += 1
    return group, labels


def classify(inx: list, dataset: ndarray, labels, k):
    """
        通过输入的向量 返回和其最相近的标签
    :param inx: 需要分类的向量
    :param dataset: 训练数据集
    :param labels: 数据集对应的标签
    :param k: 邻近数目 用于设置邻近数量 然后返回其中占比最大的标签作为结果
    :return: 输入向量对应的标签
    """
    # 返回数据集行数
    dataset_size = dataset.shape[0]
    # 采用欧氏距离公式计算向量的距离
    # 将inx按行复制dataset_size次 用于提供给dataset中的每组数据进行对位相减
    diff_mat = tile(inx, (dataset_size, 1)) - dataset
    sq_diff_mat: ndarray = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances: matrix = sq_distances ** .5
    # 对数据结果进行排序 返回对位的位次矩阵 返回从小到大的顺序
    sort_dist_indices = distances.argsort()
    class_count = {}
    # 找出最小的k个结果对应的标签
    for i in range(k):
        # 统计标签出现次数次数
        label = labels[sort_dist_indices[i]]
        class_count[label] = class_count.get(label, 0) + 1
    # 排序字典 返回value最大的一个标签
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def norm_dataset(dataset):
    """
        长度归一化 减少属性本身数据大小特征的差异
    :param dataset:
    :return:
    """
    min_data = dataset.min(0)
    max_data = dataset.max(0)
    ranges = max_data - min_data
    # 数据集行数量
    m = dataset.shape[0]
    # 归一化公式：new_value = (old_value-min)/(max-min)
    # 生成dataset相同大小的矩阵
    norm_data = dataset - tile(min_data, (m, 1))
    norm_data = norm_data / tile(ranges, (m, 1))

    return norm_data, ranges, min_data


def run():
    # 测试数据
    test = [1, 0.8, 1]
    # 加载数据
    group, labels = creat_dataset('datingTestSet2.txt')
    # 数据归一化
    group, ranges, min_data = norm_dataset(group)
    # 展示数据散点图
    show_scatter(group[:, 0], group[:, 1], labels)
    # 对测试数据进行归一化
    test = (array(test) - min_data) / ranges
    # 获取输出向量对应的标签
    print(str(classify(test, group, labels, 3)))


if __name__ == '__main__':
    run()
