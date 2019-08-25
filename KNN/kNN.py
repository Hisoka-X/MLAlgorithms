import operator

from numpy import array, tile, ndarray, matrix

from Common.Util import show_scatter


def creat_dataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
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


def run():
    # 加载数据
    group, labels = creat_dataset()
    # 展示数据散点图
    show_scatter(group[:, 0], group[:, 1])
    # 获取输出向量对应的标签
    print(str(classify([1, 0.8], group, labels, 3)))


if __name__ == '__main__':
    run()
