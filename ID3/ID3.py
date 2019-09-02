"""
    ID3生成决策树
"""
import operator

from math import log

from common.util import create_plot


def calc_shannon_ent(dataset):
    """
        计算数据集香农熵
    """
    size = len(dataset)
    label_map = {}
    for data in dataset:
        label = data[-1]
        if label not in label_map:
            label_map[label] = 0
        label_map[label] += 1
    shannon_ent = 0.0
    for label_size in label_map.values():
        prob = float(label_size) / size
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def split_dataset(dataset, axis, value):
    """
        抽取特征的符合值的数据集
        需要将消耗的特征从符合的数据集中移除
    :param dataset: 原数据集
    :param axis: 特征数组坐标
    :param value: 特征值
    :return: 抽取完成的数据集
    """
    ret_dataset = []
    for data in dataset:
        if data[axis] == value:
            curr_data = data[:axis]
            curr_data.extend(data[axis + 1:])
            ret_dataset.append(curr_data)
    return ret_dataset


def get_best_feature(dataset):
    """
        获取划分数据集的最好的特征 通过计算每个特征对应的信息增益得到
        信息增益=数据划分前的香农熵-数据划分后的香农熵比例之和
    """
    base_entropy = calc_shannon_ent(dataset)
    feature_num = len(dataset[0]) - 1
    best_feature = -1
    best_info_gain = 0.0
    for i in range(feature_num):
        value_set = set([data[i] for data in dataset])
        feature_entropy = 0.0
        for value in value_set:
            value_dataset = split_dataset(dataset, i, value)
            prob = len(value_set) / float(len(value_dataset))
            feature_entropy += prob * calc_shannon_ent(value_dataset)
        feature_info_gain = base_entropy - feature_entropy
        if feature_info_gain > best_info_gain:
            best_info_gain = feature_info_gain
            best_feature = i
    return best_feature


def create_tree(dataset, labels):
    """
        :param dataset 数据集
        :param labels 数据feature名称
        创建决策树
    """
    # 判断数据集的labels是否是单一 单一就返回对应的label
    data_label = [data[-1] for data in dataset]
    if len(set(data_label)) == 1:
        return data_label[0]
    # 判断数据集是否feature已经使用完 使用完就返回数量占比最大的label
    if len(dataset[0]) == 1:
        return get_most_label(data_label)
    # 选择信息增益最高的feature
    index = get_best_feature(dataset)
    best_label = labels[index]
    tree = {best_label: {}}
    del labels[index]
    value_list = [data[index] for data in dataset]
    value_set = set(value_list)
    for value in value_set:
        tree[best_label][value] = create_tree(split_dataset(dataset, index, value), labels[:])
    return tree


def create_dataset():
    """
        创建数据集和标签
    """
    # dataset = [[1, 1, 'yes'],
    #            [1, 1, 'yes'],
    #            [1, 0, 'no'],
    #            [0, 1, 'no'],
    #            [0, 1, 'no']
    #            ]
    # labels = ['no surfacing', 'flippers']
    fs = open('lenses.txt')
    lines = fs.readlines()
    dataset = [line.strip().split('\t') for line in lines]
    labels = ['age', 'prescript', 'astigmatic', 'tear rate']
    return dataset, labels


def get_most_label(labels):
    """
        获取数据集中占比最大的label
    """
    label_map = {}
    for label in labels:
        if label not in label_map:
            label_map[label] = 0
        label_map[label] += 1
    sorted_labels = sorted(label_map, key=operator.itemgetter(1), reverse=True)
    return sorted_labels[0][0]


def get_test_label(tree, labels, test_data):
    """
        依据生成的决策树获取测试数据的分类
    """
    first_fut = list(tree.keys())[0]
    label_index = labels.index(first_fut)
    second_tree = tree[first_fut]
    label = ''
    for key in second_tree.keys():
        if test_data[label_index] == key:
            if isinstance(second_tree[key], dict):
                label = get_test_label(second_tree[key], labels, test_data)
            else:
                label = second_tree[key]
    return label


def run():
    dataset, labels = create_dataset()
    tree = create_tree(dataset, labels)
    create_plot(tree)
    print(str(tree))


if __name__ == '__main__':
    run()
