"""
    朴素贝叶斯算法 用于分类 通过得出某类的概率 这里用来进行垃圾文本过滤
"""
from numpy import zeros, ones, log


def load_dataset():
    """
        加载字符数据 返回数据集和标签
    """

    dataset = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 0代表不是脏话 1代表脏话
    labels = [0, 1, 0, 1, 0, 1]
    return dataset, labels


def get_vocab_list(dataset):
    """
        获取对应的文本集中的所有单词set集
    """
    vocab_list = set([])
    for data in dataset:
        vocab_list = vocab_list | set(data)
    return list(vocab_list)


def get_words_in_vocab(vocab_list: list, words_list):
    """
        将输入文档依据单词是否保存在词汇表中转换为文档向量
    :param vocab_list: 词汇表
    :param words_list: 转换文档
    :return: 文档向量
    """
    words_length = len(vocab_list)
    words_matrix = zeros(words_length)
    for word in words_list:
        if word in vocab_list:
            words_matrix[vocab_list.index(word)] = 1
    return words_matrix


def train_naive_bayes(train_matrix, labels):
    """
        依据文档向量和对应的标签对数据进行训练 其实就是一个计算出单词和标签的概率的过程
    :param train_matrix 文档向量
    :param labels 文档向量对应的结果标签
    :return: 每个结果对应的单词出现概率，标签(结果)概率
    """
    vocab_length = len(train_matrix[0])
    # 这两个变量用于保存对应标签的对应的字符数量 这里初始化为1而不为零是因为如果有为0的值后面用于相乘会使数据清零
    # label1_matrix = zeros(vocab_length)
    label1_matrix = ones(vocab_length)
    # label0_matrix = zeros(vocab_length)
    label0_matrix = ones(vocab_length)
    # 标签数量和单词数量
    label1_size, label0_size, label0_words, label1_words = 0, 0, 2, 2
    train_size = len(train_matrix)
    for i in range(train_size):
        if labels[i] == 1:
            label1_size += 1
            label1_matrix += train_matrix[i]
            label1_words += sum(train_matrix[i])
        else:
            label0_size += 1
            label0_matrix += train_matrix[i]
            label0_words += sum(train_matrix[i])
    # 返回在标签为0的数据中 词汇表中的每个单词出现占所有出现单词比列，同前，返回标签为0的词汇向量占所有词汇向量的比列
    return label0_matrix / label0_words, label1_matrix / label1_words, label0_size / float(len(labels))


def classify_data(need_classify_data, label0_words_probability, label1_words_probability, label0_probability):
    """
        对数据进行分率 利用贝叶斯准则 p(c|x,y)=p(x,y|c)p(c)/p(x,y)
    """
    # 因为概率为1和概率为0计算所用分母 p(x,y) 都是一样的 不影响比较结果 所以可以不计算
    # python中对数据小的数据相乘 最后可能四舍五入为0 这里利用代数公式 ln(a*b)=ln(a)+ln(b) 可以将乘法转换为加法 避免这种情况
    wp0 = log(label0_words_probability)
    wp1 = log(label1_words_probability)
    # need_classify_data中只有0或1 所以相乘可以直接取出有效的数据
    p0 = sum(need_classify_data * wp0) + log(label0_probability)
    p1 = sum(need_classify_data * wp1) + log(1 - label0_probability)
    if p0 > p1:
        return 0
    else:
        return 1


def run():
    dataset, labels = load_dataset()
    vocab_list = get_vocab_list(dataset)
    train_matrix = []
    for data in dataset:
        train_matrix.append(get_words_in_vocab(vocab_list, data))
    label0_words_probability, label1_words_probability, label0_probability = train_naive_bayes(train_matrix, labels)
    test_data1 = ['love', 'my', 'dalmation']
    test_matrix1 = get_words_in_vocab(vocab_list, test_data1)
    result = classify_data(test_matrix1, label0_words_probability, label1_words_probability, label0_probability)
    print(str(test_data1) + ' is ' + str(result))
    test_data2 = ['stupid', 'garbage']
    test_matrix2 = get_words_in_vocab(vocab_list, test_data2)
    result2 = classify_data(test_matrix2, label0_words_probability, label1_words_probability, label0_probability)
    print(str(test_data2) + ' is ' + str(result2))


if __name__ == '__main__':
    run()
