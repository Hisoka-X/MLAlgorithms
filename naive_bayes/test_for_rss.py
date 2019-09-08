"""
    利用RSS源的数据进行区域代表词提取
    TODO 未测试 因为没有找到合适的英文RSS源!!!
"""
import operator
import random

import feedparser

from naive_bayes.bayes import get_vocab_list, get_words_in_vocab, train_naive_bayes, classify_data


def get_rss_data():
    """
        基于RSS获取数据
    """
    feedparser.parse('')
    rss_data = []
    labels = []
    # TODO 添加RSS源数据解析

    return rss_data, labels


def filter_vocab_list(vocab_list, datalist):
    """
        排除出现次数过多的单词
    """
    new_datalist: list = []
    freq_dict = {}
    for dataset in datalist:
        new_datalist.extend(dataset)
    for words in vocab_list:
        freq_dict[words] = new_datalist.count(words)
    sorted_vocab_list = sorted(freq_dict, key=operator.itemgetter(1), reverse=True)
    # 返回出现次数30名之后的数据
    return [data[0] for data in sorted_vocab_list[30:]]


def get_top_words(vocab_list, label_wp0, label_wp1):
    """
        获取每个标签最具有代表性的词汇
    """
    print('label 0 top words:')
    for i in range(len(vocab_list)):
        if label_wp0[i] > -6.0:
            print(vocab_list[i])
    print('label 1 top words:')
    for i in range(len(vocab_list)):
        if label_wp1[i] > -6.0:
            print(vocab_list[i])


def run():
    rss_data, labels = get_rss_data()
    vocab_list = get_vocab_list(rss_data)
    new_vocab_list = filter_vocab_list(vocab_list, rss_data)
    training_index = list(range(len(rss_data)))
    test_set = []
    for i in range(int(len(rss_data) / 5)):
        num = random.randint(0, len(training_index) - 1)
        while num in test_set:
            num = random.randint(0, len(training_index) - 1)
        test_set.append(num)
        del training_index[num]
    training_matrix = []
    training_labels = []
    for i in training_index:
        training_matrix.append(get_words_in_vocab(new_vocab_list, rss_data[i]))
        training_labels.append(labels[i])
    label0_words_probability, label1_words_probability, label0_probability = train_naive_bayes(training_matrix,
                                                                                               training_labels)
    error_count = 0
    for i in test_set:
        result = classify_data(get_words_in_vocab(new_vocab_list, rss_data[i]), label0_words_probability,
                               label1_words_probability,
                               label0_probability)
        if result != labels[i]:
            error_count += 1
    print('error rate is {}'.format(float(error_count) / len(test_set)))
    get_top_words(vocab_list, label0_words_probability, label1_words_probability)


if __name__ == '__main__':
    run()
