"""
    测试朴素贝叶斯算法
    这里对邮件进行过滤
"""
import random
import re

from naive_bayes.bayes import get_vocab_list, get_words_in_vocab, train_naive_bayes, classify_data


def text_parse(text):
    """
     对字符串进行过滤处理 去除无效的字符
    """
    words = re.split(r'\W+', text)
    return [t.lower() for t in words if len(t) > 2]


def get_email_data():
    """
        获取邮件词汇和对应的标签
    """
    labels = []
    email_data = []
    for i in range(1, 26):
        spam_text = open('email/spam/%d.txt' % i, encoding='utf-8').read()
        email_data.append(text_parse(spam_text))
        labels.append(1)
        ham_text = open('email/ham/%d.txt' % i, encoding='utf-8').read()
        email_data.append(text_parse(ham_text))
        labels.append(0)
    return email_data, labels


def run():
    email_data, labels = get_email_data()
    vocab_list = get_vocab_list(email_data)
    # 随机选择部分数据用于训练 其余用于测试 （留存交叉验证）
    training_set = list(range(len(email_data)))
    test_set = []
    for i in range(10):
        num = random.randint(0, len(training_set) - 1)
        while num in test_set:
            num = random.randint(0, len(training_set) - 1)
        test_set.append(num)
        del (training_set[num])
    # 获取训练集矩阵
    training_matrix = []
    training_labels = []
    for i in training_set:
        training_matrix.append(get_words_in_vocab(vocab_list, email_data[i]))
        training_labels.append(labels[i])
    label0_words_probability, label1_words_probability, label0_probability = train_naive_bayes(training_matrix,
                                                                                               training_labels)
    # 测试数据集验证
    error_count = 0
    for i in test_set:
        result = classify_data(get_words_in_vocab(vocab_list, email_data[i]), label0_words_probability,
                               label1_words_probability,
                               label0_probability)
        print(' '.join(email_data[i]) + ' result is {}, need is {}, index is {}'.format(result, labels[i], i + 1))
        if result != labels[i]:
            error_count += 1
    print('error rate is ', float(error_count) / len(test_set))


if __name__ == '__main__':
    run()
