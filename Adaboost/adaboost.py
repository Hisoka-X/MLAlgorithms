"""
    AdaBoost元算法 通过构建一个简单的弱分类器 强化为强分类器
"""
from numpy import ones, mat, shape, inf, zeros, log, multiply, exp, sign, dot


def stump_classify(data_matrix, feature_index, thresh_val, thresh_symbol):
    """
        根据条件对数据进行分类 返回分类结果

        :param data_matrix 数据集
        :param feature_index 所选特征位置
        :param thresh_val 决策值
        :param thresh_symbol 决策符号
    """
    m = len(data_matrix)
    result = ones((m, 1))
    if thresh_symbol == 'lt':
        result[data_matrix[:, feature_index] <= thresh_val] = -1.0
    else:
        result[data_matrix[:, feature_index] > thresh_val] = -1.0
    return result


def build_stump(data_matrix, labels, weight):
    """
        依据权重构建错误指数最小的弱分类器 错误的数据所占权重较高 所以会优先优化这些数据
    """
    data_matrix = mat(data_matrix)
    step_size = 10.0
    m, n = shape(data_matrix)
    min_weight_error = inf
    stump_info = {}
    best_predicted = zeros((m, 1))
    # 遍历所有的抉择可能（相对所有） 找出加权后错误率最小的预测结果
    for i in range(n):
        # 计算每次需要移动的步长
        min_data = data_matrix[:, i].min()
        step_length = (data_matrix[:, i].max() - min_data) / step_size
        for j in range(-1, int(step_size) + 1):
            thresh_val = min_data + step_length * j
            for thresh_symbol in ['lt', 'gt']:
                predicted_result = stump_classify(data_matrix, i, thresh_val, thresh_symbol)
                error_matrix = ones((m, 1))
                error_matrix[predicted_result == labels.T] = 0
                weight_error = dot(weight.T, error_matrix)
                if min_weight_error > weight_error:
                    min_weight_error = weight_error
                    stump_info['feature_index'] = i
                    stump_info['thresh_val'] = thresh_val
                    stump_info['thresh_symbol'] = thresh_symbol
                    best_predicted = predicted_result.copy()
    return stump_info, min_weight_error, best_predicted


def adaboost_train(data_matrix, labels, iter_time=40):
    """
        利用adaboost算法获得强分类器
    """
    # 保留每个弱分类器的信息 组成一个强分类器
    weak_class_list = []
    m, n = shape(data_matrix)
    weight = ones((m, 1)) / m
    # 记录每个数据被目前所有分类器分类后的结果 依据分类器权重计算
    agg_class_result = zeros((m, 1))
    print('labels', labels)
    for i in range(iter_time):
        # 获取当前权重最佳分类器
        stump_info, min_weight_error, best_predicted = build_stump(data_matrix, labels, weight)
        # 根据分类器的错误率计算出分类器的权重
        alpha = float(0.5 * log((1 - min_weight_error) / min_weight_error))
        stump_info['alpha'] = alpha
        weak_class_list.append(stump_info)
        print('i ', i)
        # print('weight: ', weight.T)
        # 获取下次计算的数据权重 不是分类器权重 这里的权重根据错误分类的数据的来 错误的数据对应的权重越高
        # 有助于下次生成分类器 分类成功
        # 计算新的数据
        expon = multiply(-1 * alpha * mat(labels).T, best_predicted)
        weight = multiply(weight, exp(expon))
        weight = weight / weight.sum()
        # 计算目前所有分类器加权后的数据分类结果
        agg_class_result += alpha * best_predicted
        # print('best predict ', best_predicted)
        # print('agg class result ', agg_class_result)
        # 计算错误率
        error_num = multiply(sign(agg_class_result) != mat(labels).T, ones((m, 1)))
        error_rate = error_num.sum() / m
        print('error rate', error_rate)
        print('###########################################')
        if error_rate == 0.0:
            break
    return weak_class_list


def class_use_adaboost(data_matrix_to_class, adaboost_classify):
    """
        利用adaboost分类器进行分类
    """
    # 循环每个弱分类器
    data_matrix_to_class = mat(data_matrix_to_class)
    m, n = shape(data_matrix_to_class)
    agg_class_result = mat(zeros((m, 1)))
    for i in range(len(adaboost_classify)):
        # 单个弱分类器的分类结果
        stump_result = stump_classify(data_matrix_to_class, adaboost_classify[i]['feature_index'],
                                      adaboost_classify[i]['thresh_val'], adaboost_classify[i]['thresh_symbol'])
        # 分类结果乘以权重得到单个分类器结果占权结果
        agg_class_result += stump_result * adaboost_classify[i]['alpha']
        # print('agg class result: ', agg_class_result)
    return sign(agg_class_result)


def load_dataset(filename):
    """
        加载数据集
    """
    lines = open(filename).readlines()
    data_matrix = []
    labels = []
    num_feature = len(lines[0].split('\t'))
    for line in lines:
        features_array = []
        features = line.strip().split('\t')
        for i in range(num_feature - 1):
            features_array.append(float(features[i]))
        data_matrix.append(features_array)
        labels.append(float(features[-1]))
    return mat(data_matrix), mat(labels)


def test_for_horse_colic():
    data_matrix, labels = load_dataset('horseColicTraining2.txt')
    adaboost_classify = adaboost_train(data_matrix, labels)

    test_data_matrix, test_labels = load_dataset('horseColicTest2.txt')
    classify_result = class_use_adaboost(test_data_matrix, adaboost_classify)
    m, n = shape(test_data_matrix)
    error_count = mat(ones((m, 1)))
    error_num = error_count[classify_result != test_labels.T].sum()
    print('test error count %f ' % (float(error_num) / m))


if __name__ == '__main__':
    test_for_horse_colic()
