"""
    序列最小化算法 完整版
"""
from numpy import shape, multiply, nonzero, mat, array, exp, sign
from numpy.matlib import zeros

from support_vector_machine.smo_simple import select_j_rand, clip_alpha, load_data_set, plot_scatter


class OptStruct:
    """
        保存重要的值 只是作为一个保存数据的对象
    """

    def __init__(self, data_matrix, labels, c, toler, kernel_type=('none',)):
        self.data_matrix = data_matrix
        self.label_matrix = labels
        self.c = c
        self.toler = toler
        self.m = shape(data_matrix)[0]
        self.alphas = zeros((self.m, 1))
        self.b = 0
        self.e_cache = zeros((self.m, 2))  # 缓存E的值 第一列表示E是否有效 第二列保存对应的值 有效意味着已经计算好了
        # 初始化核函数的计算结果
        self.k = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.k[:, i] = kernel_trans(self.data_matrix, self.data_matrix[i], kernel_type)


def smo_complete(data_matrix, labels, c, toler, max_iter=500, kernel_type=('none',)):
    """
        完整版smo 序列最小化算法 循环模块
    """
    os = OptStruct(mat(data_matrix), labels, c, toler, kernel_type)
    iter_times = 0
    alpha_pairs_changed = 0
    entire_set = True  # 表示是否为全alpha遍历
    # 这里的退出条件为循环次数超过最大值 或者 遍历了整个alpha也没有发生改变
    while iter_times < max_iter and (alpha_pairs_changed > 0 or entire_set):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(os.m):
                alpha_pairs_changed += inner_l(i, os)
                print('fullSet, iter: %d i:%d, pairs changed %d' % (iter_times, i, alpha_pairs_changed))
            iter_times += 1
        else:
            # 遍历非边界值
            non_bound_i = nonzero((os.alphas.A > 0) * (os.alphas.A < c))[0]
            for i in non_bound_i:
                alpha_pairs_changed += inner_l(i, os)
                print('non-bound, iter: %d i:%d, pairs changed %d' % (iter_times, i, alpha_pairs_changed))
            iter_times += 1
        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = True
        print('iteration number %d' % iter_times)
    return os.b, os.alphas


def inner_l(i, os: OptStruct):
    """
        完整版 smo 序列最小化算法 计算模块 包含一次alpha优化 注释见smo_simple.py
    """
    ei = calc_e(os, i)
    if ((os.label_matrix[i] * ei < -os.toler) and (os.alphas[i] < os.c)) or \
            ((os.label_matrix[i] * ei > os.toler) and (os.alphas[i] > 0)):
        j, ej = select_j(i, os, ei)
        alpha_old_i = os.alphas[i].copy()
        alpha_old_j = os.alphas[j].copy()
        if os.label_matrix[i] != os.label_matrix[j]:
            limit_l = max(0, os.alphas[j] - os.alphas[i])
            limit_h = min(os.c, os.c + os.alphas[j] - os.alphas[i])
        else:
            limit_l = max(0, os.alphas[j] + os.alphas[i] - os.c)
            limit_h = min(os.c, os.alphas[j] + os.alphas[i])
        if limit_l == limit_h:
            print('L==H')
            return 0
        eta = os.k[i, i] + os.k[j, j] - 2.0 * os.k[i, j]
        if eta <= 0:
            print('eta<=0')
            return 0
        os.alphas[j] += os.label_matrix[j] * (ei - ej) / eta
        os.alphas[j] = clip_alpha(os.alphas[j], limit_h, limit_l)
        update_e(os, j)
        if abs(os.alphas[j] - alpha_old_j) < 0.0001:
            print('j not moving enough')
            return 0
        os.alphas[i] += os.label_matrix[j] * os.label_matrix[i] * (alpha_old_j - os.alphas[j])
        update_e(os, i)
        b1 = (-ei - os.label_matrix[i] * (os.alphas[i] - alpha_old_i) * os.k[i, i]
              - os.label_matrix[j] * (os.alphas[j] - alpha_old_j) * os.k[i, j] + os.b)[0, 0]
        b2 = (-ej - os.label_matrix[i] * (os.alphas[i] - alpha_old_i) * os.k[i, j]
              - os.label_matrix[j] * (os.alphas[j] - alpha_old_j) * os.k[j, j] + os.b)[0, 0]
        if 0 < os.alphas[i] < os.c:
            os.b = b1
        elif 0 < os.alphas[j] < os.c:
            os.b = b2
        else:
            os.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def calc_e(os: OptStruct, j):
    """
        计算索引为k对应的E的值
    """

    fx = float(multiply(os.alphas, os.label_matrix).T * os.k[:, j]) + os.b
    return fx - float(os.label_matrix[j])


def select_j(i, os: OptStruct, ei):
    """
        选择内循环的alpha值
    """
    max_j = -1
    max_delta_e = 0
    max_ej = 0

    os.e_cache[i] = [1, ei]  # 将缓存中的i位置的值设置为有效
    valid_e_cache_list = nonzero(os.e_cache[:, 0].A)[0]  # .A 表示转换为数组 获取有效的E缓存 有效代表着对应的alpha处于0和C之间
    if len(valid_e_cache_list) > 1:
        # 循环选择Ei和Ej差距最大的一个j 因为差距越大代表优化后端变化越大  aj变化和Ei-Ej成正相关
        for j in valid_e_cache_list:
            if j == i:
                continue
            ej = calc_e(os, j)
            delta_e = abs(ei - ej)
            if delta_e > max_delta_e:
                max_ej = ej
                max_delta_e = delta_e
                max_j = j
        return max_j, max_ej
    else:
        # 如果第一次 也就是valid_e_cache_list中都是无效的 就随机选择一个j进行优化
        j = select_j_rand(i, os.m)
        ej = calc_e(os, j)
    return j, ej


def update_e(os: OptStruct, j):
    """
        更新E值到缓存
    """
    ej = calc_e(os, j)
    os.e_cache[j] = [1, ej]


def kernel_trans(data_matrix, data_i, kernel_type):
    """
        使用核函数转换数据
    """
    m = len(data_matrix)
    k = zeros((m, 1))
    if kernel_type[0] == 'none':
        k = data_matrix * data_i.T
    elif kernel_type[0] == 'rbf':
        for i in range(m):
            delta_row = data_matrix[i] - data_i
            k[i] = -delta_row * delta_row.T
        k = exp(k / (2 * kernel_type[1] ** 2))
    else:
        raise NameError('kernel type only support none or rbf')
    return k


def run():
    data_matrix, label_matrix = load_data_set('testSet.txt')
    b, alphas = smo_complete(data_matrix, mat(label_matrix).transpose(), 0.6, 0.001)
    w = multiply(alphas.T, mat(label_matrix)) * data_matrix
    plot_scatter(array(data_matrix), label_matrix, alphas, b, w)


def run_with_kernel(reach):
    """
        添加核函数功能运行
    """
    data_matrix, label_matrix = load_data_set('testSetRBF.txt')
    data_matrix = mat(data_matrix)
    label_matrix = mat(label_matrix).T
    kernel_type = ('rbf', reach)
    b, alphas = smo_complete(data_matrix, label_matrix, 0.6, 0.001, kernel_type=kernel_type)
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
    print('the training error rate is %f' % float(error_count / m))

    # 遍历测试向量 计算正确率
    data_matrix, label_matrix = load_data_set('testSetRBF2.txt')
    data_matrix = mat(data_matrix)
    m = shape(data_matrix)[0]
    label_matrix = mat(label_matrix).T
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(support_matrix, data_matrix[i, :], kernel_type)
        predict = kernel_eval.T * multiply(support_label, alphas[support_vector_index]) + b
        if sign(predict) != sign(label_matrix[i]):
            error_count += 1
    print('the test error rate is %f' % float(error_count / m))
    plot_scatter(array(data_matrix), label_matrix, alphas, b, None)


if __name__ == '__main__':
    # run()
    run_with_kernel(2)
