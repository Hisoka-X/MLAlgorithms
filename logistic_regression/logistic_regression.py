"""
    逻辑回归算法
"""
from numpy import mat, shape, ones, exp


def get_dataset():
    fs = open('testSet.txt')
    lines = fs.readlines()
    dataset = []
    labels = []
    for line in lines:
        data = line.strip().split('\t')
        dataset.append([1.0, float(data[0]), float(data[1])])
        labels.append(int(data[-1]))
    return dataset, labels


def sigmoid(x):
    return 1.0 / (1 + exp(-x))


def grad_ascent(dataset, labels):
    """
        逻辑：
            1. 通过sigmoid函数 1/(1/epx(-z)) 将数据取值规范到0-1之间，其中z=w0*x0+w1*x1+w2*x2+...+wn*xn
            2. 通过似然函数（似然函数的值越高，代表利用此时最高位对应的系数得到的sigmoid函数计算出的值概率和理想概率最接近）得到系数对应的函数
            3. 通过梯度上升来计算出似然函数的最大值对应的系数（sigmoid函数的系数即为似然函数的参数）
            4. 关于似然函数：似然函数用于计算概率问题 利用已知的数据集和对应结果概率得出一个数据集对应系数对数据集结果概率影响的函数 函数最高点就是概率最接近的时刻
            5. 关于梯度上升：对函数进行偏导数计算，然后根据步长*对应参数的偏导数进行参数叠加 直到叠加效率很低或者次数达到限制 完成梯度上升 获取到最好的函数参数 这个参数就是系数 函数就是似然函数
    :param dataset: 参数矩阵
    :param labels: 标签矩阵 均为0或1
    :return: 参数对应的系数
    """
    dataset_matrix = mat(dataset)
    labels_matrix = mat(labels).transpose()
    m, n = shape(dataset_matrix)
    # 步长
    alpha = 0.001
    # 最大迭代次数
    max_cycles_time = 500
    # 系数矩阵 初始都为1
    weights = ones((n, 1))
    for i in range(max_cycles_time):
        # 函数实际值
        h = sigmoid(dataset_matrix * weights)
        # 函数理想值和实际值差距 差距越小 下次改变力度越小
        error = labels_matrix - h
        # 该公式根据似然函数对数化之后转换而来 wn=wn+α*i=1∑m (labels(i)−h(i))*x(i)
        weights += alpha * dataset_matrix.transpose() * error
    return weights


def run():
    dataset, labels = get_dataset()
    weights = grad_ascent(dataset, labels)
    print(str(weights))


if __name__ == '__main__':
    run()