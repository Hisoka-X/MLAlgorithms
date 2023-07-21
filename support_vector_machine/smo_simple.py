"""
    SMO算法 序列最小化算法 简化版
"""
import random

from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from numpy import mat, shape, zeros, multiply, arange, ndarray
from numpy.ma import array



def load_data_set(file_name):
    """
        数据集加载
    """
    data_matrix = []
    label_matrix = []
    for line in open(file_name).readlines():
        line_attr = line.strip().split('\t')
        data_matrix.append([float(line_attr[0]), float(line_attr[1])])
        label_matrix.append(float(line_attr[2]))
    return data_matrix, label_matrix


def select_j_rand(i: int, m: int):
    """
        在m范围内选择一个不等于i的值
    """
    j = i
    while j == i:
        j = random.randint(0, m - 1)
    return j


def clip_alpha(a, h, l):
    """
        调整alpha的值 保证其在范围内
    :param a: alpha值
    :param h: 允许最大值
    :param l: 允许最小值y76
    :return: 调整后的值
    """
    if a > h:
        return h
    if a < l:
        return l
    return a


def smo_simple(data_matrix, label_matrix, c, toler, max_iter_times=500):
    """
        简化版序列最小化算法 简化了每次数据参数向量的选择 直接采用随机的方式选择进行优化的参数
        c表示惩罚参数 toler表示软间隔最大化中的松弛参数
    """
    # 数据格式转换
    data_matrix = mat(data_matrix)
    label_matrix = mat(label_matrix).transpose()
    # 利用数据行数生成对应行数的系数alpha 初始化都为0
    m, n = shape(data_matrix)
    alphas = mat(zeros((m, 1)))
    iter_times = 0
    # 截距
    b = 0
    # 在最大允许迭代次数之内进行计算
    while iter_times < max_iter_times:
        alpha_changed = False
        # 遍历所有数据 然后对对应的的alpha参数进行计算
        for i in range(m):
            # 计算出当前预测函数计算出的实际值 然后和理想值进行比对 获取差值用于计算新的alpha参数值
            # 这里的T表示转置 通过这种方式实现两个向量的点乘
            fxi = float(multiply(alphas, label_matrix).T * (data_matrix * data_matrix[i, :].T)) + b
            ei = fxi - float(label_matrix[i])
            # 通过kkt条件得到的支持向量 支持向量必须满足0<=alpha<=c 选择违反kkt条件的向量进行对应的alpha优化
            if ((label_matrix[i] * ei < -toler) and (alphas[i] < c)) or (
                    (label_matrix[i] * ei > toler) and (alphas[i] > 0)):
                # 随机选择第二个alpha变量 这就是简化版的原因 在第二个alpha变量选择上简单化
                j = select_j_rand(i, m)
                fxj = float(multiply(alphas, label_matrix).T * (data_matrix * data_matrix[j, :].T)) + b
                ej = fxj - float(label_matrix[j])
                # 取出改变前的数据 重新分配内存地址 用于比较
                alpha_old_i = alphas[i].copy()
                alpha_old_j = alphas[j].copy()
                # 通过二变量优化问题得到a1和a2的关系 然后得出l和h的取值
                if label_matrix[i] != label_matrix[j]:
                    limit_l = max(0, alphas[j] - alphas[i])
                    limit_h = min(c, c + alphas[j] - alphas[i])
                else:
                    limit_l = max(0, alphas[j] + alphas[i] - c)
                    limit_h = min(c, alphas[j] + alphas[i])
                # 相等说明alpha取值只能是一个值 不符合kkt条件
                if limit_h == limit_l:
                    print('L==H')
                    continue
                # 计算分母 K11+K22-2K12=||f1-f2||² K为核函数(这里没有使用核函数 直接使用两个向量的点乘) f为映射函数
                eta = (data_matrix[i, :] * data_matrix[i, :].T + data_matrix[j, :] * data_matrix[j, :].T
                       - 2.0 * data_matrix[i, :] * data_matrix[j, :].T)
                if eta <= 0:
                    continue
                alphas[j] += label_matrix[j] * (ei - ej) / eta
                alphas[j] = clip_alpha(alphas[j], limit_h, limit_l)
                # 检查更新后的alpha值是否精度足够
                if abs(alphas[j] - alpha_old_j) < 0.00001:
                    print('j not moving enough')
                    continue
                # 因为a1y1+a2y2=k k为常数 通过其他alpha向量和对应的标签值得到 所以可以通过a2的改变量得到a1需要改变的量
                alphas[i] -= label_matrix[i] * label_matrix[j] * (alphas[j] - alpha_old_j)
                # 得到alphas的值后 可以通过函数得到对应的b的值
                # 然后根据alpha值的大小 是否在0<alpha<c或者alpha=0或者alpha=c来进行b值的选取
                b1 = (-ei - label_matrix[i] * (alphas[i] - alpha_old_i) * data_matrix[i, :] * data_matrix[i, :].T -
                      label_matrix[j] * (alphas[j] - alpha_old_j) * data_matrix[i, :] * data_matrix[j, :].T + b)
                b2 = (-ej - label_matrix[i] * (alphas[i] - alpha_old_i) * data_matrix[i, :] * data_matrix[j, :].T -
                      label_matrix[j] * (alphas[j] - alpha_old_j) * data_matrix[j, :] * data_matrix[j, :].T + b)
                if (0 < alphas[i]) and (c > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (c > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2
                alpha_changed = True
                print('iter: %d i:%d alpha_changed %s' % (iter_times, i, alpha_changed))
        if alpha_changed:
            # 如果向量发生更新 所有数据重新遍历计算
            # 需要保证数据在重复max_iter_times次之后依然alpha向量没有更新 这就是趋于稳定
            iter_times = 0
        else:
            iter_times += 1
        print('iteration number: %d' % iter_times)
    return alphas, b[0, 0]


def plot_scatter(data_matrix, label_matrix, alphas, b, w):
    """
        画出散点图和支持向量和平面
    """
    label1_x = []
    label2_x = []
    label1_y = []
    label2_y = []
    ax: Axes = pyplot.figure().add_subplot(111)
    for i in range(len(data_matrix)):
        if label_matrix[i] == 1:
            label1_x.append(data_matrix[i, 0])
            label1_y.append(data_matrix[i, 1])
        else:
            label2_x.append(data_matrix[i, 0])
            label2_y.append(data_matrix[i, 1])
        # 对支持向量做特别标注
        if alphas[i] > 0:
            circle = Circle((data_matrix[i, 0], data_matrix[i, 1]), radius=0.1, facecolor='none',
                            edgecolor=(0, 0.8, 0.8),
                            linewidth=3, alpha=0.5)
            ax.add_patch(circle)
    ax.scatter(label1_x, label1_y, s=30, c='red', marker='s')
    ax.scatter(label2_x, label2_y, s=30, c='green', marker='o')
    if w is not None:
        x = arange(-2.0, 8.0, 0.1)
        # 通过W·X+b=0得到x1和x2的关系 得出x2也就是y
        w0 = w[0, 0]
        w1 = w[0, 1]
        y = (-b - w0 * x) / w1
        ax.plot(x, y)
    pyplot.xlabel('x1')
    pyplot.ylabel('x2')
    pyplot.show()


def run():
    data_matrix, label_matrix = load_data_set('testSet.txt')
    alphas, b = smo_simple(data_matrix, label_matrix, 0.6, 0.001)
    w = multiply(alphas.T, mat(label_matrix)) * data_matrix
    plot_scatter(array(data_matrix), label_matrix, alphas, b, w)


if __name__ == '__main__':
    run()
