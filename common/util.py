from matplotlib import pyplot
from numpy import array

decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def show_scatter(nums1: list, nums2: list, labels: array):
    """
    根据两个数组展示对应散点图
    """
    # add-subplot中的111表示：画布1行1列 该画占第1个
    pyplot.figure().add_subplot(111).scatter(nums1, nums2, 15.0 * array(labels), 15.0 * array(labels))
    pyplot.show()


def plot_node(node_txt, center_pt, parent_pt, node_type):
    # 参数说明
    # node_txt 文本
    # xy 坐标1 箭头顶端
    # xytext 坐标2 文字位于此坐标旁
    # textcoords 坐标系
    # xycoords 坐标系
    # bbox 文本框
    # arrowprops 额外参数
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt,
                             textcoords='axes fraction', va='center', ha='center', bbox=node_type,
                             arrowprops=arrow_args)


# def create_plot():
#     """
#         创建树形图
#     """
#     fig = pyplot.figure(1, facecolor='white')
#     fig.clf()
#     create_plot.ax1 = pyplot.subplot(111, frameon=False)
#     plot_node('decision node', (0.5, 0.1), (0.1, 0.5), decision_node)
#     plot_node('leaf node', (0.8, 0.1), (0.3, 0.8), leaf_node)
#     pyplot.show()


def get_num_leafs(tree):
    """
        获取树的叶子节点数目
    """
    num_leafs = 0
    first_fut = list(tree.keys())[0]
    second_tree = tree[first_fut]
    for key in second_tree.keys():
        if isinstance(second_tree[key], dict):
            num_leafs += get_num_leafs(second_tree[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(tree):
    """
        获取树的深度
    """
    depth = 0
    first_fut = list(tree.keys())[0]
    second_tree = tree[first_fut]
    for key in second_tree.keys():
        if isinstance(second_tree[key], dict):
            this_depth = 1 + get_tree_depth(second_tree[key])
        else:
            this_depth = 1
        if this_depth > depth:
            depth = this_depth
    return depth


def plot_mid_text(child_pt, parent_pt, text):
    """
        在父子节点坐标之间打印字符
    """
    x_mid = (parent_pt[0] - child_pt[0]) / 2.0 + child_pt[0]
    y_mid = (parent_pt[1] - child_pt[1]) / 2.0 + child_pt[1]
    create_plot.ax1.text(x_mid, y_mid, text)


def plot_tree(tree, parent_pt, node_txt):
    num_leafs = get_num_leafs(tree)
    depth = get_tree_depth(tree)
    first_fut = list(tree.keys())[0]
    child_pt = (plot_tree.x_off + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_w, plot_tree.y_off)
    plot_mid_text(child_pt, parent_pt, node_txt)
    plot_node(first_fut, child_pt, parent_pt, decision_node)
    second_dict = tree[first_fut]
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_d
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            plot_tree(second_dict[key], child_pt, str(key))
        else:
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_w
            plot_node(second_dict[key], (plot_tree.x_off, plot_tree.y_off), child_pt, leaf_node)
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), child_pt, str(key))
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_d


def create_plot(tree):
    """
        根据树结构打印出图形
    """
    fig = pyplot.figure(1, facecolor='white')
    fig.clf()
    ax_props = dict(xticks=[], yticks=[])
    create_plot.ax1 = pyplot.subplot(111, frameon=False, **ax_props)
    plot_tree.total_w = float(get_num_leafs(tree))
    plot_tree.total_d = float(get_tree_depth(tree))
    plot_tree.x_off = -0.5 / plot_tree.total_w
    plot_tree.y_off = 1.0
    plot_tree(tree, (0.5, 1.0), '')
    pyplot.show()


# if __name__ == '__main__':
#     mytree = {'flippers': {0: 'no', 1: {'no surfacing': {0: 'no', 1: 'yes'}}}}
#     create_plot(mytree)
