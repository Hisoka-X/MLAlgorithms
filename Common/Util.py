from matplotlib import pyplot


def show_scatter(nums1: list, nums2: list):
    """
    根据两个数组展示对应散点图
    """
    # add-subplot中的111表示：画布1行1列 该画占第1个
    pyplot.figure().add_subplot(111).scatter(nums1, nums2)
    pyplot.show()
