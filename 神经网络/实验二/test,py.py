import numpy as np
import matplotlib.pyplot as plt


def show_image(img, title=None):
    """显示一个二值图像（0/1），其中1 为白色，0 为黑色"""
    plt.imshow(img, cmap='gray_r')  # gray_r让1是白色，0是黑色
    if title:
        plt.title(title)
    plt.axis('off')


def bin2bipolar(x):
    """
    将0/1编码的二维矩阵转换为-1/+1编码的一维向量
    x: 二维 numpy array, 值为0或1
    """
    # 将形状 (h, w) 展开为 (h*w,)
    # 0 -> -1, 1 -> +1
    return np.where(x.flatten() == 0, -1, 1)


def bipolar2bin(x, shape=(5, 5)):
    """
    将-1/+1编码的一维向量转换回0/1二维矩阵（默认恢复到5x5）
    x: 一维 numpy array, -1/+1 编码
    shape: 二维图像的 (height, width)
    """
    x_bin = np.where(x.reshape(shape) == -1, 0, 1)
    return x_bin


def hopfield_train(X):
    """
    根据 Hebb 规则训练 Hopfield 网络
    X: (p, N) 的数组，每行是一个待存储样本的向量，元素取值 -1/+1
    返回: 权值矩阵 W (N x N)
    """
    p, N = X.shape
    # 初始化权值矩阵
    W = np.zeros((N, N))
    # Hebb 学习规则: W += x^i * (x^i)^T
    for i in range(p):
        W += np.outer(X[i], X[i])
    # 去掉自连接（对角线设为0）
    np.fill_diagonal(W, 0)
    return W


def hopfield_recall(init_state, W, max_iter=10):
    """
    使用同步更新方式对初始状态向量进行 Hopfield 联想检索
    init_state: 一维向量, -1/+1
    W: 权值矩阵 (N x N)
    max_iter: 最大迭代次数
    return: 最终收敛后的状态向量
    """
    state = init_state.copy()
    for _ in range(max_iter):
        # 同步更新：s_new = sgn(W @ s_old)
        new_state = np.sign(W @ state)
        # 若出现 0，则将其视为 +1（可选处理）
        new_state[new_state == 0] = 1

        # 如果状态不再变化，则表示已经收敛
        if np.array_equal(new_state, state):
            break
        state = new_state
    return state


def corrupt_image(vec, flip_ratio=0.2):
    """
    随机翻转指定比例的像素（-1/+1）
    vec: 原始向量 (N,)
    flip_ratio: 翻转比例 (0~1)
    """
    corrupted = vec.copy()
    n_flip = int(len(vec) * flip_ratio)
    flip_indices = np.random.choice(len(vec), n_flip, replace=False)
    corrupted[flip_indices] *= -1  # -1 <-> +1翻转
    return corrupted


def main():
    # ============ 1. 准备二值图像 ============
    # 这里以简单的 5x5 二值图像为例，你也可以自己替换为更复杂的图像
    imgA = np.array([
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1]
    ], dtype=int)

    imgB = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ], dtype=int)

    # ============ 2. 转换为 -1/+1 编码 ============
    vecA = bin2bipolar(imgA)
    vecB = bin2bipolar(imgB)

    # 训练数据 X，假设一次存储两幅图像
    X = np.array([vecA, vecB])  # shape: (2, 25)

    # ============ 3. 训练 Hopfield 网络 (计算权值矩阵) ============
    W = hopfield_train(X)

    # ============ 4. 测试与检索 ============
    np.random.seed(42)  # 固定随机种子，便于复现
    corruptedA = corrupt_image(vecA, flip_ratio=0.2)  # 破坏图像A
    corruptedB = corrupt_image(vecB, flip_ratio=0.2)  # 破坏图像B

    # 使用 Hopfield 进行恢复
    recoveredA = hopfield_recall(corruptedA, W)
    recoveredB = hopfield_recall(corruptedB, W)

    # ============ 5. 可视化比较 ============
    plt.figure(figsize=(10, 4))

    plt.subplot(2, 4, 1)
    show_image(imgA, title="Original A")

    plt.subplot(2, 4, 2)
    show_image(bipolar2bin(corruptedA), title="Corrupted A")

    plt.subplot(2, 4, 3)
    show_image(bipolar2bin(recoveredA), title="Recovered A")

    plt.subplot(2, 4, 4)
    diffA = np.sum(bipolar2bin(recoveredA) != imgA)
    plt.title(f"Diff A: {diffA} pixels")
    plt.imshow(bipolar2bin(recoveredA) != imgA, cmap="gray_r")
    plt.axis('off')

    plt.subplot(2, 4, 5)
    show_image(imgB, title="Original B")

    plt.subplot(2, 4, 6)
    show_image(bipolar2bin(corruptedB), title="Corrupted B")

    plt.subplot(2, 4, 7)
    show_image(bipolar2bin(recoveredB), title="Recovered B")

    plt.subplot(2, 4, 8)
    diffB = np.sum(bipolar2bin(recoveredB) != imgB)
    plt.title(f"Diff B: {diffB} pixels")
    plt.imshow(bipolar2bin(recoveredB) != imgB, cmap="gray_r")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
