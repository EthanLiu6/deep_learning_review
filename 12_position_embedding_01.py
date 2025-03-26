import math
import torch
import torch.nn
# import torch.nn.functional as F
from matplotlib import pyplot as plt
import pandas as pd


def position_embedding(ipt):
    position = torch.zeros_like(ipt)
    print("position.shape: ", position.shape)

    # 三角位置编码
    # d是编码向量的维度，k是第k个token，i是第i个维度的位置编码分量
    # p_k_2i = math.sin(k / (10000 ** (2i/d)))
    d = ipt.shape[-1]
    print("dim: ", d)
    print("batch: ", ipt.shape[0])
    print("seq_len: ", ipt.shape[1])
    for batch_n in range(ipt.shape[0]):  # batch
        for k in range(ipt.shape[1]):
            # print("k:", k)
            for i in range(d):
                # print("i:", i)
                if (i & 1) == 1:  # 奇
                    p_k_i = math.cos(k / (10000 ** (i / d)))
                else:
                    p_k_i = math.sin(k / (10000 ** (i / d)))
                position[batch_n][k][i] = p_k_i

    print("position nums:", position.numel())  # 2 * 5 * 64
    return position


def show_position(position):
    batch = position.shape[0]
    position_len = position.shape[1]
    d = position.shape[-1]
    print("batch:", batch)
    print("position_len:", position_len)
    print("dim:", d)

    for batch_n in range(batch):
        print(f"batch_{batch_n}", "-" * 35)

        for k in range(position_len):
            title = f'batch:{batch_n + 1}, token:{k}, position_embedding'
            plt.plot(range(d), position[batch_n][k], label='position_embedding', color='b', linestyle='--', marker='o')
            plt.title(title)
            plt.xlabel('position embedding dim')
            plt.ylabel(f'p_token: {k + 1}')
            plt.show()
            print("=" * 35)
        break  # 看第一个batch就行


def token0_position_save_scv(position_np, csv_name):
    df = pd.DataFrame(position_np)
    df.columns = [f'dim {i}' for i in range(position_np.shape[1])]
    df.to_csv(csv_name, index_label='token_k')


if __name__ == '__main__':
    ipt = torch.randn(2, 32, 64)  # [batch_size, seq_len, embedding_dim]
    position = position_embedding(ipt)
    # print(position[0])

    position_np = position[0].numpy()
    token0_position_save_scv(position_np, './data/token0_position.csv')

    show_position(position)
