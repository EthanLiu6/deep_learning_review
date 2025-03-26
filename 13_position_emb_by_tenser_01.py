import math
import torch
import matplotlib.pyplot as plt


def position_embedding(ipt):
    batch_size, seq_len, d = ipt.shape
    position = torch.zeros_like(ipt)

    # 生成位置索引
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # [seq_len, 1]
    div_term = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))

    # 偶数维度使用 sin，奇数维度使用 cos
    position[:, :, 0::2] = torch.sin(pos * div_term)  # 偶数
    position[:, :, 1::2] = torch.cos(pos * div_term)  # 奇数

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
            plt.xlabel('d')
            plt.ylabel(f'p_token{k + 1}')
            plt.show()
            print("=" * 35)


if __name__ == '__main__':
    ipt = torch.randn(2, 32, 64)  # [batch_size, seq_len, embedding_dim]
    position = position_embedding(ipt)
    show_position(position)
