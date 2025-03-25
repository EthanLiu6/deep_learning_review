import torch
import torch.nn as nn


def padding_mask(ipt, seq_len: int = 4):
    # ipt: [batch_size, seq_len, embedding_dim]
    # need mask size: [batch_size, seq_len], because of 'attention' will set them to zero
    masked_ipt = []
    for batch in range(len(ipt)):
        if len(ipt[batch]) < seq_len:
            masked_ipt.append(ipt[batch] + [0] * (seq_len - (len(ipt[batch]))))

        elif len(ipt[batch]) > seq_len:
            masked_ipt.append(ipt[batch][: seq_len])
        else:
            masked_ipt.append(ipt[batch])
    masked_ipt = torch.Tensor(masked_ipt)
    # print(masked_ipt)
    return masked_ipt


if __name__ == '__main__':
    ipt = [
        [3, 1, 4, 6, 2],
        [2, 3, 4, 5],
        [3, 2, 3]
    ]
    masked_ipt = padding_mask(ipt)
    print(masked_ipt)
