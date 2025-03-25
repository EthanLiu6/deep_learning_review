import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax(x: torch.Tensor):
    mu = sum(torch.exp(n) for n in x)
    return [torch.exp(x_i)/mu for x_i in x]


if __name__ == '__main__':
    tsr1 = torch.tensor([2, 4, 5], dtype=torch.float)
    tsr2 = torch.tensor([[[2, 4, 5],
                          [2, 4, 5]]], dtype=torch.float)

    # print(nn.Sigmoid(tsr1))
    # print(F.sigmoid(tsr1))
    print(softmax(tsr1))
    print('*' * 30)
    print(F.softmax(tsr1, dim=0))
