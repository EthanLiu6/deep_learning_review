import torch
import torch.nn as nn

if __name__ == '__main__':
    tsr1 = torch.tensor([2, 4, 5])
    tsr2 = torch.tensor([[[2, 4, 5],
                          [2, 4, 5]]], dtype=torch.float)

    print(tsr2.shape)
    m = nn.BatchNorm1d(2)
    # m = nn.LayerNorm(2)
    print(m.weight)
    print(m.bias)
    print(m(tsr2))
