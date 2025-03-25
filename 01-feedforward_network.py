"""
多层感知机到前馈神经网络
"""
import torch


class MLP:
    def __init__(self, input_size: int = 5, output_size: int = 4):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = 2
        self.weight1 = torch.randn(input_size, output_size) * torch.sqrt(
            torch.tensor(2.0) / (input_size + output_size))  # init with Xavier
        self.weight2 = torch.randn(1, output_size) * torch.sqrt(
            torch.tensor(2.0) / (output_size + output_size))  # init with Xavier

        self.bias1 = torch.zeros((1, output_size))
        self.bias2 = torch.zeros((1, output_size))

        # ignore, this is for the next step: backward
        self.W1_grad = torch.empty_like(self.weight1)
        self.W2_grad = torch.empty_like(self.weight2)
        self.B1_grad = torch.empty_like(self.bias1)
        self.B2_grad = torch.empty_like(self.bias2)

    def mlp(self, X):
        h1 = torch.matmul(X, self.weight1) + self.bias1
        print("h1 size:", h1.shape)
        out1 = self.sigmoid_(h1)
        print("out1 size:", out1.shape)
        out = torch.matmul(out1, self.weight2.T) + self.bias2.T
        print("out size:", out.shape)
        return out

    def sigmoid_(self, input_):
        return 1 / (1 + torch.exp(-input_))


if __name__ == '__main__':
    input_ = torch.tensor([[1.0, 4, 3, 5],
                          [3, 6, 2, 6],
                          [2, 5, 3, 6]])  # [3, 4]
    model = MLP(4, 3)  # in 3, out 4
    out = model.mlp(input_)  # [3, 4] @ [4, 3] = [3, 3], [3, 3] @ [3, 1] = [3, 1]
    print(out)

