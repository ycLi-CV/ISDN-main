import math

import torch

if __name__ == "__main__":

    b = torch.empty(1000,1000)

    b = torch.nn.init.kaiming_uniform_(b, a=math.sqrt(100))

    print(b.mean(), b.std())
    print(b)