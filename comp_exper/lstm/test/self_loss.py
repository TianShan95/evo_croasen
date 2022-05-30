import math

import torch

# a =
# d = [3, 2, 1]
a = torch.tensor([1, 2, 3]).float()
b = torch.tensor([4, 5, 6]).float()

# c = torch.log(b)
c = 1 - b
print(c)