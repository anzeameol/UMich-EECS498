import torch


def f(p):
    p["a"] = p["a"] + 1
    return p


p = {}
p["a"] = 0

c1 = f(p)
c2 = f(p)

print(c1)
print(c2)


torch.optim.Adam()
