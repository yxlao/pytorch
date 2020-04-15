import torch as th

shape = (2, 268435466)
a = th.ones(shape).cuda().to(th.int64)

print("// start reduction")
b = th.sum(a, 0, keepdim=True)
print("// end reduction")
print(b.shape)
print(b)
