import torch as th
import numpy as np

# shape = (2, 268435466)
# a = th.ones(shape).cuda().to(th.int64)

# print("// start reduction")
# b = th.sum(a, 0, keepdim=True)
# print("// end reduction")
# print(b.shape)
# print(b)

a_np = np.array([22, 23, 20, 9, 6, 14, 18, 13, 15, 3, 17, 0,
                 7,  21, 11, 1, 4, 2,  10, 19, 5,  8, 16, 12]).reshape((2, 12)).astype(np.float32)
a = th.Tensor(a_np).cuda()
# print(th.argmin(a, 0))
th.argmin(a, 0)
