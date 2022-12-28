from MyRBM.rbm0 import RBM
import numpy as np

rbm = RBM(2, 4)

# rbm.W = np.array([
#     [1.0, -1],
#     [-1, 1]
# ])
#
# rbm.Bv *= 0
# rbm.Bh *= 0
#
# print(rbm.sample_x([1, 0]))


data = np.array([
    [1, 0],
    [0, 1],
])

rbm.fit(data, epochs=1000, lr=5e-2, gibb_samples=1)

print(rbm.un_p([1, 1]))
print(rbm.un_p([0, 0]))
print(rbm.un_p([1, 0]))
print(rbm.un_p([0, 1]))
