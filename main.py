# %%
import numpy as np
import math
import matplotlib.pyplot as plt

M = 16
N = 1024
n = np.arange(1024).reshape((1, -1))
f0 = 77e9
c = 3e8
lba = c / f0
d = lba / 2
signalIn = np.zeros((M, N))

# 构造噪声
noise = (np.random.randn(M, N) + np.random.randn(M, N) * 1j) / np.sqrt(2)
R_noise = np.dot(noise, np.conjugate(noise.T)) / N

# 构造相干信号
theta_c = [10, 20]  # 入射角度
rou = [1, 1]  # 相干度
SNR_c = [10, 10]  # 信噪比
w_c = 0
Amp_u = [np.power(10, x / 20) for x in SNR_c]
x_c = np.zeros((M, N))
for i in range(len(theta_c)):
    a_c = rou[i] * np.exp(1j * 2 * np.pi * d / lba * np.arange(M).T * math.sin(math.radians(theta_c[i])))
    a_c = a_c.reshape((16, 1))
    s_c = np.exp(1j * w_c * np.arange(N)) * np.exp(1j * 2 * np.pi * np.random.rand())
    s_c = s_c.reshape((1, 1024))
    x_c = x_c + np.dot(a_c, s_c)
R_c = np.dot(x_c, np.conjugate(x_c.T)) / N
R = R_c + R_noise


def MVDR(signal, thetaSet, d, lbd):
    K = len(thetaSet)
    M = np.size(signal, 1)
    A = np.zeros((M, K), dtype=complex)
    phi = 2 * np.pi * d * np.sin(np.radians(thetaSet)) / lbd
    phi = phi.reshape((1, -1))
    for i in range(M):
        A[i, :] = np.exp(1j * i * phi)  # 此没有负号，说明法线右边为正角度，左边为负角度
    # DOA
    R_inv = np.linalg.inv(signal)
    Amp = np.zeros((K, 1))
    for i in range(K):
        temp = np.dot(np.dot(np.conjugate(A[:, i].T), R_inv), A[:, i])
        Amp[i] = 1 / np.abs(temp)
    theta = thetaSet.reshape((-1, 1))
    return Amp, theta


thetaSet = np.arange(-90, 91)
Amp, theta = MVDR(R, thetaSet, d, lba)
plt.figure()
plt.plot(theta, Amp)
plt.title("MVDR")
plt.xlabel(r"\theta")
plt.ylabel(r"P_{MVDR}(\theta)")
plt.show()
