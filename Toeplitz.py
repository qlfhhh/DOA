#%%
import numpy as np
from numpy import pi,e
import math
from scipy.linalg import toeplitz
from matplotlib import pyplot as plt
import scipy
def Multiple_Toeplitz_function(Signal,SnapNum,K,d,N,lmda):
    """
    %Signal                         Data received by the array
    %SnapNum                        Snapshots number
    %K                              The number of signal sources
    %d                              Array element spacing
    %N                              The number of received elements
    %lmda                           Carrier wavelength
    """
    CovMatix = Signal@np.matrix.getH(Signal)/SnapNum
    m = int(np.ceil((N+1)/2))
    CovariceNew = np.zeros((m,m))
    J = np.fliplr(np.eye(m,m))
    for i in range(np.size(CovMatix,0)):
        row = CovMatix[m-1:,i]
        col = CovMatix[:m,i][::-1]
        R = toeplitz(col,row)
        CovariceNew = CovariceNew+R@np.matrix.getH(R)
    R = CovariceNew+J@np.conj(CovariceNew)@J
    [U,S,V] = np.linalg.svd(R)
    Us = U[:,:K]
    Us1 = Us[:-1,:]
    Us2 = Us[1:,:]
    Us12 = np.concatenate((Us1,Us2),1)
    [E,Sa,Va] = np.linalg.svd(np.matrix.getH(Us12)@Us12)
    E11 = E[:K,:K]
    E12 = E[:K,K:2*K]
    E21 = E[K:2*K,:K]
    E22 = E[K:2*K,K:2*K]
    M = -(E12@np.linalg.inv(E22))
    Dm,Vm = np.linalg.eig(M)
    print(Dm)
    result = np.degrees(np.arcsin(np.angle(Dm)/pi))
    result.sort()
    return result

def loss(target,num):
    return math.sqrt((target[0]-num[0])**2+(target[1]-num[1])**2)
if __name__=="__main__":
    total_loss = 0
    # Signals = np.zeros((100,11,64),dtype=complex)
    # Labels = np.zeros((100,2))
    Signals = np.loadtxt("Signals.csv",dtype=complex).reshape((-1,11,64))
    Labels = np.loadtxt("Labels.csv")
    #Features = np.zeros((100))
    print(Labels)
    for i in range(100):
        M = 2 * 5 + 1
        f0 = 150e3
        c = 1500
        SnapNum = 64
        # source_doa = np.random.randint(-20,21,size=2)
        # source_doa.sort()
        # print(source_doa)
        source_doa = Labels[i]
        lmda = c / f0
        d = lmda / 2
        w = 2 * pi * f0
        K = len(source_doa)
        # SNR = 10
        # temp_theta = np.sin(np.radians(source_doa)).reshape((1, -1))
        # A = np.exp(
        #     -1j * np.matrix.getH(np.arange(1, M + 1) - (M + 1) / 2).reshape((-1, 1)) @ temp_theta * d * 2 * pi / lmda)
        # Si = np.power(10, ((SNR / 2) / 10) * np.exp(1j * w * np.random.randn(K, 1) @ np.arange(SnapNum).reshape((1, -1))))
        # Si[1, :] = Si[0, :] * np.exp(1j * np.random.rand(1) * 2 * pi)
        # Sig = A @ Si
        # Signal = Sig + (1 / np.sqrt(2)) * (np.random.randn(M, SnapNum) + 1j * np.random.randn(M, SnapNum))
        Signal = Signals[i]
        res = Multiple_Toeplitz_function(Signal,SnapNum,K,d,M,lmda)
        total_loss+=loss(source_doa,res)
        print(res)
    #     Signals[i,:,:] = Signal
    #     Labels[i,:] = source_doa
    # np.savetxt("Signals.csv", Signals.flatten())
    # np.savetxt("Labels.csv", Labels)



