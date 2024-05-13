import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
filename = f"3q13data.txt"
data = np.loadtxt(filename,usecols=0)
n=data.size
t=range(0,n)
plt.figure(1)
plt.plot(t,data,".",label="Data Points")
plt.legend(fontsize=10)
plt.xlabel("$x_p$",fontsize=15)
plt.ylabel("f($x_p$)",fontsize=15)
dft=np.fft.fft(data,norm='ortho')
k=np.fft.fftfreq(n,1)
ii=np.argsort(k)
plt.figure(2)
plt.plot(k[ii],dft.real[ii],".-",label="DFT of data")
plt.legend(fontsize=10)
plt.xlabel("$k_q$",fontsize=10)
plt.ylabel("$F(k_q)$",fontsize=10)
power=dft*np.conj(dft)
power=power/n
plt.figure(3)
ii=np.argsort(k)
plt.plot(k[ii],power[ii],"-")
plt.plot(k,power,".",label="power sprectrum")
plt.legend(fontsize=10)
plt.xlabel("$k_q$",fontsize=10)
plt.ylabel("$P_n(k_q)$",fontsize=10)
bins=10
bw=int(n/bins)
binned_pow=np.zeros(bw)
for kk in range(bw):
  for i in range(bins):
    binned_pow[kk]+=power[kk+i*bw]
   
bb=np.fft.fftfreq(bw,1)
plt.figure()
ii=np.argsort(bb)
plt.plot(bb[ii],binned_pow[ii],"-*",label="Binned power spectrum")
plt.legend(fontsize=10)
plt.xlabel("bins",fontsize=10)
plt.ylabel("$P_n(k_q)$",fontsize=10)
plt.show()


