import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
def f(x):
  return 1
def delta(x):
  if x==0:
   return 16000 
  else:
   return 0
n=500
xmax=10000
xmin=-10000
dx=(xmax-xmin)/(n-1)
x= np.arange(xmin,xmax+dx,dx,dtype=np.complex_)
data=np.zeros(n,dtype=np.complex_)
analytic=np.zeros(n,dtype=np.complex_)
k=np.fft.fftfreq(n,dx)
k=2*np.pi*k
kk=np.linspace(k.min(),k.max(),num=n,endpoint=True)
for i in range(0,n,1):
  data[i]=f(x[i])
  analytic[i]=delta(kk[i])
nft=np.fft.fft(data,norm='ortho')
aft=dx*np.sqrt(n/(2.0*np.pi))*(np.exp(-1j*k*x.min()))*nft
plt.plot(k,aft.real,"r",label="Fourier Transform of f(x)=1 using Numpy")
plt.plot(kk,analytic.real,"g",label="Analytic Fourier Transform")
plt.legend(fontsize=8)
plt.title("Fourier transform of constant function,n=500, xmax=|xmin|=10000",fontsize=8)
plt.xlabel("k",fontsize=8)
plt.ylabel("F(k)",fontsize=8)
axes=plt.gca()
axes.set_ylim([0,10000])
plt.show()

