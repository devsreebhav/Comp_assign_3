import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
def sinc(x):
  if(x==0):
   return 1
  else:
   return np.sin(x)/x
def Box(x):
   if(x>1):
    return 0
   elif(x<-1):
    return 0
   else:
    return(np.sqrt(np.pi/2))
n=1024   
xmax=500
xmin=-500
dx=(xmax-xmin)/(n-1)
x= np.arange(xmin,xmax+dx,dx,dtype=np.complex_)
data=np.zeros(n,dtype=np.complex_)
box=np.zeros(n,dtype=np.complex_)
k=np.fft.fftfreq(n,dx)
k=2*np.pi*k
kk=np.linspace(k.min(),k.max(),num=n,endpoint=True)
for i in range(0,n,1):
  data[i]=sinc(x[i])
  box[i]=Box(kk[i])
  nft=np.fft.fft(data,norm='ortho')
aft=dx*np.sqrt(n/(2.0*np.pi))*(np.exp(-1j*k*x.min()))*nft
plt.plot(k,aft.real,"r",label="Fourier Transform using Numpy")
plt.plot(kk,box.real,"g",label="Analytic Fourier Transform(Box function)")
plt.legend(fontsize=8)
plt.title("Fourier transform of $sinc(x)$",fontsize=8)
plt.xlabel("k",fontsize=8)
plt.ylabel("F(k)",fontsize=8)
plt.show()
outname="fftnumpy.txt"
A=np.zeros([n,4])
A[:,0]=k
A[:,1]=aft.real
A[:,2]=kk
A[:,3]=box.real
