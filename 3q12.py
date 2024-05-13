import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy as sp
import matplotlib.pyplot as plt
def f1(x):
  return np.exp(-x**2)
def f2(x):
  return np.exp(-4*x**2)
n=256
data1=np.zeros(n,dtype=np.complex_)
data2=np.zeros(n,dtype=np.complex_)
x=np.zeros(n,dtype=np.complex_)
xmin=-7
xmax=7
dx=(xmax-xmin)/(n-1)
for i in range(n):
  x[i]=xmin + i*dx
  data1[i]=f1(x[i])
  data2[i]=f2(x[i])
dft1=np.fft.fft(data1,norm='ortho')
dft2=np.fft.fft(data2,norm='ortho')
dft=dft1*dft2
conv=np.fft.ifftshift(np.fft.ifft(dft,norm='ortho'))
conv=conv*dx*np.sqrt(n)
# Compute the convolution analytically
def analytical_convolution(x):
    from scipy.integrate import quad
    def integrand(t):
        return np.exp(-t**2) * np.exp(-4*(x - t)**2)
    result, _ = quad(integrand, -np.inf, np.inf)
    return result
analytical_result = np.array([analytical_convolution(x[i]) for x[i] in x])
plt.plot(x, analytical_result, label='Analytical Convolution')
plt.plot(x.real,conv.real,".",label="Numerical Convolution")
plt.legend(fontsize=10)
plt.xlabel("$x$",fontsize=10)
plt.show()
