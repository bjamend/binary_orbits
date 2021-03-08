import numpy as np 
import matplotlib.pyplot as plt 

dat = np.loadtxt('solution.dat')
x = dat[:,1]
y = dat[:,2]
z = dat[:,3]

r = []

for i in range(len(x)):
    rv = np.sqrt(x[i]*x[i] + y[i]*y[i])
    r.append(rv)

plt.plot(r,z)
plt.title('Z vs. R')
plt.xlabel('R (kpc)')
plt.ylabel('Z (kpc)')
plt.show()