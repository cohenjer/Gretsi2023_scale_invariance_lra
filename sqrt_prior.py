import numpy as np
import matplotlib.pyplot as plt

# Plotting the allure of (x-y)^2 + lambda*sqrt(l1(x)), 1D and 2D, for small and large lambda

# 1D
y = 2
x = np.linspace(-2,4,100)
lamb = 5
minx = lamb**(2/3)
valminx = minx + lamb/np.sqrt(minx)
print(valminx,y)

f = (x-y)**2 + lamb*np.sqrt(np.abs(x))

#plt.plot(x,f)
#plt.show()

# 2D
y = np.array([0.5,3])
N = 201
x1 = np.linspace(-4,4,N) 
x2 = np.linspace(-4,4,N) 
for lamb in np.linspace(0,4,20):
    f2d = np.zeros((N,N))
    for i,x1i in enumerate(x1):
        for j,x2j in enumerate(x2):
            f2d[i,j] = (x1i - y[0])**2 + (x2j - y[1])**2 + lamb*np.sqrt(np.abs(x1i)+np.abs(x2j))
    x1min,x2min = np.unravel_index(np.argmin(f2d),f2d.shape)
    x1min = x1[x1min]
    x2min = x2[x2min]
    print(x1min,x2min, lamb)
# l'enfer sur terre commence
X1,X2 = np.meshgrid(x1,x2)
plt.figure()
plt.contour(x1,x2,f2d,levels=20)
plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X1, X2, f2d,cmap='viridis', edgecolor='none')
plt.show()