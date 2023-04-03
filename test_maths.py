import numpy as np
import matplotlib.pyplot as plt

a = 5
b = 2
p = 2
lambset = np.linspace(1e-1,5,1000)

t1 = [lamb*a for lamb in lambset]
t2 = [1/2/(lamb**p)*b for lamb in lambset]
f = [t1[i] + t2[i] for i in range(len(lambset))]

lamb_theory = (b/a)**(1/3)
t1_theory = (a*np.sqrt(b/2)*np.sqrt(2))**(2/3)
t2_theory = ((a*np.sqrt(b/2)*np.sqrt(2))**(2/3))/2
pos_minf = np.argmin(f)
pos_eq = np.argmin(np.abs(np.array(t1)-2*np.array(t2)))

print("t1 and t2 values at equality, also theory")
print(t1[pos_eq], t2[pos_eq], t1_theory, t2_theory)
print("Lambda value for optimality")
print(lambset[pos_minf], lambset[pos_eq], lamb_theory)

plt.figure()
plt.plot(lambset,t1)
plt.plot(lambset,2*np.array(t2))
plt.plot(lambset,f)
plt.legend(['t1','2t2','f'])
plt.show()