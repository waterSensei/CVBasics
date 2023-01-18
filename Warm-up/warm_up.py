import numpy as np

a = np.array([[2, 3, 4], [5, 2, 200]])
print('a = ')
print(a)

b = a[:1]
print('b = ')
print(b)

f = np.random.randn(400, 1)+3
print('f = ')
print(f)

g = f[f > 0]*3
print('g = ')
print(g)

x = np.zeros(100) + 0.45
print('x = ')
print(x)

y = 0.5 * np.ones([1, len(x)])
print('y = ')
print(y)

z = x + y
print('z = ')
print(z)

l = np.linspace(1, 499, 250, dtype=int)
print('l = ')
print(l)

m = l[::-2]
print('m = ')
print(m)

m[m > 50] = 0
print('m = ')
print(m)
