import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal


hbar = 1
m = 1
N = 200
L = 1
x = np.linspace(0, L, N)
dx = x[1] - x[0]
K = hbar**2 / (2 * m * dx**2)
V = np.zeros(N - 2)
diagonal = 2 * K + V
off_diagonal = -K * np.ones(N - 3)
E, psi = eigh_tridiagonal(
    diagonal,
    off_diagonal,
    select='i',
    select_range=(0, 0)
)

print("Particle in a box")
print("Numerical energy =", E[0])
x = np.linspace(-2, 2, N)
dx = x[1] - x[0]
K = hbar**2 / (2 * m * dx**2)
V.zeros(N - 2)
for i in range(N - 2):
    if x[i + 1] < -0.5 or x[i + 1] > 0.5:
        V[i] = 10
diagonal = 2 * K + V
off_diagonal = -K * np.ones(N - 3)
E, psi = eigh_tridiagonal(
    diagonal,
    off_diagonal,
    select='i',
    select_range=(0, 0)
)

print("\nFinite well")
print("Ground-state energy =", E[0])
x = np.linspace(-5, 5, N)
dx = x[1] - x[0]

K = hbar**2 / (2 * m * dx**2)
V = 0.5 * x[1:-1]**2
diagonal = 2 * K + V
off_diagonal = -K * np.ones(N - 3)
E, psi = eigh_tridiagonal(
    diagonal,
    off_diagonal,
    select='i',
    select_range=(0, 0)
)

print("\nHarmonic oscillator")
print("Numerical energy =", E[0])
print("Exact energy =", 0.5)
psi = psi[:, 0]

plt.plot(x[1:-1], psi)
plt.xlabel("Position x")
plt.ylabel("Wavefunction")
plt.title("Harmonic Oscillator Ground State")
plt.grid()
plt.show()
