# verlet_orbit.py
import numpy as np
import matplotlib.pyplot as plt
from math import atan2
from scipy.signal import find_peaks   # optional: used to estimate period

# PARAMETERS
k = 1.0       # central-force strength: F = -k * r / r^3
m = 1.0       # particle mass (set m=1)
r0 = np.array([1.0, 0.0])   # initial position
v0 = np.array([0.0, 0.9])   # initial velocity (0.9 < circular speed 1.0 -> elliptical)
dt = 0.001    # timestep
tmax = 50.0
nsteps = int(tmax / dt)

# HELPERS
def accel(pos):
    r = np.linalg.norm(pos)
    if r == 0:
        return np.array([0.0, 0.0])
    return -k * pos / (r**3)

def energy(pos, vel):
    r = np.linalg.norm(pos)
    ke = 0.5 * m * np.dot(vel, vel)
    pe = -k / r
    return ke + pe

def ang_momentum(pos, vel):
    return m * (pos[0]*vel[1] - pos[1]*vel[0])

# ALLOCATE
ts = np.linspace(0, tmax, nsteps+1)
pos = np.zeros((nsteps+1, 2))
vel = np.zeros((nsteps+1, 2))
pos[0] = r0.copy()
vel[0] = v0.copy()

# VELOCITY-VERLET (leapfrog style)
a = accel(pos[0])
for i in range(nsteps):
    v_half = vel[i] + 0.5 * a * dt
    pos[i+1] = pos[i] + v_half * dt
    a_new = accel(pos[i+1])
    vel[i+1] = v_half + 0.5 * a_new * dt
    a = a_new

# DIAGNOSTICS
rs = np.linalg.norm(pos, axis=1)
thetas = np.array([atan2(y,x) for x,y in pos])
thetas_unwrapped = np.unwrap(thetas)
energies = np.array([energy(pos[i], vel[i]) for i in range(len(ts))])
Ls = np.array([ang_momentum(pos[i], vel[i]) for i in range(len(ts))])

# PLOTS (each figure is standalone)
plt.figure(figsize=(6,6))
plt.plot(pos[:,0], pos[:,1])
plt.scatter([0],[0], s=40)
plt.xlabel('x'); plt.ylabel('y')
plt.title('Trajectory (x vs y)')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('trajectory.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(7,4))
plt.plot(ts, rs)
plt.xlabel('t'); plt.ylabel('r(t)')
plt.title('Radial distance vs time')
plt.savefig('r_vs_t.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(7,4))
plt.plot(ts, thetas_unwrapped)
plt.xlabel('t'); plt.ylabel('theta (unwrapped)')
plt.title('Angle vs time (unwrapped)')
plt.savefig('theta_vs_t.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(7,4))
relE = (energies - energies[0]) / abs(energies[0])
plt.plot(ts, relE)
plt.xlabel('t'); plt.ylabel('relative energy change')
plt.title('Relative energy change vs time')
plt.savefig('energy_vs_t.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(7,4))
relL = (Ls - Ls[0]) / abs(Ls[0])
plt.plot(ts, relL)
plt.xlabel('t'); plt.ylabel('relative angular momentum change')
plt.title('Relative angular momentum change vs time')
plt.savefig('L_vs_t.png', dpi=150, bbox_inches='tight')
plt.show()

# Estimate period from radial peaks (optional)
peaks, _ = find_peaks(rs)
period_est = None
if len(peaks) >= 2:
    period_est = np.mean(np.diff(ts[peaks[:6]])) if len(peaks) >= 7 else np.mean(np.diff(ts[peaks]))
print("Initial radius:", np.linalg.norm(r0))
print("Initial speed:", np.linalg.norm(v0))
print("Initial energy:", energies[0])
print("Initial L:", Ls[0])
print("Estimated radial period (if detectable):", period_est)
