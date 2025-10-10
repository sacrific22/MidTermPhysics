import numpy as np
import matplotlib.pyplot as plt
from math import atan2


# PARAMETERS
k = 1.0       #How strong the pull is: F = -k * r / r^3
m = 1.0       #Mass of the particle (set m=1)
r0 = np.array([1.0, 0.0])   # Starting position (x=1, y=0)
v0 = np.array([0.0, 0.9])   # Starting velocity (0.9 < circular speed 1.0 -> elliptical)
dt = 0.001    #How small each time step is
tmax = 50.0 # Max simulation time
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
pos = np.zeros((nsteps+1, 2))  # Store all positions
vel = np.zeros((nsteps+1, 2))  # Store all velocities
pos[0] = r0.copy()             # Set initial position
vel[0] = v0.copy()             # Set initial velocity

# VELOCITY-VERLET (velocities are updated by half a step, positions by a full step)
a = accel(pos[0])
for i in range(nsteps):
    v_half = vel[i] + 0.5 * a * dt
    pos[i+1] = pos[i] + v_half * dt
    a_new = accel(pos[i+1])
    vel[i+1] = v_half + 0.5 * a_new * dt
    a = a_new

# DIAGNOSTICS

thetas = np.array([atan2(y,x) for x,y in pos])  # Angle at each time
thetas_unwrapped = np.unwrap(thetas)


# PLOTS (Separate Visualisations)
plt.figure(figsize=(6,6))
plt.plot(pos[:,0], pos[:,1])
plt.scatter([0],[0], s=40)
plt.xlabel('x'); plt.ylabel('y')
plt.title('Trajectory (x vs y)')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('trajectory.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(7,4))
plt.plot(ts, thetas_unwrapped)
plt.xlabel('t'); plt.ylabel('theta (unwrapped)')
plt.title('Angle vs time (unwrapped)')
plt.savefig('theta_vs_t.png', dpi=150, bbox_inches='tight')
plt.show()
