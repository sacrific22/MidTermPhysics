import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
k = 1.0       # F = -k * r / r^3
m = 1.0
r0 = np.array([1.0, 0.0])
v0 = np.array([0.0, 0.9])   # 0.9 < 1.0 -> elliptical orbit
dt = 0.001
tmax = 50.0
nsteps = int(tmax / dt)

# FUNCTIONS
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

# ARRAYS
ts = np.linspace(0, tmax, nsteps+1)
pos = np.zeros((nsteps+1, 2))
vel = np.zeros((nsteps+1, 2))
pos[0] = r0.copy()
vel[0] = v0.copy()

# VELOCITY-VERLET
a = accel(pos[0])
for i in range(nsteps):
    v_half = vel[i] + 0.5 * a * dt
    pos[i+1] = pos[i] + v_half * dt
    a_new = accel(pos[i+1])
    vel[i+1] = v_half + 0.5 * a_new * dt
    a = a_new

# === REPORT GUIDELINES ===

# --- Kepler's 2nd law: equal areas in equal times ---
# Area swept between t and t+dt is approximately (1/2) * |r x dr|
areas = 0.5 * np.abs(np.cross(pos[:-1], pos[1:] - pos[:-1]))
window = int(1.0 / dt)  # 1 second window for smoothing
area_rate = np.convolve(areas, np.ones(window)/window, mode='valid') / dt

plt.figure(figsize=(7,4))
plt.plot(ts[:len(area_rate)], area_rate)
plt.xlabel('t')
plt.ylabel('dA/dt')
plt.title("Kepler’s 2nd Law: Areal velocity over time")
plt.grid(True)
plt.savefig("kepler_area_velocity.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"Areal velocity (mean): {np.mean(area_rate):.5f} ± {np.std(area_rate):.5f}")

# --- Orbital Period ---
r = np.linalg.norm(pos, axis=1)
r_smooth = np.convolve(r, np.ones(200)/200, mode='same')
minima = np.where((r_smooth[1:-1] < r_smooth[:-2]) & (r_smooth[1:-1] < r_smooth[2:]))[0] + 1
if len(minima) > 1:
    period = (ts[minima[1]] - ts[minima[0]])
    print(f"Estimated orbital period: {period:.3f} s")
else:
    print("Could not determine full period (orbit not completed).")

# --- Plot r(t) ---
plt.figure(figsize=(7,4))
plt.plot(ts, r)
plt.xlabel('t')
plt.ylabel('r')
plt.title('Radial Distance vs Time')
plt.grid(True)
plt.savefig("r_vs_t.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Plot orbit ---
plt.figure(figsize=(6,6))
plt.plot(pos[:,0], pos[:,1])
plt.scatter([0],[0], s=40, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Orbit Trajectory')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("trajectory.png", dpi=150, bbox_inches='tight')
plt.show()
