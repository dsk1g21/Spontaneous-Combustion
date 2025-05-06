import numpy as np
import matplotlib.pyplot as plt

def cone(x, y, center=(0, 0), height=8.0, radius=10.0):
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    return np.clip(height * (1 - r / radius), 0, height)

L = 1 # radius of pile
N = 100 #number of points
dx = dy = L / N # distance between points
x = np.linspace(0, L, N) # meshpoints in space
y = np.linspace(0, L, N) # meshpoints in space

Da = 1 # diffusion coefficient
F = 0.5
time = 10
T0 = 0 # temperature at surface of pile
dt =  min(dx**2 / (4 * Da), dy**2 / (4 * Da)) 
print(f"dt: {dt}")
u = np.zeros((N, N)) 
u[0, :] = u[-1, :] = T0 # boundary conditions
#u[:, 0] = u[:, -1] = T0 # boundary conditions
t_nodes = int(time / dt) # number of time steps

u_n = np.zeros_like(u)

def plot_cone(center=(0, 0), height=1.0, radius=1.0):
    x = np.linspace(center[0] - radius, center[0] + radius, 100)
    y = np.linspace(center[1] - radius, center[1] + radius, 100)
    X, Y = np.meshgrid(x, y)
    Z = cone(X, Y, center=center, height=height, radius=radius)
    return Z

for i in range(N):
    for j in range(N):
        u_n[i, j] = plot_cone(center=(x[i], y[j]), height=8.0, radius=10.0)[i, j]

reaction_term = np.zeros_like(u)
for i in range(1, N-1):
    for j in range(1, N-1):
        reaction_term[i, j] = np.exp(u_n[i, j]) 

for t in range(t_nodes):
    u_n[:, :] = u[:, :]  
    for i in range(1, N-1):
        for j in range(1, N-1):
            dd_ux = (u_n[i+1, j] - 2*u_n[i, j] + u_n[i-1, j]) / dx**2
            dd_uy = (u_n[i, j+1] - 2*u_n[i, j] + u_n[i, j-1]) / dy**2
            reaction = reaction_term[i, j]
            u[i, j] = u_n[i, j] + dt * ((1/Da) * (dd_ux + dd_uy) + reaction)



plt.imshow(u, cmap='hot', interpolation='nearest')
plt.colorbar(label='Temperature')
plt.title(f'Temperature Distribution in Pile at t = {time}, Da = {Da}')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
save_path = rf"C:\Users\jaro7\OneDrive\Documents\cool dawgs\MDEs\temperature_distribution_Da_{Da}_time_{time}.png"
#plt.savefig(save_path, dpi=300)
plt.show()
