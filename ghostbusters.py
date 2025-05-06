import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.linalg import norm 
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# at grid points x_i, i=0,1,...,N_l
# u(x_i,0) = 0, u(0,t) = 0, u(L,t) = 0

x = np.linspace(0, 10, 100)  # meshpoints in space
y = np.linspace(0, 10, 100)  # meshpoints in space
X, Y = np.meshgrid(x, y)  # meshgrid for 3D plot

def solver_explicit(Da, T, I, F):
    # Initialize the solution array
    L = 10  # Length of the domain view as interval [0, L]
    Nl = 100 # Subintervals
    x = np.linspace(0, L, Nl + 1)  # meshpoints in space
    dx = x[1] - x[0]  # mesh size
    dt = dx**2 * F  # time step size
    Nt = int(round(T / float(dt)))  # number of time steps
    t = np.linspace(0, T, Nt + 1)  # meshpoints in time
    
    u = np.zeros(Nl + 1)  # u(x_i,0) = 0, i=0,1,...,N_l
    u_n = np.zeros(Nl + 1)  # u(x_i,t) at time t_n, i=0,1,...,N_l
    
    for i in range(1, Nl +1):
        u_n[i] = I(x[i])
    
    for n in range(0, Nt):
        if n % 1000 == 0:
            print(f"Step {n}/{Nt}")
        for i in range(1, Nl):
            diffusion = (u_n[i + 1] - 2 * u_n[i] + u_n[i - 1])
            reaction = np.exp(min(u_n[i], 20))
            u[i] = u_n[i] + (dt / (Da * dx**2)) * diffusion + dt * reaction
            #u[i] = u_n[i] + dx * (u_n[i + 1] - 2 * u_n[i] + u_n[i - 1]) + dt * f(x[i], t[n])
            
        u[0] = 0  # Boundary condition at x=0
        u[-1] = 0  # Boundary condition at x=L
        u_n , u = u, u_n  
    return x, u_n  

def solver_implicit(Daa, T, I, F):
    L = 10  # Length of the domain view as interval [0, L]
    Nl = 100 # Subintervals
    x = np.linspace(0, L, Nl + 1)  # meshpoints in space
    dx = x[1] - x[0]  # mesh size
    dt = dx**2 * F  # time step size
    Nt = int(round(T / float(dt)))  # number of time steps
    t = np.linspace(0, T, Nt + 1)  # meshpoints in time
    tol = 1e-5  # tolerance for convergence
    max_iter = 1000  # maximum number of iterations for Newton's method
   # ff = 1/ Da * dt / dx**2
    Da = 1/Daa
    u = np.zeros(Nl + 1)  # u(x_i,t) = 0, i=0,1,...,N_l
    u_n = np.zeros_like(u)  # u(x_i,0) at time t_n, i=0,1,...,N_l
    u[0] = 0  # Boundary condition at x=0
    u[-1] = 0  # Boundary condition at x=L
    
    T = np.eye(Nl - 1) * -2*Da / dx**2  # tridiagonal matrix for diffusion term
    offdiag = np.eye(Nl - 1, k=1) + np.eye(Nl - 1, k=-1)
    T += offdiag * 1 * Da / dx**2  # add off-diagonal terms for diffusion
    
    for i in range(0, Nl + 1):
        u_n[i] = I(x[i])
        
    for n in range(Nt):
        # initial guess: previous time step
        u_inner = u_n[1:-1].copy()

        for k in range(max_iter):
            # residual: F(u) = A u + dt * exp(u) - u_n
            expu = np.exp(np.clip(u_inner, -20, 20))
            F = T @ u_inner + dt * expu - u_n[1:-1]

            # check convergence
            if np.linalg.norm(F, ord=2) < tol:
                break

            # Jacobian: A + dt * diag(exp(u))
            J = T + dt * np.diag(expu)

            # Newton step
            delta = solve(J, -F)
            u_inner += delta

        u[1:-1] = u_inner
        u[0] = 0
        u[-1] = 0
        u_n[:] = u  # update for next step

    return x, u_n
   # for n in range(Nt):
    #    for j in range(1, Nl + 1):
     #       u[j] =  (1 / dx ** 2) * (u_n[j] * ff * T[j, j] + u[j-1]) + np.exp(np.clip(u_n[j], -100, 50)) * dt 


   # return x, u_n

def cone(x, y, center=(0, 0), height=8.0, radius=10.0):
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    return np.clip(height * (1 - r / radius), 0, height)

def solver2D_explicit(Da, T, F):
    L = 10  # Length of the domain view as interval [0, L]
    Nl = 99 # Subintervals
    x = np.linspace(0, L, Nl + 1)  # meshpoints in space
    y = np.linspace(0, L, Nl + 1)  # meshpoints in space
    dx = x[1] - x[0]  # mesh size
    dy = y[1] - y[0]  # mesh size
    dt = dx**2 * F  # time step size
    X, Y = np.meshgrid(x, y)

    Nt = int(round(T / float(dt)))  # number of time steps
    t = np.linspace(0, T, Nt + 1)  # meshpoints in time
    u = cone(X, Y, center=(0, 0), height=8.0, radius=10.0)
    B = np.zeros_like(u)  # initialize solution array
    B[0, :] = 0  # Boundary condition at x=0
    B[-1, :] = 0  # Boundary condition at x=L
    for n in range(Nt):
        for j in range(1, Nl):
            for k in range(1, Nl):
                du_dx2 = u[j+1, k] - 2*u[j, k] + u[j-1, k]
                du_dy2 = u[j, k+1] - 2*u[j, k] + u[j, k-1]
                diffusion = (du_dx2 + du_dy2) / dx**2
                reaction = np.clip(np.exp(np.clip(u[j, k], -20, 20)), 0, 1000)

                B[j, k] = u[j, k] + dt * (Da * diffusion + reaction)
        
        # Apply Dirichlet boundary condition (zero at boundary)
        B[0, :] = B[-1, :] = B[:, 0] = B[:, -1] = 0
        u[:, :] = B[:, :]  # update for next step
    return x, y, B

def solver2D_implicit(Daa, T, I, F):
    x = 0

def I(x):
    # initial condition function
    return 0.0001

T = 2   # time
Da = 1.0   # Diffusivity
F = 0.001    # Stability factor

x, y, u = solver2D_explicit(Da, T, F)
print("Max temperature:", np.max(u))
print("Min temperature:", np.min(u))

'''plt.figure(figsize=(8, 6))
plt.contourf(x, y, u, levels=100, cmap='hot')
plt.colorbar(label='Temperature')
plt.title('2D Reaction-Diffusion: Spontaneous Combustion')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()'''
X, Y = np.meshgrid(x, y)  # assuming x, y are from solver2D_explicit
U0 = cone(X, Y)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 7))

# Plot the initial cone
surf1 = ax.plot_surface(X, Y, U0, cmap='pink', linewidth=0, antialiased=False, alpha=0.5)

# Plot the final solution
surf2 = ax.plot_surface(X, Y, u, cmap='hot', linewidth=0, antialiased=False, alpha=0.8)

ax.set_title("3D View of Initial Cone and Final Temperature")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')

ax.set_zlim(0, 8)
ax.zaxis.set_major_formatter('{x:.02f}')

# Optional: Add color bar for the final solution
fig.colorbar(surf2, shrink=0.5, aspect=5, label='Final Temperature')

plt.show()