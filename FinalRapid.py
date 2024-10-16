import numpy as np
import matplotlib.pyplot as plt
import os

# This a 2D SPECS program.
# close all; clear; clc
fsize = 14
num_sims = 10

# rng("shuffle")

# Set parameters
# Simulation information 
total_t = 400         # unit: s. The total simulation time
dt = 0.1               # unit: s. Time step
dt_save = 1           # unit: s. The period to save data
num_steps = round(total_t/dt)

total_x = 800          # unit: um. Channel length
dx = 5                 # unit: um
num_x = round(total_x/dx)

total_y = 400          # unit: um. Channel width
dy = 5                 # unit: um
num_y = round(total_y/dy)

# parameters for Green's function (concentration)
Max = 4e6
Xs = total_x/2
Ys = total_y/2
size_grad = 200 
mag = 5.6
diff_rate = 1.5*0.16 # diffusion coefficient for Green's function (concentration)      
fixed_time = 2000 # fixed static concentration 

num_cell = 10 # %20000;     	# Number of the cells

gradient_x_min = Xs - size_grad / 2
gradient_x_max = Xs + size_grad / 2
gradient_y_min = Ys - size_grad / 2
gradient_y_max = Ys + size_grad / 2
# initialize the location of each cell
x0 = np.zeros((num_cell,1))
y0 = np.zeros((num_cell,1))

# z param
z0 = np.zeros((num_cell,1))  # Initialize z values -> [[0]] (1 cell)

for i in range(num_cell):
    xtmp = total_x * np.random.rand()
    ytmp = total_y * np.random.rand()
    if not ( gradient_x_min <= xtmp <= gradient_x_max and gradient_y_min <= ytmp <= gradient_y_max ):
        x0[i, 0] = (np.random.rand())*total_x
        y0[i, 0] = (np.random.rand())*total_y
    else:
        x0[i, 0] = 10*mag*2*(np.random.rand()-0.5)+total_x/2
        y0[i, 0] = 10*mag*2*(np.random.rand()-0.5)+total_y/2
    # z param
    z0[i, 0] = 0 # initial value of z


def rapid_cell_1(S, m, dt):
    #Constants
    K_on = 12e-3
    K_off = 1.7e-3
    Ks_on = 1e6
    Ks_off = 100
    n = 6
    ns = 12
    cheR = 0.16
    cheB = 0.28
    mb0 = 0.65
    H = 10.3
    a = 0.0625
    b = 0.0714
    cheYt = 9.7

    K_y = 100
    K_z = 30
    G_y = 0.1
    K_s = 0.45

    # Receptor free energy
    F = n * (eps_val(m) + np.log((1 + S / K_off) / (1 + S / K_on))) + ns * (eps_val(m) + np.log((1 + S / Ks_off) / (1 + S / Ks_on)))

    # Cluster activity
    A = 1 / (1 + np.exp(F))

    # Rate of receptor methylation
    m = m + dm(cheR, cheB, a, b, A) * dt

    # Steady-state CheY-P concentration
    cheYp = 3 * (K_y * K_s * A) / (K_y * K_s * A + K_z + G_y)

    # CCW motor bias
    mb = ccw_motor_bias(cheYp, mb0, H)

    return m, mb

def eps_val(m):
    eps = [1.0, 0.5, 0.0, -0.3, -0.6, -0.85, -1.1, -2.0, -3.0]
    eps_vals = np.zeros_like(m)

    for idx in range(len(m)):
        if m[idx] <= 0:
            eps_vals[idx] = eps[0]
        elif m[idx] >= 8:
            eps_vals[idx] = eps[8]
        else:  # linear interpolation
            upper = int(np.ceil(m[idx] + 1))
            lower = int(np.floor(m[idx] + 1))
            slope = eps[upper] - eps[lower]
            eps_vals[idx] = eps[lower] + slope * (m[idx] + 1 - lower)
    
    return eps_vals

    

def dm(cheR, cheB, a, b, A):
    return a * (1 - A) * cheR - b * A * cheB

def ccw_motor_bias(cheYp, mb0, H):
    return (1 + (1 / mb0 - 1) * (cheYp) ** H) ** (-1)
def Green_gradient_old(Max, Xs, Ys, D, point, t):
    r2 = (point[:, 0] - Xs)**2 + (point[:, 1] - Ys)**2
    G = Max * np.exp(-r2 / (4 * D * t)) / (4 * np.pi * D * t)
    return G


def Green_gradient(Max, Xs, Ys, D, point, t):
    r2 = (point[:, 0] - Xs)**2 + (point[:, 1] - Ys)**2
    G = Max * np.exp(-r2[0] / (4 * D * t)) / (4 * np.pi * D * t) # IMPORTANT: see if you really need to change the r2 to r2[0]
    return np.array([G])

















def plot_traj(x_array, y_array, fsize, G, dir_path, num_x, num_y, N_cell, total_x, total_y, Max, Xs, Ys, D, t):
    colorc = np.array([
        [0.502, 0.502, 0.502],  # gray
        [0, 1, 1],               # turquoise
        [1, 0, 1],               # magenta
        [0.9412, 0.4706, 0],     # orange
        [0.251, 0, 0.502],       # purple
        [1, 0.8, 0.2],           # gold
        [0.502, 0.251, 0],       # brown
        [0.502, 0.502, 0.502],   # gray
        [0, 0.502, 0.502],       # green
        [0, 0, 0.4]              # black
    ])
    
    # Create a figure
    fig = plt.figure()
    plt.axis('equal')
    plt.rcParams.update({'font.size': fsize})

    # Generate grid for plotting G
    x = np.arange(0, total_x, 0.1)
    y = np.arange(0, total_y, 0.1)
    x, y = np.meshgrid(x, y)

    # Calculate G using Green_gradient function
    points = np.column_stack((x.ravel(), y.ravel()))
    G = Green_gradient_old(Max, Xs, Ys, D, points, t)
    G = np.reshape(G, x.shape)
    
    # Plot contour of G with 5 levels
    Cc = plt.contourf(x, y, G, 5, linewidths=3, cmap='viridis', levels=100)
    cbar = plt.colorbar(Cc, location='bottom')  # Adjusted location here
    cbar.ax.tick_params(labelsize=fsize)  # Set font size for colorbar ticks
    cbar.set_ticks(np.arange(0, np.ceil(np.max(G)) + 1, 100))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Plot each cell's trajectory
    for ii in range(N_cell):
        plt.plot(x_array[ii, :], y_array[ii, :], '-', color=colorc[ii, :])
    
    # Plot initial positions
    plt.plot(x_array[:, 0], y_array[:, 0], 'xw', markersize=10, linewidth=2, label='Initial positions')
    plt.plot(x_array[:, -1], y_array[:, -1], 'xr', markersize=10, linewidth=2, label='Ending positions')
    
    # Set axis limits and add title
    plt.xlim(0, total_x)
    plt.ylim(0, total_y)
    plt.title('Trajectory: initial (white) & ending (red)')
    
    # Save the plot
    plt.savefig(f'{dir_path}/trajectory.png')

    # Show the plot (if needed)
    #plt.show()





















for i in range(num_sims):
    
    dir_path = f'./RC_sim_{i}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        os.rmdir(dir_path)
        os.makedirs(dir_path)
    
    # initialize the location of each cell
    x = np.copy(x0)
    y = np.copy(y0)
    
    # z param
    z = z0.copy()  # Initialize z positions

    
    x_array = np.zeros((num_cell,num_steps+1))
    x_array[:, 0] = x[:, 0]
    y_array = np.zeros((num_cell,num_steps+1))
    y_array[:, 0] = y[:, 0]
    
    # z param
    z_array = np.zeros((num_cell, num_steps + 1))  # Track z values
    z_array[:, 0] = z[:, 0]
    
    # Run & Tumble related paramerers
    RUN_VELOCITY = 16.5		# unit: um/s. Average run velocity
    RUN_TIME = 0.8				# unit: s. Average run time
    TUMBLE_TIME = 0.2			# unit: s. Constant tumble time 
    TUMBLE_STEP_MAX = TUMBLE_TIME / dt
    ROT_DIF_CONST = 30			# Average directional change in 1s
    
    # Get the Ligand Concentration Profile
    # Generate the ligand concentration profile or load the file contain the
    # ligand concentration information (make sure the dimensions of the matrix 
    # are consistant with the parameters defined above).
    
    # Method 1, generate a ligand concentration profile:
    ligand_2d = np.zeros((num_x, num_y))
    for i in range(num_x):
        for j in range(num_y):
            current_asp = Green_gradient(Max, Xs, Ys, diff_rate, np.array([[i, j]]), fixed_time)
            ligand_2d[i, j] = current_asp[0]
            # ligand_2d(i, j) = 200 * (i/num_x) + 400;
            # ligand_2d(i, j) = 400;
            
    #ligand = repmat(ligand_2d, [1 1 num_steps]);
    ligand = np.tile(ligand_2d, (num_steps, 1, 1)) # make sure to make this work
    
    # Method 2, define your ligand concentration and save in a file. Load it here:
    # load data_ligand.mat
    
    # find steady-state methylation at receptor cluster
    # m = repmat(5,num_cell,1);
    m = np.tile(5, (num_cell, 1))
    
    for i in range(num_cell):
        current_asp = Green_gradient(Max, Xs, Ys, diff_rate, np.array([[x[i, 0], y[i, 0]]]), fixed_time)
        for ii in range(400):  # Stay in this one spot for 400 steps to equilibrate
            m[i], p = rapid_cell_1(current_asp, m[i], dt)

    
    # Declaration and Initialization
    mold = np.zeros((num_cell, 1))  # 0 = tumble; 1 = run.
    tumble_counter = np.zeros((num_cell, 1)) + 1  # The duration of a tumble
    direction = np.zeros((num_cell, 1))  # Cell's run direction, 0-360;
    L = np.zeros((num_cell, 1))  # Ligand concentration at the location of each cell
    
    for t in range(num_steps):
        s = ligand[t, :, :] # Get the ligand profile at current time point
        
        for n in range(num_cell):
            temp1 = int(np.ceil(x[n, 0] / dx))
            temp2 = int(np.ceil(y[n, 0] / dy))
            
            if temp1 < 0:
                temp1 = 0
            elif temp1 >= s.shape[0]:
                temp1 = s.shape[0] - 1
            
            if temp2 < 0:
                temp2 = 0
            elif temp2 >= s.shape[1]:
                temp2 = s.shape[1] - 1   
            
            L[n, 0] = s[temp1, temp2] # Get the ligand concentration at the location of each cell
        
        m, p = rapid_cell_1(L, m, dt)
        
        '''this is where the movement occures'''
        for i in range(num_cell):
            
            if mold[i, 0] == 0:  # If was tumble
                if tumble_counter[i, 0] < TUMBLE_STEP_MAX:
                    mold[i, 0] = 0
                    tumble_counter[i, 0] += 1
                else:
                    mold[i, 0] = 1
                    direction[i, 0] = np.random.rand() * 360
            else:  # If was run
                r = np.random.rand()
                if r < p[i, 0]:
                    mold[i, 0] = 1  # tumble
                else:
                    mold[i, 0] = 0  # run
                    tumble_counter[i, 0] = 1

            
            if mold[i, 0] == 1:  # run
                # Rotational diffusion
                direction[i, 0] = direction[i, 0] + np.random.randn() * ROT_DIF_CONST * np.sqrt(dt)
                # Ensure direction is within [0, 360)
                direction[i, 0] = direction[i, 0] % 360
                
                # Update position
                # x[i, 0] = x[i, 0] + RUN_VELOCITY * dt * np.cos(direction[i, 0] * np.pi / 180)
                new_x = x[i, 0] + RUN_VELOCITY * dt * np.cos(direction[i, 0] * np.pi / 180)
                # y[i, 0] = y[i, 0] + RUN_VELOCITY * dt * np.sin(direction[i, 0] * np.pi / 180)
                new_y = y[i, 0] + RUN_VELOCITY * dt * np.sin(direction[i, 0] * np.pi / 180)
                # z param -> returned as a [value]
                new_z = Green_gradient(Max, Xs, Ys, diff_rate, np.array([[new_x, new_y]]), t)
                
                if new_z[0] >= z[i, 0]:  # Move to new position only if gradient is higher or equal in case there is a lot of space that is the same gradient
                    x[i, 0] = new_x
                    y[i, 0] = new_y
                    z[i, 0] = new_z[0]
                
                if new_z[0] < 50:
                    x[i, 0] = new_x
                    y[i, 0] = new_y
                    z[i, 0] = new_z[0]
                
                if x[i, 0] < 0:
                    x[i, 0] = 0
                    #direction[n, 0] = np.floor(direction[n, 0] / 180) * 180 + 90
                    direction[i, 0] = 180 - direction[i, 0]
                elif x[i, 0] > total_x:
                    x[i, 0] = total_x
                    #direction[n, 0] = np.floor(direction[n, 0] / 180) * 180 + 90
                    direction[i, 0] = 180 - direction[i, 0]
                if y[i, 0] < 0:
                    y[i, 0] = 0
                    #direction[n, 0] = (1 - np.floor(direction[n, 0] / 270)) * 180
                    direction[i, 0] = 180 - direction[i, 0]
                elif y[i, 0] > total_y:
                    y[i, 0] = total_y
                    #direction[n, 0] = np.floor(direction[n, 0] / 90) * 180
                    direction[i, 0] = 180 - direction[i, 0]
                # Handle corner cases
                if (x[i, 0] <= 0 or x[i, 0] >= total_x) and (y[i, 0] <= 0 or y[i, 0] >= total_y):
                    mold[i, 0] = 0
                    tumble_counter[i, 0] = 1
                
        x_array[:, t + 1] = x[:, 0]
        y_array[:, t + 1] = y[:, 0]
        
        # z param
        z_array[:, t + 1] = z[:, 0]
    
    plot_traj(x_array, y_array, fsize, current_asp, dir_path, num_x, num_y, num_cell, total_x, total_y, Max, Xs, Ys, diff_rate, fixed_time)
    