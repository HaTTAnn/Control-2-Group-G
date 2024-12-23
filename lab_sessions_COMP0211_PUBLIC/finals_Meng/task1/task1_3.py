import numpy as np
import matplotlib.pyplot as plt
from observer import Observer
from dc_model import SysDyn
from regulator_model import RegulatorModel
from scipy.linalg import solve_discrete_are, inv
from numpy.linalg import matrix_rank

# Motor Parameters
J = 0.01      # Inertia (kg*m^2)
b = 0.1       # Friction coefficient (N*m*s)
K_t = 1       # Motor torque constant (N*m/A)
K_e = 0.01    # Back EMF constant (V*s/rad)
R_a = 1.0     # Armature resistance (Ohm)
L_a = 0.001   # Armature inductance (H)

# Desired Eigenvalues for Observer
lambda_1 = -11
lambda_2 = -600

# Simulation Parameters
t_start = 0.0
t_end = 0.05
dt = 0.00001  # Smaller time step for Euler integration
time = np.arange(t_start, t_end, dt)
num_steps = len(time)

# Initial Conditions for the System [omega, I_a]
x_init = np.array([0.0, 0.0])  # True system state [omega, I_a]
motor_model = SysDyn(J, b, K_t, K_e, R_a, L_a, dt, x_init)
motor_model.checkControlabilityContinuos()

# Initial Conditions for the Observer [omega_hat, I_a_hat]
x_hat_init = np.array([0.0, 0.0])  # Initial guess for the observer state [omega_hat, I_a_hat]
observer = Observer(motor_model.A, motor_model.B, motor_model.C, dt, x_hat_init)

# Compute the observer gain L
# Place the eigenvalues of (A - L C) at desired locations
observer.ComputeObserverGains(lambda_1, lambda_2)

# Initializing MPC
num_states = 2
num_controls = 1
constraints_flag = False

# ATTENTION! here we do not use the MPC but we only use its function to compute A,B,Q and R 
# Horizon length
N_mpc = 10
regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states, constr_flag=constraints_flag)
# define system matrices
regulator.setSystemMatrices(dt, motor_model.getA(), motor_model.getB())
# check the stability of the discretized system
regulator.checkStability()
# check controlability of the discretized system
regulator.checkControllabilityDiscrete()
# Define the cost matrices
Qcoeff = [1000, 0.0]
Rcoeff = [0.06] * num_controls

regulator.setCostMatrices(Qcoeff, Rcoeff)

Q, R = regulator.getCostMatrices()
A = regulator.getDiscreteA()
B = regulator.getDiscreteB()

# Desired state x_d
x_ref = np.array([100, 0])

# Solve the discrete-time algebraic Riccati equation
P = solve_discrete_are(A, B, Q, R)

# Calculate the optimal control law K
K = inv(R + B.T @ P @ B) @ B.T @ P @ A


# Calculate the feedforward term
# Compute pseudoinverse of B
#A_cont=motor_model.getA()
#B_cont=motor_model.getB()
B_pinv = np.linalg.pinv(B)  # Result is a (1x2) matrix
# Compute Delta x (this is for the discrete system)
delta_x = A @ x_ref # Result is a (2x1) vector
# Compute u_ff
u_ff = - B_pinv @ delta_x 

# Preallocate arrays for storing results
omega = np.zeros(num_steps)
I_a = np.zeros(num_steps)
hat_omega = np.zeros(num_steps)
hat_I_a = np.zeros(num_steps)
T_m_true = np.zeros(num_steps)
T_m_estimated = np.zeros(num_steps)
V_terminal = np.zeros(num_steps)
V_terminal_hat = np.zeros(num_steps)
x_i_k = np.zeros(num_states)
x_i_all = np.zeros((num_steps, num_states))

x_cur = x_init
x_hat_cur = x_hat_init

for k in range(num_steps):
    t = time[k]
    V_a = -K @ (x_cur - x_ref) + u_ff
    cur_y = motor_model.step(V_a)
    x_cur = motor_model.getCurrentState()
    V_terminal[k] = cur_y.item()  # Ensure single-element array is properly handled
    x_hat_cur, y_hat_cur = observer.update(V_a, cur_y)

    omega[k] = x_cur[0]
    I_a[k] = x_cur[1]
    hat_omega[k] = x_hat_cur[0]
    hat_I_a[k] = x_hat_cur[1]
    T_m_true[k] = K_t * I_a[k]
    T_m_estimated[k] = K_t * hat_I_a[k]
    V_terminal_hat[k] = y_hat_cur.item()  # Ensure single-element array is properly handled

# Function to calculate settling time
def calculate_settling_time(time, signal, threshold=0.02):
    steady_state = signal[-1]
    lower_bound = steady_state * (1 - threshold)
    upper_bound = steady_state * (1 + threshold)
    
    for i in reversed(range(len(signal))):
        if not (lower_bound <= signal[i] <= upper_bound):
            if i + 1 < len(time):
                return time[i + 1]  # Settling time is the next time step
            else:
                return time[-1]  # Avoid index out of bounds
    return time[-1]

# Function to calculate overshoot
def calculate_overshoot(signal):
    steady_state = signal[-1]
    peak_value = np.max(signal)
    overshoot = (peak_value - steady_state) / steady_state * 100
    return overshoot

# Function to calculate steady-state error
def calculate_steady_state_error(signal, reference):
    steady_state = signal[-1]
    error = reference - steady_state
    return error

# Reference values
omega_ref = 100  

# Calculate metrics
settling_time_omega = calculate_settling_time(time, omega)
overshoot_omega = calculate_overshoot(omega)
steady_state_error_omega = calculate_steady_state_error(omega, omega_ref)

# Print results
print("Settling Time for Angular Velocity:", settling_time_omega, "s")
print("Overshoot for Angular Velocity:", overshoot_omega, "%")
print("Steady-State Error for Angular Velocity:", steady_state_error_omega)



# Plotting Results
plt.figure(figsize=(12, 10))

plt.subplot(5, 1, 1)
plt.plot(time, omega, label=r'True $\omega$ (rad/s)')
plt.plot(time, hat_omega, '--', label=r'Estimated $\hat{\omega}$ (rad/s)')
plt.title('Angular Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 2)
plt.plot(time, I_a, label=r'True $I_a$ (A)')
plt.plot(time, hat_I_a, '--', label=r'Estimated $\hat{I}_a$ (A)')
plt.title('Armature Current')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 3)
plt.plot(time, T_m_true, label=r'True $T_m$ (N*m)')
plt.plot(time, T_m_estimated, '--', label=r'Estimated $\hat{T}_m$ (N*m)')
plt.title('Motor Torque')
plt.xlabel('Time (s)')
plt.ylabel('Torque (N*m)')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 4)
plt.plot(time, V_terminal, label=r'Measured $V_{terminal}$ (V)')
plt.plot(time, V_terminal_hat, '--', label=r'Estimated $\hat{V}_{terminal}$ (V)')
plt.title('Terminal Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)

# plt.subplot(5, 1, 5)
# plt.plot(time, x_i_all[:, 0], label=r'Integral of Output Error $\int e_1$')
# plt.title('Integral of Output Error')
# plt.xlabel('Time (s)')
# plt.ylabel('Integral of Output Error')
# plt.legend()
# plt.grid(True)

plt.tight_layout()
plt.show()
