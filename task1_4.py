import numpy as np
import matplotlib.pyplot as plt
from observer import Observer
from dc_model import SysDyn
from regulator_model import RegulatorModel
from scipy.linalg import solve_discrete_are, inv

# Function to calculate settling time
def calculate_settling_time(time, signal, threshold=0.02):
    steady_state = signal[-1]
    lower_bound = steady_state * (1 - threshold)
    upper_bound = steady_state * (1 + threshold)
    for i in reversed(range(len(signal))):
        if not (lower_bound <= signal[i] <= upper_bound):
            return time[i + 1] if i + 1 < len(time) else time[-1]
    return time[-1]

# Function to calculate overshoot
def calculate_overshoot(signal, reference):
    peak_value = np.max(signal)
    overshoot = ((peak_value - reference) / reference) * 100
    return overshoot

# Function to calculate steady-state error
def calculate_steady_state_error(signal, reference):
    steady_state = signal[-1]
    error = reference - steady_state
    return error

def simulation(variation,control_method="LQR"):
    # Motor Parameters
    J = 0.01      # Inertia (kg*m^2)
    b = 0.1       # Friction coefficient (N*m*s)
    K_t = 1    # Motor torque constant (N*m/A)
    K_e = 0.01 * variation   # Back EMF constant (V*s/rad)
    R_a = 1.0  # Armature resistance (Ohm)
    L_a = 0.001  # Armature inductance (H)

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
    observer.ComputeObserverGains(lambda_1, lambda_2)

    # Preallocate arrays for storing results
    omega = np.zeros(num_steps)
    I_a = np.zeros(num_steps)
    hat_omega = np.zeros(num_steps)
    hat_I_a = np.zeros(num_steps)
    T_m_true = np.zeros(num_steps)
    T_m_estimated = np.zeros(num_steps)
    V_terminal = np.zeros(num_steps)
    V_terminal_hat = np.zeros(num_steps)


    x_cur = x_init
    x_hat_cur = x_hat_init

    if control_method == "LQR" :
        # Initialize LQR
        Qcoeff = [1000, 0.0]
        Rcoeff = [0.06]
        num_states = 2
        num_controls = 1
        N_mpc = 10
        regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)
        regulator.setSystemMatrices(dt, motor_model.getA(), motor_model.getB())
        regulator.setCostMatrices(Qcoeff, Rcoeff)
        Q, R = regulator.getCostMatrices()
        A = regulator.getDiscreteA()
        B = regulator.getDiscreteB()
        x_ref = np.array([5, 0])
        P = solve_discrete_are(A, B, Q, R)
        K = inv(R + B.T @ P @ B) @ B.T @ P @ A
        B_pinv = np.linalg.pinv(B)
        delta_x = A @ x_ref
        u_ff = -B_pinv @ delta_x

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
            

    else:
        # Initialize MPC
        num_states = 2
        num_controls = 1
        constraints_flag = False
        N_mpc = 10
        regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states, constr_flag=constraints_flag)
        regulator.setSystemMatrices(dt, motor_model.getA(), motor_model.getB())
        Qcoeff = [1000.0, 0.0]
        Rcoeff = [0.06] * num_controls
        regulator.setCostMatrices(Qcoeff, Rcoeff)
        x_ref = np.array([-5, 0])
        regulator.propagation_model_regulator_fixed_std(x_ref)
        B_in = {'max': np.array([1e12] * num_controls), 'min': np.array([-1e12] * num_controls)}
        B_out = {'max': np.array([1e8, 1e9]), 'min': np.array([-1e8, -1e9])}
        regulator.setConstraintsMatrices(B_in, B_out)
        regulator.compute_H_and_F()
        for k in range(num_steps):
            # Time stamp
            t = time[k]
            print("Current time: ", t)
            
            # compute control input
            u_mpc = regulator.compute_solution(x_hat_cur)
            V_a = u_mpc[0]

            cur_y = motor_model.step(V_a)
            # IMPORTANT remember that X_cur is the true state of the but it cannot be accessed in the real world
            x_cur = motor_model.getCurrentState()
            
            # Output measurement (Terminal Voltage)
            V_terminal[k] = cur_y
            
            x_hat_cur,y_hat_cur = observer.update(V_a, cur_y)

            # Store results
            omega[k] = x_cur[0]
            I_a[k] = x_cur[1]
            hat_omega[k] = x_hat_cur[0]
            hat_I_a[k] = x_hat_cur[1]
            T_m_true[k] = K_t * I_a[k]
            T_m_estimated[k] = K_t * hat_I_a[k]
            V_terminal_hat[k] = y_hat_cur

    omega_ref = 5  # Desired angular velocity reference
    # Calculate performance metrics
    settling_time = calculate_settling_time(time, omega)
    overshoot = calculate_overshoot(omega, omega_ref)
    steady_state_error = calculate_steady_state_error(omega, omega_ref)

    return V_terminal,omega,I_a,hat_omega,hat_I_a,T_m_true,T_m_estimated,V_terminal_hat,time,settling_time,overshoot,steady_state_error

results = []
variations = [0.8, 1, 1.2]
for variation in variations:
    results.append(simulation(variation,"LQR"))

# Define variation labels
variation_labels = ["-20%", "0%", "20%"]
# Plotting all results
plt.figure(figsize=(12, 20))
for i, (V_terminal,omega,I_a,hat_omega,hat_I_a,T_m_true,T_m_estimated,V_terminal_hat,time,settling_time,overshoot,steady_state_error) in enumerate(results):
    print(f"  Settling Time: {settling_time:.4f} s")
    print(f"  Overshoot: {overshoot:.2f} %")
    print(f"  Steady-State Error: {steady_state_error:.4f}")
    plt.subplot(4, 1, 1)
    plt.plot(time, omega, label=f'True $\\omega$ (rad/s), {variation_labels[i]}')
    plt.plot(time, hat_omega, '--', label=f'Estimated $\\hat{{\\omega}}$ (rad/s), {variation_labels[i]}')

    plt.subplot(4, 1, 2)
    plt.plot(time, I_a, label=f'True $I_a$ (A), {variation_labels[i]}')
    plt.plot(time, hat_I_a, '--', label=f'Estimated $\\hat{{I}}_a$ (A), {variation_labels[i]}')

    plt.subplot(4, 1, 3)
    plt.plot(time, T_m_true, label=f'True $T_m$ (N*m), {variation_labels[i]}')
    plt.plot(time, T_m_estimated, '--', label=fr'Estimated $\hat{{T}}_m$ (N*m), {variation_labels[i]}')

    plt.subplot(4, 1, 4)
    plt.plot(time, V_terminal, label=f'Measured $V_{{terminal}}$ (V), {variation_labels[i]}')
    plt.plot(time, V_terminal_hat, '--', label=f'Estimated $\\hat{{V}}_{{terminal}}$ (V), {variation_labels[i]}')

plt.subplot(4, 1, 1)
plt.title('Angular Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.title('Armature Current')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.title('Motor Torque')
plt.xlabel('Time (s)')
plt.ylabel('Torque (N*m)')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)
plt.title('Terminal Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 20))

