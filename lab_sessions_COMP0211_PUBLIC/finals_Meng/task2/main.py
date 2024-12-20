import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper,dyn_cancel, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin, differential_drive_controller_adjusting_bearing
from simulation_and_control import differential_drive_regulation_controller,regulation_polar_coordinates,regulation_polar_coordinate_quat,wrap_angle,velocity_to_wheel_angular_velocity
import pinocchio as pin
from regulator_model import RegulatorModel

constraints_flag = True


def init_simulator(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)
    init_joint_position = sim.GetInitMotorAngles()
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    init_joint_position = sim.GetInitMotorAngles()

    # Define a goal position as a delta from the initial position
    delta_position = np.array([0.5, 0.3, -0.4, 0.2, -0.3, 0.1, 0.2])
    goal_position = init_joint_position + delta_position
    
    return sim, dyn_model, num_joints,init_joint_position, goal_position


def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    sim,dyn_model,num_joints,init_joint,goal_joints=init_simulator(conf_file_name)

    # getting time step
    time_step = sim.GetTimeStep()

    # initializing MPC
     # Define the matrices
    num_states = num_joints * 2
    num_controls = num_joints
    
    # TODO define the matrice A_cont and B_cont assuming that the system is linear
    # Construct A matrix
    # A is a block matrix of the form:
    # A = [[0, I],
    #      [0, 0]]
    #
    # where each block is a 7x7 matrix.

    A_cont = np.zeros((num_states, num_states))
    A_cont[0:num_joints, num_joints:num_states] = np.eye(num_joints)

    # Construct B matrix
    # B is a block matrix of the form:
    # B = [[0],
    #      [I]]
    #
    # where each block is 7x7.

    B_cont = np.zeros((num_states, num_controls))  # 14x7 matrix
    B_cont[num_joints:num_states, :] = np.eye(num_controls)
    
    #Verify Matrix Dimensions
    assert A_cont.shape == (num_states, num_states), "A_cont dimensions are incorrect."
    assert B_cont.shape == (num_states, num_controls), "B_cont dimensions are incorrect."
    
    # Discretise the Matrices
    
    #Horizon length
    N_mpc = 10
    # Initialize the regulator model
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states,constr_flag=constraints_flag)
    # define system matrices
    regulator.setSystemMatrices(time_step,A_cont,B_cont)

    # Define the cost matrices

    Qcoeff_joint_pos = [500] * num_controls
    Qcoeff_joint_vel = [5] * num_controls
    # making one vector Qcoeff
    Qcoeff = np.hstack((Qcoeff_joint_pos, Qcoeff_joint_vel))
    Rcoeff = [0.1]*num_controls

    regulator.setCostMatrices(Qcoeff,Rcoeff)
    # define goal position and velocity
    # goal_joints: desired positions of dimension num_joints
    # desired_velocities: desired velocities of dimension num_joints
    # For simplicity, let's assume you want zero velocities at the goal:
    desired_velocities = np.zeros(num_joints)
    #x_ref = goal_joints
    x_ref = np.hstack([goal_joints, desired_velocities])
    regulator.propagation_model_regulator_fixed_std(x_ref)
    regulator.compute_H_and_F()
    B_in = {'max': np.array([100000000000000] * num_controls), 'min': np.array([-1000000000000] * num_controls)}
    B_out = {'max': np.array([100000000]*num_states), 'min': np.array([-100000000]*num_states)}
    # creating constraints matrices
    regulator.setConstraintsMatrices(B_in,B_out)
    regulator.compute_H_and_F()

    # Define tight constraints for joints 1, 2, and 3
    joint_min_bounds = np.array([-0.5, -1.5, -10.5])
    joint_max_bounds = np.array([0.5, 1.5, 10.5])

    # For the rest of the joints, set large bounds
    large_number = 1e6
    joint_min_bounds_full = np.concatenate((joint_min_bounds, [-large_number] * (num_states - len(joint_min_bounds))))
    joint_max_bounds_full = np.concatenate((joint_max_bounds, [large_number] * (num_states - len(joint_max_bounds))))

    # Ensure the full bounds are NumPy arrays
    joint_min_bounds_full = np.array(joint_min_bounds_full)
    joint_max_bounds_full = np.array(joint_max_bounds_full)

    # Define state constraints
    B_out = {
        'min': joint_min_bounds_full,
        'max': joint_max_bounds_full
    }

    # Input constraints (assuming large bounds)
    B_in = {
        'min': np.full(num_controls, -large_number),
        'max': np.full(num_controls, large_number)
    }

    # Set constraints in the regulator
    regulator.setConstraintsMatrices(B_in, B_out)
    
    # Recompute S_bar and T_bar with the updated constraints
    regulator.propagation_model_regulator_fixed_std(x_ref)
    regulator.compute_H_and_F()
    
    print(f"S_bar shape: {regulator.S_bar.shape}")
    print(f"T_bar shape: {regulator.T_bar.shape}")
    print(f"G shape: {regulator.G.shape}")
    print(f"W shape: {regulator.W.shape}")
    print(f"S shape: {regulator.S.shape}")
    
    # Data storage
    q_mes_all, qd_mes_all, u_mpc_all, time_all = [], [], [], []

    # command object
    cmd = MotorCommands()

    current_time = 0.0
    total_time = 5.0

    while current_time < total_time:
        # Measured state 
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
        

        # Figure out what the controller should do next
        # MPC section/ low level controller section ##################################################################
       
        x0_mpc = np.vstack((q_mes, qd_mes))
        x0_mpc = x0_mpc.flatten()
        u_mpc = regulator.compute_solution(x0_mpc)

        # Store data for visualization
        q_mes_all.append(q_mes.copy())
        qd_mes_all.append(qd_mes.copy())
        u_mpc_all.append(u_mpc.copy())
        time_all.append(current_time)
        
        # Return the optimal control sequence
        u_mpc = u_mpc[:num_controls] 
        tau_cmd = dyn_cancel(dyn_model, q_mes, qd_mes, u_mpc)
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)
        sim.Step(cmd, "torque")

        ##### advance simulation ##################################################################
        sim.Step(cmd, "torque")
        time_step = sim.GetTimeStep()

        # Exit logic with 'q' key (unchanged)
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        

        # Update current time
        current_time += time_step
        print(f"Current time: {current_time}")

    # Convert lists to arrays
    q_mes_all = np.array(q_mes_all)
    qd_mes_all = np.array(qd_mes_all)
    u_mpc_all = np.array(u_mpc_all)
    time_all = np.array(time_all)
    
    # Plotting
    plt.figure(figsize=(12, 8))

    # Plot joint positions
    plt.subplot(3, 1, 1)
    for i in range(num_joints):
        plt.plot(time_all, q_mes_all[:, i], label=f'Joint {i+1}')
    for i in range(num_joints):
        plt.plot(time_all, np.ones_like(time_all)*goal_joints[i], '--', label=f'Joint {i+1} Ref')
    plt.title('Joint Positions')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [rad]')
    plt.legend()

    # Plot joint velocities
    plt.subplot(3, 1, 2)
    for i in range(num_joints):
        plt.plot(time_all, qd_mes_all[:, i], label=f'Joint {i+1}')
    plt.title('Joint Velocities')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [rad/s]')
    plt.legend()

    # Plot control inputs
    plt.subplot(3, 1, 3)
    for i in range(num_controls):
        plt.plot(time_all, u_mpc_all[:, i], label=f'Control {i+1}')
    plt.title('Control Inputs (u_mpc)')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Input')
    plt.legend()

    plt.tight_layout()
    plt.savefig('task2_1')
    plt.show()

    
    
    
if __name__ == '__main__':
    main()