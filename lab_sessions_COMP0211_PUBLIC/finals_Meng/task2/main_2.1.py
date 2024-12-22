import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, dyn_cancel
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
    
    return sim, dyn_model, num_joints, init_joint_position, goal_position


def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    sim, dyn_model, num_joints, init_joint, goal_joints = init_simulator(conf_file_name)

    # Get time step
    time_step = sim.GetTimeStep()

    # Initialize MPC
    num_states = num_joints * 2
    num_controls = num_joints

    # Construct A and B matrices
    A_cont = np.zeros((num_states, num_states))
    A_cont[0:num_joints, num_joints:num_states] = np.eye(num_joints)
    B_cont = np.zeros((num_states, num_controls))
    B_cont[num_joints:num_states, :] = np.eye(num_controls)
    
    # Initialize the regulator model
    N_mpc = 10
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states, constr_flag=constraints_flag)
    regulator.setSystemMatrices(time_step, A_cont, B_cont)

    # Define cost matrices
    Qcoeff_joint_pos = [500] * num_controls
    Qcoeff_joint_vel = [5] * num_controls
    Qcoeff = np.hstack((Qcoeff_joint_pos, Qcoeff_joint_vel))
    Rcoeff = [0.5] * num_controls
    regulator.setCostMatrices(Qcoeff, Rcoeff)

    # Define goal position and velocity
    desired_velocities = np.zeros(num_joints)
    x_ref = np.hstack([goal_joints, desired_velocities])
    regulator.propagation_model_regulator_fixed_std(x_ref)
    regulator.compute_H_and_F()

    # Constraints
    joint_min_bounds = np.array([-1e6, -1e6, -1e6])
    joint_max_bounds = np.array([1e6, 1e6, 1e6])
    large_number = 1e6
    joint_min_bounds_full = np.concatenate((joint_min_bounds, [-large_number] * (num_states - len(joint_min_bounds))))
    joint_max_bounds_full = np.concatenate((joint_max_bounds, [large_number] * (num_states - len(joint_max_bounds))))

    B_out = {'min': joint_min_bounds_full, 'max': joint_max_bounds_full}
    B_in = {'min': np.full(num_controls, -large_number), 'max': np.full(num_controls, large_number)}

    regulator.setConstraintsMatrices(B_in, B_out)
    regulator.propagation_model_regulator_fixed_std(x_ref)
    regulator.compute_H_and_F()

    # Data storage
    q_mes_all, qd_mes_all, u_mpc_all, time_all = [], [], [], []

    cmd = MotorCommands()
    current_time = 0.0
    total_time = 5.0

    while current_time < total_time:
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        x0_mpc = np.hstack((q_mes, qd_mes))
        u_mpc = regulator.compute_solution(x0_mpc)

        q_mes_all.append(q_mes.copy())
        qd_mes_all.append(qd_mes.copy())
        u_mpc_all.append(u_mpc.copy())
        time_all.append(current_time)

        u_mpc = u_mpc[:num_controls]
        tau_cmd = dyn_cancel(dyn_model, q_mes, qd_mes, u_mpc)
        cmd.SetControlCmd(tau_cmd, ["torque"] * 7)
        sim.Step(cmd, "torque")

        current_time += time_step
        print(f"Current time: {current_time}")

    q_mes_all = np.array(q_mes_all)
    qd_mes_all = np.array(qd_mes_all)
    u_mpc_all = np.array(u_mpc_all)
    time_all = np.array(time_all)

    # Plot all joint positions
    plt.figure()
    for i in range(num_joints):
        plt.plot(time_all, q_mes_all[:, i], label=f'Joint {i+1} Position')
        plt.plot(time_all, np.ones_like(time_all) * goal_joints[i], '--', label=f'Joint {i+1} Reference')
    plt.title('Joint Positions vs. Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [rad]')
    plt.legend()
    plt.savefig('all_joint_positions.png')
    plt.close()

    # Plot all joint velocities
    plt.figure()
    for i in range(num_joints):
        plt.plot(time_all, qd_mes_all[:, i], label=f'Joint {i+1} Velocity')
    plt.title('Joint Velocities vs. Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [rad/s]')
    plt.legend()
    plt.savefig('all_joint_velocities.png')
    plt.close()

    # Plot all control inputs
    plt.figure()
    for i in range(num_controls):
        plt.plot(time_all, u_mpc_all[:, i], label=f'Control {i+1}')
    plt.title('Control Inputs vs. Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Input')
    plt.legend()
    plt.savefig('all_control_inputs.png')
    plt.close()


if __name__ == '__main__':
    main()
