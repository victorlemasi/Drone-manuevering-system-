import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple
import time

@dataclass
class DroneState:
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    orientation: np.ndarray  # [roll, pitch, yaw]
    angular_velocity: np.ndarray  # [wx, wy, wz]

class DronePhysics:
    def __init__(self):
        # Drone physical parameters
        self.mass = 1.5  # kg
        self.gravity = 9.81  # m/s^2
        self.max_thrust = 30.0  # N
        self.drag_coefficient = 0.1
        self.moment_of_inertia = np.array([0.1, 0.1, 0.15])  # kg*m^2
        self.dt = 0.01  # simulation time step

    def update_state(self, state: DroneState, thrust: float, torques: np.ndarray) -> DroneState:
        # Convert orientation to rotation matrix
        R = self._euler_to_rotation_matrix(state.orientation)
        
        # Calculate forces
        thrust_force = R @ np.array([0, 0, thrust])
        drag_force = -self.drag_coefficient * state.velocity
        gravity_force = np.array([0, 0, -self.mass * self.gravity])
        total_force = thrust_force + drag_force + gravity_force
        
        # Update linear motion
        acceleration = total_force / self.mass
        new_velocity = state.velocity + acceleration * self.dt
        new_position = state.position + new_velocity * self.dt
        
        # Update angular motion
        angular_acceleration = torques / self.moment_of_inertia
        new_angular_velocity = state.angular_velocity + angular_acceleration * self.dt
        new_orientation = state.orientation + new_angular_velocity * self.dt
        
        # Normalize angles to [-pi, pi]
        new_orientation = np.mod(new_orientation + np.pi, 2 * np.pi) - np.pi
        
        return DroneState(
            position=new_position,
            velocity=new_velocity,
            orientation=new_orientation,
            angular_velocity=new_angular_velocity
        )

    def _euler_to_rotation_matrix(self, euler_angles: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = euler_angles
        
        # Roll rotation
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Pitch rotation
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Yaw rotation
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx

class DroneController:
    def __init__(self):
        # PID gains for position control
        self.kp_pos = np.array([2.0, 2.0, 2.0])
        self.ki_pos = np.array([0.1, 0.1, 0.1])
        self.kd_pos = np.array([1.0, 1.0, 1.0])
        
        # PID gains for attitude control
        self.kp_att = np.array([5.0, 5.0, 5.0])
        self.ki_att = np.array([0.1, 0.1, 0.1])
        self.kd_att = np.array([1.0, 1.0, 1.0])
        
        # Integral terms
        self.pos_integral = np.zeros(3)
        self.att_integral = np.zeros(3)

    def compute_control(self, current_state: DroneState, target_position: np.ndarray, 
                       target_yaw: float) -> Tuple[float, np.ndarray]:
        # Position control
        pos_error = target_position - current_state.position
        self.pos_integral += pos_error * 0.01  # dt = 0.01
        pos_derivative = -current_state.velocity
        
        desired_acceleration = (
            self.kp_pos * pos_error +
            self.ki_pos * self.pos_integral +
            self.kd_pos * pos_derivative
        )
        
        # Convert desired acceleration to thrust and attitude
        thrust_magnitude = np.linalg.norm(desired_acceleration) * 1.5  # mass
        thrust_direction = desired_acceleration / (np.linalg.norm(desired_acceleration) + 1e-6)
        
        # Compute desired orientation
        desired_roll = np.arcsin(thrust_direction[1])
        desired_pitch = -np.arctan2(thrust_direction[0], thrust_direction[2])
        desired_yaw = target_yaw
        
        desired_orientation = np.array([desired_roll, desired_pitch, desired_yaw])
        
        # Attitude control
        att_error = desired_orientation - current_state.orientation
        att_error = np.mod(att_error + np.pi, 2 * np.pi) - np.pi  # Normalize to [-pi, pi]
        
        self.att_integral += att_error * 0.01
        att_derivative = -current_state.angular_velocity
        
        torques = (
            self.kp_att * att_error +
            self.ki_att * self.att_integral +
            self.kd_att * att_derivative
        )
        
        return thrust_magnitude, torques

class DroneSimulation:
    def __init__(self):
        self.physics = DronePhysics()
        self.controller = DroneController()
        self.state_history = []
        
        # Initial state
        self.state = DroneState(
            position=np.array([0., 0., 0.]),
            velocity=np.array([0., 0., 0.]),
            orientation=np.array([0., 0., 0.]),
            angular_velocity=np.array([0., 0., 0.])
        )

    def run_simulation(self, waypoints: List[np.ndarray], simulation_time: float):
        num_steps = int(simulation_time / self.physics.dt)
        current_waypoint_idx = 0
        
        for _ in range(num_steps):
            # Store current state
            self.state_history.append([
                self.state.position.copy(),
                self.state.orientation.copy()
            ])
            
            # Check if we reached current waypoint
            distance_to_waypoint = np.linalg.norm(
                self.state.position - waypoints[current_waypoint_idx][:3])
            if distance_to_waypoint < 0.5 and current_waypoint_idx < len(waypoints) - 1:
                current_waypoint_idx += 1
            
            # Compute control inputs
            target_position = waypoints[current_waypoint_idx][:3]
            target_yaw = waypoints[current_waypoint_idx][3]
            thrust, torques = self.controller.compute_control(
                self.state, target_position, target_yaw)
            
            # Update state
            self.state = self.physics.update_state(self.state, thrust, torques)

    def visualize_trajectory(self):
        history = np.array(self.state_history)
        positions = np.array([state[0] for state in history])
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Trajectory')
        
        # Plot start and end points
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  color='g', marker='o', s=100, label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  color='r', marker='o', s=100, label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create waypoints [x, y, z, yaw]
    waypoints = [
        np.array([0., 0., 2., 0.]),        # Takeoff
        np.array([5., 0., 2., np.pi/2]),   # Move forward and rotate
        np.array([5., 5., 2., np.pi]),     # Move right
        np.array([0., 5., 2., -np.pi/2]),  # Move back
        np.array([0., 0., 2., 0.]),        # Return to start
        np.array([0., 0., 0., 0.])         # Land
    ]
    
    # Run simulation
    sim = DroneSimulation()
    sim.run_simulation(waypoints, simulation_time=20.0)
    sim.visualize_trajectory()
