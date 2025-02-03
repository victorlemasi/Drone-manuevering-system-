import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional
import time
from scipy.spatial import KDTree
from queue import PriorityQueue
from matplotlib.animation import FuncAnimation

@dataclass
class Obstacle:
    position: np.ndarray  # Center position [x, y, z]
    radius: float        # Radius of the obstacle

@dataclass
class DroneState:
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    orientation: np.ndarray  # [roll, pitch, yaw]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    battery_level: float = 100.0  # Battery percentage
    motor_speeds: np.ndarray = np.zeros(4)  # Individual motor speeds

class DronePhysics:
    def __init__(self):
        # Drone physical parameters
        self.mass = 1.5  # kg
        self.gravity = 9.81  # m/s^2
        self.max_thrust = 30.0  # N
        self.drag_coefficient = 0.1
        self.moment_of_inertia = np.array([0.1, 0.1, 0.15])  # kg*m^2
        self.dt = 0.01  # simulation time step
        
        # Motor parameters
        self.num_motors = 4
        self.motor_distance = 0.2  # Distance from center to motor (m)
        self.max_motor_speed = 2000  # RPM
        self.motor_thrust_coefficient = 1e-5
        self.motor_torque_coefficient = 1e-7
        
        # Battery parameters
        self.battery_capacity = 5000  # mAh
        self.battery_voltage = 11.1  # V
        self.power_consumption_coefficient = 0.01

    def update_state(self, state: DroneState, motor_speeds: np.ndarray) -> DroneState:
        # Calculate thrust and torques from motor speeds
        thrusts = self.motor_thrust_coefficient * motor_speeds**2
        total_thrust = np.sum(thrusts)
        
        # Calculate torques
        torques = np.zeros(3)
        for i in range(self.num_motors):
            angle = i * (2 * np.pi / self.num_motors)
            torques[0] += thrusts[i] * self.motor_distance * np.sin(angle)  # Roll
            torques[1] += thrusts[i] * self.motor_distance * np.cos(angle)  # Pitch
            torques[2] += ((-1)**i) * self.motor_torque_coefficient * motor_speeds[i]**2  # Yaw
        
        # Convert orientation to rotation matrix
        R = self._euler_to_rotation_matrix(state.orientation)
        
        # Calculate forces
        thrust_force = R @ np.array([0, 0, total_thrust])
        air_resistance = -self.drag_coefficient * state.velocity * np.abs(state.velocity)
        gravity_force = np.array([0, 0, -self.mass * self.gravity])
        total_force = thrust_force + air_resistance + gravity_force
        
        # Add wind disturbance
        wind = self._generate_wind_disturbance()
        acceleration = total_force / self.mass + wind
        
        # Update linear motion
        new_velocity = state.velocity + acceleration * self.dt
        new_position = state.position + new_velocity * self.dt
        
        # Update angular motion
        angular_acceleration = torques / self.moment_of_inertia
        new_angular_velocity = state.angular_velocity + angular_acceleration * self.dt
        new_orientation = state.orientation + new_angular_velocity * self.dt
        
        # Update battery
        power_consumption = np.sum(motor_speeds**2) * self.power_consumption_coefficient
        new_battery_level = state.battery_level - power_consumption * self.dt
        
        return DroneState(
            position=new_position,
            velocity=new_velocity,
            orientation=np.mod(new_orientation + np.pi, 2 * np.pi) - np.pi,
            angular_velocity=new_angular_velocity,
            battery_level=max(0.0, new_battery_level),
            motor_speeds=motor_speeds
        )

    def _generate_wind_disturbance(self) -> np.ndarray:
        wind_strength = 0.1
        return np.random.normal(0, wind_strength, 3)

    def _euler_to_rotation_matrix(self, euler_angles: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = euler_angles
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx

class PathPlanner:
    def __init__(self, obstacles: List[Obstacle], boundary: Tuple[np.ndarray, np.ndarray]):
        self.obstacles = obstacles
        self.min_bounds, self.max_bounds = boundary
        self.safety_margin = 0.5
        
    def plan_path(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        def heuristic(pos):
            return np.linalg.norm(pos - goal)
        
        def is_valid_position(pos):
            if not all(self.min_bounds <= pos) or not all(pos <= self.max_bounds):
                return False
            
            for obs in self.obstacles:
                if np.linalg.norm(pos - obs.position) < (obs.radius + self.safety_margin):
                    return False
            return True
        
        open_set = PriorityQueue()
        open_set.put((0, tuple(start)))
        came_from = {tuple(start): None}
        cost_so_far = {tuple(start): 0}
        
        movements = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    if x == 0 and y == 0 and z == 0:
                        continue
                    movements.append(np.array([x, y, z]))
        
        step_size = 0.5
        
        while not open_set.empty():
            current = np.array(open_set.get()[1])
            
            if np.linalg.norm(current - goal) < step_size:
                path = []
                while tuple(current) in came_from:
                    path.append(current)
                    current = came_from[tuple(current)]
                path.reverse()
                return self._smooth_path(path)
            
            for movement in movements:
                next_pos = current + movement * step_size
                
                if not is_valid_position(next_pos):
                    continue
                
                new_cost = cost_so_far[tuple(current)] + np.linalg.norm(movement) * step_size
                
                if tuple(next_pos) not in cost_so_far or new_cost < cost_so_far[tuple(next_pos)]:
                    cost_so_far[tuple(next_pos)] = new_cost
                    priority = new_cost + heuristic(next_pos)
                    open_set.put((priority, tuple(next_pos)))
                    came_from[tuple(next_pos)] = current
        
        return [start, goal]

    def _smooth_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        if len(path) < 3:
            return path
        
        smoothed = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            for look_ahead in range(len(path) - 1, current_idx, -1):
                if self._is_clear_path(path[current_idx], path[look_ahead]):
                    smoothed.append(path[look_ahead])
                    current_idx = look_ahead
                    break
            current_idx += 1
        
        return smoothed

    def _is_clear_path(self, start: np.ndarray, end: np.ndarray) -> bool:
        direction = end - start
        distance = np.linalg.norm(direction)
        direction = direction / distance
        
        num_checks = int(distance / 0.1)
        for i in range(num_checks):
            point = start + direction * (i * distance / num_checks)
            if not all(self.min_bounds <= point) or not all(point <= self.max_bounds):
                return False
            for obs in self.obstacles:
                if np.linalg.norm(point - obs.position) < (obs.radius + self.safety_margin):
                    return False
        return True

class DroneController:
    def __init__(self):
        # Position control gains
        self.kp_pos = np.array([2.0, 2.0, 2.0])
        self.ki_pos = np.array([0.1, 0.1, 0.1])
        self.kd_pos = np.array([1.0, 1.0, 1.0])
        
        # Attitude control gains
        self.kp_att = np.array([5.0, 5.0, 5.0])
        self.ki_att = np.array([0.1, 0.1, 0.1])
        self.kd_att = np.array([1.0, 1.0, 1.0])
        
        # Control limits
        self.max_tilt = np.pi/4
        self.max_yaw_rate = np.pi
        
        # Integral terms
        self.pos_integral = np.zeros(3)
        self.att_integral = np.zeros(3)
        self.integral_limit = 10.0
        
        # Previous errors
        self.prev_pos_error = np.zeros(3)
        self.prev_att_error = np.zeros(3)
        
        # Obstacle avoidance parameters
        self.obstacle_influence_radius = 2.0
        self.obstacle_repulsion_gain = 1.0

    def compute_control(self, current_state: DroneState, target_position: np.ndarray,
                       target_yaw: float, dt: float, obstacles: List[Obstacle]) -> np.ndarray:
        # Add obstacle avoidance
        avoidance_force = self._compute_obstacle_avoidance(current_state.position, obstacles)
        modified_target = target_position + avoidance_force
        
        # Emergency landing if battery low
        if current_state.battery_level < 10.0:
            modified_target[2] = 0.0
        
        # Position control
        pos_error = modified_target - current_state.position
        self.pos_integral += pos_error * dt
        self.pos_integral = np.clip(self.pos_integral, -self.integral_limit, self.integral_limit)
        
        pos_derivative = (pos_error - self.prev_pos_error) / dt
        self.prev_pos_error = pos_error
        
        desired_acceleration = (
            self.kp_pos * pos_error +
            self.ki_pos * self.pos_integral +
            self.kd_pos * pos_derivative
        )
        
        # Convert to thrust and attitude
        thrust_direction = desired_acceleration / (np.linalg.norm(desired_acceleration) + 1e-6)
        
        desired_roll = np.clip(np.arcsin(thrust_direction[1]), -self.max_tilt, self.max_tilt)
        desired_pitch = np.clip(-np.arctan2(thrust_direction[0], thrust_direction[2]),
                              -self.max_tilt, self.max_tilt)
        desired_yaw = target_yaw
        
        desired_orientation = np.array([desired_roll, desired_pitch, desired_yaw])
        
        # Attitude control
        att_error = desired_orientation - current_state.orientation
        att_error = np.mod(att_error + np.pi, 2 * np.pi) - np.pi
        
        self.att_integral += att_error * dt
        self.att_integral = np.clip(self.att_integral, -self.integral_limit, self.integral_limit)
        
        att_derivative = (att_error - self.prev_att_error) / dt
        self.prev_att_error = att_error
        
        torques = (
            self.kp_att * att_error +
            self.ki_att * self.att_integral +
            self.kd_att * att_derivative
        )
        
        return self._compute_motor_speeds(
            np.linalg.norm(desired_acceleration) * 1.5,
            torques
        )

    def _compute_obstacle_avoidance(self, position: np.ndarray, obstacles: List[Obstacle]) -> np.ndarray:
        total_force = np.zeros(3)
        
        for obstacle in obstacles:
            distance_vec = position - obstacle.position
            distance = np.linalg.norm(distance_vec)
            
            if distance < self.obstacle_influence_radius:
                force_magnitude = self.obstacle_repulsion_gain * (
                    1.0 / distance - 1.0 / self.obstacle_influence_radius
                ) * 1.0 / (distance ** 2)
                
                force_direction = distance_vec / distance
                total_force += force_magnitude * force_direction
        
        return total_force

    def _compute_motor_speeds(self, thrust: float, torques: np.ndarray) -> np.ndarray:
        mixer_matrix = np.array([
            [1, -1, -1, 1],
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1]
        ]) * 0.25
        
        control_vector = np.array([thrust, torques[0], torques[1], torques[2]])
        motor_speeds = mixer_matrix @ control_vector
        motor_speeds = np.sqrt(np.maximum(0, motor_speeds))
        return np.clip(motor_speeds, 0, 2000)

class DroneSimulation:
    def __init__(self):
        self.physics = DronePhysics()
        self.controller = DroneController()
        self.state_history = []
        self.battery_history = []
        
        # Define simulation boundary
        self.boundary = (np.array([-10, -10, 0]), np.array([10, 10, 10]))
        
        # Create obstacles
        self.obstacles = [
            Obstacle(np.array([3.0, 3.0, 2.0]), 1.0),
