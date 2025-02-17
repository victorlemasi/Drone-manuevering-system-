# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Union
import time
from enum import Enum, auto
import json
import logging
from datetime import datetime
import threading
import queue
from pathlib import Path

# Scientific and numerical libraries
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import pandas as pd
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise

# Visualization
import seaborn as sns
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px

class FlightMode(Enum):
    MANUAL = auto()
    POSITION_HOLD = auto()
    WAYPOINT = auto()
    RETURN_TO_HOME = auto()
    EMERGENCY_LANDING = auto()
    OBSTACLE_AVOIDANCE = auto()
    FOLLOW_ME = auto()
    CIRCULAR = auto()
    INSPECTION = auto()

@dataclass
class MissionParameters:
    home_position: np.ndarray
    waypoints: List[np.ndarray]
    target_velocity: float = 2.0
    hover_time: float = 2.0
    max_altitude: float = 30.0
    min_battery: float = 20.0
    return_home_battery: float = 30.0
    completion_radius: float = 0.2

class EnvironmentalConditions:
    def __init__(self):
        self.wind_speed = 0.0
        self.wind_direction = np.zeros(3)
        self.temperature = 20.0
        self.pressure = 101325.0
        self.humidity = 50.0
        self.magnetic_declination = 0.0
        
    def update(self, dt: float):
        # Update wind conditions
        self.wind_speed += np.random.normal(0, 0.1 * dt)
        self.wind_speed = max(0, min(self.wind_speed, 15.0))
        
        # Update wind direction
        self.wind_direction += np.random.normal(0, 0.1 * dt, 3)
        self.wind_direction /= np.linalg.norm(self.wind_direction + 1e-6)
        
        # Update temperature with daily cycle
        time_of_day = (time.time() % 86400) / 86400  # 0 to 1
        self.temperature = 20 + 5 * np.sin(2 * np.pi * time_of_day)
        
        # Update pressure
        self.pressure += np.random.normal(0, 10 * dt)

class LidarSensor:
    def __init__(self, num_beams: int = 32, max_range: float = 100.0, 
                 scan_rate: float = 10.0, angular_resolution: float = 0.5):
        self.num_beams = num_beams
        self.max_range = max_range
        self.scan_rate = scan_rate
        self.angular_resolution = angular_resolution
        self.last_scan_time = 0
        self.failure_probability = 0.0001
        self.is_failed = False
        
        # Calculate beam angles
        elevation_angles = np.linspace(-30, 10, num_beams)
        azimuth_angles = np.arange(0, 360, angular_resolution)
        self.elevation_angles, self.azimuth_angles = np.meshgrid(elevation_angles, azimuth_angles)

    def read(self, state: DroneState, obstacles: List[Obstacle], current_time: float) -> Dict[str, Any]:
        if self.is_failed:
            return {'status': 'failed', 'point_cloud': None}
            
        if current_time - self.last_scan_time < 1.0/self.scan_rate:
            return None
            
        if np.random.random() < self.failure_probability:
            self.is_failed = True
            return self.read(state, obstacles, current_time)
            
        point_cloud = []
        R = Rotation.from_euler('xyz', state.orientation).as_matrix()
        
        for elev, azim in zip(self.elevation_angles.flatten(), self.azimuth_angles.flatten()):
            # Calculate beam direction
            beam_dir = np.array([
                np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
                np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
                np.sin(np.radians(elev))
            ])
            beam_dir = R @ beam_dir
            
            # Find closest intersection with obstacles
            min_distance = self.max_range
            for obstacle in obstacles:
                intersection = self._ray_sphere_intersection(
                    state.position, beam_dir, obstacle.position, obstacle.radius)
                if intersection is not None:
                    min_distance = min(min_distance, intersection)
            
            if min_distance < self.max_range:
                point = state.position + beam_dir * min_distance
                intensity = 1.0 / (min_distance + 1e-6)
                point_cloud.append(np.append(point, intensity))
        
        self.last_scan_time = current_time
        return {
            'status': 'operational',
            'point_cloud': np.array(point_cloud) if point_cloud else None
        }

    def _ray_sphere_intersection(self, ray_origin: np.ndarray, ray_dir: np.ndarray, 
                               sphere_center: np.ndarray, sphere_radius: float) -> Optional[float]:
        oc = ray_origin - sphere_center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - sphere_radius * sphere_radius
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None
            
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        return t if t > 0 else None

class Magnetometer:
    def __init__(self):
        self.noise = SensorNoise(std_dev=0.1)
        self.bias = np.random.normal(0, 0.1, 3)
        self.failure_probability = 0.0001
        self.is_failed = False
        self.calibration_matrix = np.eye(3) + np.random.normal(0, 0.01, (3, 3))
        
    def read(self, state: DroneState, environment: EnvironmentalConditions) -> Dict[str, Any]:
        if self.is_failed:
            return {'status': 'failed', 'magnetic_field': None}
            
        if np.random.random() < self.failure_probability:
            self.is_failed = True
            return self.read(state, environment)
            
        # Earth's magnetic field (simplified)
        magnetic_north = np.array([
            np.cos(np.radians(environment.magnetic_declination)),
            np.sin(np.radians(environment.magnetic_declination)),
            0
        ])
        
        # Transform to body frame
        R = Rotation.from_euler('xyz', state.orientation).as_matrix()
        magnetic_body = R.T @ magnetic_north
        
        # Apply calibration, bias, and noise
        magnetic_measurement = self.calibration_matrix @ magnetic_body + self.bias
        magnetic_measurement = np.array([
            self.noise.apply(magnetic_measurement[i], self.physics.dt)
            for i in range(3)
        ])
        
        return {
            'status': 'operational',
            'magnetic_field': magnetic_measurement
        }

class DataLogger:
    def __init__(self, log_dir: str = "drone_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.current_log = None
        self.log_queue = queue.Queue()
        self.is_logging = False
        
    def start_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"flight_log_{timestamp}.csv"
        self.current_log = open(log_file, 'w')
        self.is_logging = True
        
        # Start logging thread
        threading.Thread(target=self._logging_thread, daemon=True).start()
        
    def log_data(self, data: Dict[str, Any]):
        if self.is_logging:
            self.log_queue.put(data)
            
    def stop_logging(self):
        self.is_logging = False
        if self.current_log:
            self.current_log.close()
            
    def _logging_thread(self):
        while self.is_logging:
            try:
                data = self.log_queue.get(timeout=1.0)
                # Flatten and write data
                flat_data = self._flatten_dict(data)
                if not self.current_log.tell():  # If file is empty, write header
                    self.current_log.write(','.join(flat_data.keys()) + '\n')
                self.current_log.write(','.join(map(str, flat_data.values())) + '\n')
                self.current_log.flush()
            except queue.Empty:
                continue
                
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}_{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            elif isinstance(v, np.ndarray):
                for i, val in enumerate(v):
                    items.append((f"{new_key}_{i}", val))
            else:
                items.append((new_key, v))
        return dict(items)

class MissionPlanner:
    def __init__(self, params: MissionParameters):
        self.params = params
        self.current_waypoint_idx = 0
        self.mission_start_time = None
        self.hover_start_time = None
        self.flight_mode = FlightMode.POSITION_HOLD
        
    def update(self, state: DroneState, estimated_state: DroneState, 
               environment: EnvironmentalConditions) -> Tuple[np.ndarray, float]:
        if self.mission_start_time is None:
            self.mission_start_time = time.time()
            
        current_pos = estimated_state.position
        
        # Check battery level for RTH
        if estimated_state.battery_level <= self.params.return_home_battery:
            self.flight_mode = FlightMode.RETURN_TO_HOME
            
        # Handle different flight modes
        if self.flight_mode == FlightMode.EMERGENCY_LANDING:
            target_pos = current_pos.copy()
            target_pos[2] = 0
            target_yaw = estimated_state.orientation[2]
            
        elif self.flight_mode == FlightMode.RETURN_TO_HOME:
            if np.linalg.norm(current_pos - self.params.home_position) < self.params.completion_radius:
                self.flight_mode = FlightMode.POSITION_HOLD
            target_pos = self.params.home_position
            target_yaw = 0.0
            
        elif self.flight_mode == FlightMode.WAYPOINT:
            target_pos, target_yaw = self._handle_waypoint_navigation(current_pos)
            
        elif self.flight_mode == FlightMode.CIRCULAR:
            target_pos, target_yaw = self._generate_circular_trajectory(time.time() - self.mission_start_time)
            
        else:  # POSITION_HOLD
            target_pos = current_pos
            target_yaw = estimated_state.orientation[2]
            
        return target_pos, target_yaw
        
    def _handle_waypoint_navigation(self, current_pos: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.current_waypoint_idx >= len(self.params.waypoints):
            return self.params.home_position, 0.0
            
        target = self.params.waypoints[self.current_waypoint_idx]
        distance = np.linalg.norm(current_pos - target)
        
        if distance < self.params.completion_radius:
            if self.hover_start_time is None:
                self.hover_start_time = time.time()
            elif time.time() - self.hover_start_time >= self.params.hover_time:
                self.current_waypoint_idx += 1
                self.hover_start_time = None
                
        # Calculate desired yaw to face next waypoint
        direction = target - current_pos
        target_yaw = np.arctan2(direction[1], direction[0])
        
        return target, target_yaw
        
    def _generate_circular_trajectory(self, t: float) -> Tuple[np.ndarray, float]:
        radius = 5.0
        angular_velocity = 0.5
        center = self.params.home_position
        
        x = center[0] + radius * np.cos(angular_velocity * t)
        y = center[1] + radius * np.sin(angular_velocity * t)
        z = center[2]
        
        target_pos = np.array([x, y, z])
        target_yaw = angular_velocity * t
        
        return target_pos, target_yaw

class CollisionAvoidance:
    def __init__(self):
        self.safety_margin = 1.0
        self.influence_distance = 3.0
        self.max_avoidance_force = 5.0
        
    def compute_avoidance_force(self, position: np.ndarray, velocity: np.ndarray,
                              obstacles: List[Obstacle], lidar_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
        total_force = np.zeros(3)
        
        # Use LIDAR data if available
        if lidar_data and lidar_data['status'] == 'operational' and lidar_data['point_cloud'] is not None:
            point_cloud = lidar_data['point_cloud']
            for point in point_cloud:
                distance_vec = position - point[:3]
                distance = np.linalg.norm(distance_vec)
                if distance < self.influence_distance:
                    force = self._compute_repulsive_force(distance_vec, distance, velocity)
                    total_force += force
        
        # Fallback to known obstacles
        for obstacle in obstacles:
            distance_vec = position - obstacle.position
            distance = np.linalg.norm(distance_vec)
            if distance < self.influence_distance:
                force = self._compute_repulsive_force(distance_vec, distance, velocity)
                total_force += force
        
        return np.clip(total_force, -self.max_avoidance_force, self.max_avoidance_force)
        
    def _compute_repulsive_force(self, distance_vec: np.ndarray, distance: float,
                               velocity: np.</antArtifact>
      def _compute_repulsive_force(self, distance_vec: np.ndarray, distance: float,
                               velocity: np.ndarray) -> np.ndarray:
        # Normalize distance vector
        direction = distance_vec / (distance + 1e-6)
        
        # Calculate repulsive force magnitude (increases as distance decreases)
        magnitude = (1.0 / (distance + 1e-6) - 1.0 / self.influence_distance) * (1.0 / (distance ** 2))
        
        # Add velocity-dependent component to improve dynamic avoidance
        velocity_projection = np.dot(velocity, -direction)
        if velocity_projection > 0:  # Moving towards obstacle
            magnitude += velocity_projection
            
        return direction * magnitude * self.safety_margin

class SensorSuite:
    def __init__(self):
        self.imu = IMUSensor()
        self.gps = GPSSensor()
        self.barometer = Barometer()
        self.camera = Camera()
        self.lidar = LidarSensor()
        self.magnetometer = Magnetometer()
        self.physics = DronePhysics()
        self.current_time = 0

    def update(self, state: DroneState, obstacles: List[Obstacle], environment: EnvironmentalConditions, dt: float) -> Dict[str, Any]:
        self.current_time += dt
        
        return {
            'imu': self.imu.read(state),
            'gps': self.gps.read(state, self.current_time),
            'barometer': self.barometer.read(state),
            'camera': self.camera.read(state, obstacles, self.current_time),
            'lidar': self.lidar.read(state, obstacles, self.current_time),
            'magnetometer': self.magnetometer.read(state, environment)
        }

    def reset(self):
        self.imu = IMUSensor()
        self.gps = GPSSensor()
        self.barometer = Barometer()
        self.camera = Camera()
        self.lidar = LidarSensor()
        self.magnetometer = Magnetometer()
        self.current_time = 0

class DroneSimulation:
    def __init__(self):
        self.physics = DronePhysics()
        self.controller = DroneController()
        self.sensors = SensorSuite()
        self.estimator = StateEstimator()
        self.environment = EnvironmentalConditions()
        self.collision_avoidance = CollisionAvoidance()
        self.logger = DataLogger()
        
        # Define simulation boundary
        self.boundary = (np.array([-10, -10, 0]), np.array([10, 10, 10]))
        
        # Create obstacles
        self.obstacles = [
            Obstacle(np.array([3.0, 3.0, 2.0]), 1.0),
            Obstacle(np.array([-2.0, 4.0, 3.0]), 0.8),
            Obstacle(np.array([0.0, -3.0, 4.0]), 1.2)
        ]
        
        # Initialize mission parameters
        self.mission_params = MissionParameters(
            home_position=np.zeros(3),
            waypoints=[],
            target_velocity=2.0,
            hover_time=2.0,
            max_altitude=30.0,
            min_battery=20.0,
            return_home_battery=30.0,
            completion_radius=0.2
        )
        
        self.mission_planner = MissionPlanner(self.mission_params)
        
        # Initialize drone state
        self.true_state = DroneState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.zeros(3),
            angular_velocity=np.zeros(3),
            battery_level=100.0,
            motor_speeds=np.zeros(4)
        )
        
        self.time = 0.0
        self.dt = self.physics.dt
        self.logger.start_logging()

    def run_step(self, target_position: np.ndarray, target_yaw: float):
        # Update environmental conditions
        self.environment.update(self.dt)
        
        # Get sensor readings
        sensor_data = self.sensors.update(self.true_state, self.obstacles, self.environment, self.dt)
        
        # Update state estimate
        self.estimator.predict(self.dt)
        self.estimator.update_gps(sensor_data['gps'])
        self.estimator.update_imu(sensor_data['imu'])
        self.estimator.update_barometer(sensor_data['barometer'])
        
        # Compute collision avoidance force
        avoidance_force = self.collision_avoidance.compute_avoidance_force(
            self.estimator.state.position,
            self.estimator.state.velocity,
            self.obstacles,
            sensor_data['lidar']
        )
        
        # Modify target position based on avoidance force
        modified_target = target_position + avoidance_force
        
        # Use estimated state for control
        motor_speeds = self.controller.compute_control(
            self.estimator.state,
            modified_target,
            target_yaw,
            self.dt,
            self.obstacles
        )
        
        # Update true state with physics
        self.true_state = self.physics.update_state(self.true_state, motor_speeds)
        
        # Log data
        log_data = {
            'time': self.time,
            'true_state': vars(self.true_state),
            'estimated_state': vars(self.estimator.state),
            'sensor_data': sensor_data,
            'environment': vars(self.environment),
            'target': {
                'position': target_position,
                'yaw': target_yaw
            },
            'avoidance_force': avoidance_force
        }
        self.logger.log_data(log_data)
        
        self.time += self.dt

    def run_mission(self, waypoints: List[np.ndarray], target_yaw: float = 0.0):
        self.mission_params.waypoints = waypoints
        self.mission_planner = MissionPlanner(self.mission_params)
        
        try:
            while True:
                target_position, target_yaw = self.mission_planner.update(
                    self.true_state,
                    self.estimator.state,
                    self.environment
                )
                
                self.run_step(target_position, target_yaw)
                
                # Check for mission completion or emergency conditions
                if (self.mission_planner.flight_mode == FlightMode.POSITION_HOLD and
                    np.linalg.norm(self.estimator.state.position - self.mission_params.home_position) < 
                    self.mission_params.completion_radius):
                    break
                    
                if self.true_state.battery_level < self.mission_params.min_battery:
                    print("Emergency landing: Critical battery level")
                    self.mission_planner.flight_mode = FlightMode.EMERGENCY_LANDING
                
        except Exception as e:
            print(f"Mission aborted: {str(e)}")
            self.mission_planner.flight_mode = FlightMode.EMERGENCY_LANDING
            
        finally:
            self.logger.stop_logging()
            return self.true_state, self.estimator.state

def main():
    # Create simulation instance
    sim = DroneSimulation()
    
    # Define waypoints for a simple mission
    waypoints = [
        np.array([5.0, 5.0, 3.0]),
        np.array([-3.0, 4.0, 2.0]),
        np.array([0.0, 0.0, 1.0])
    ]
    
    # Run the mission
    final_true_state, final_estimated_state = sim.run_mission(waypoints)
    
    # Plot results
    sim.plot_results()
    
    print("Mission completed!")
    print(f"Final position (true): {final_true_state.position}")
    print(f"Final position (estimated): {final_estimated_state.position}")
    print(f"Final battery level: {final_true_state.battery_level:.1f}%")

if __name__ == "__main__":
    main()
