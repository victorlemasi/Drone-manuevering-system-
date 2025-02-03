from typing import Optional, Dict, Any
import numpy as np
from scipy.spatial.transform import Rotation

class SensorNoise:
    def __init__(self, 
                 mean: float = 0.0,
                 std_dev: float = 1.0,
                 random_walk_scale: float = 0.0,
                 bias_instability: float = 0.0):
        self.mean = mean
        self.std_dev = std_dev
        self.random_walk_scale = random_walk_scale
        self.bias_instability = bias_instability
        self.current_bias = 0.0
        self.current_random_walk = 0.0

    def apply(self, value: float, dt: float) -> float:
        # Update random walk
        self.current_random_walk += np.random.normal(0, self.random_walk_scale * np.sqrt(dt))
        
        # Update bias instability using first-order Gauss-Markov process
        self.current_bias = (self.current_bias * np.exp(-dt) + 
                           np.random.normal(0, self.bias_instability * np.sqrt(1 - np.exp(-2*dt))))
        
        # Add all noise components
        noisy_value = (value + 
                      np.random.normal(self.mean, self.std_dev) + 
                      self.current_random_walk + 
                      self.current_bias)
        return noisy_value

class IMUSensor:
    def __init__(self):
        # Accelerometer noise parameters
        self.accel_noise = SensorNoise(
            std_dev=0.01,  # m/s^2
            random_walk_scale=0.001,  # m/s^2/sqrt(s)
            bias_instability=0.0005  # m/s^2
        )
        
        # Gyroscope noise parameters
        self.gyro_noise = SensorNoise(
            std_dev=0.001,  # rad/s
            random_walk_scale=0.0001,  # rad/s/sqrt(s)
            bias_instability=0.00001  # rad/s
        )
        
        self.last_update_time: Optional[float] = None
        self.failure_probability = 0.0001
        self.is_failed = False

    def read(self, state: DroneState, gravity: float = 9.81) -> Dict[str, np.ndarray]:
        if self.is_failed:
            return {
                'accelerometer': np.zeros(3),
                'gyroscope': np.zeros(3),
                'status': 'failed'
            }

        # Simulate random sensor failure
        if np.random.random() < self.failure_probability:
            self.is_failed = True
            return self.read(state)

        # Calculate true acceleration in body frame
        R = Rotation.from_euler('xyz', state.orientation).as_matrix()
        gravity_body = R.T @ np.array([0, 0, -gravity])
        accel_body = R.T @ state.velocity  # TODO: Add proper acceleration calculation
        
        # Add noise to each component
        noisy_accel = np.array([
            self.accel_noise.apply(accel_body[i] + gravity_body[i], self.physics.dt)
            for i in range(3)
        ])
        
        noisy_gyro = np.array([
            self.gyro_noise.apply(state.angular_velocity[i], self.physics.dt)
            for i in range(3)
        ])

        return {
            'accelerometer': noisy_accel,
            'gyroscope': noisy_gyro,
            'status': 'operational'
        }

class GPSSensor:
    def __init__(self):
        self.position_noise = SensorNoise(std_dev=1.0)  # 1m standard deviation
        self.velocity_noise = SensorNoise(std_dev=0.1)  # 0.1 m/s standard deviation
        self.update_rate = 10  # Hz
        self.last_update = 0
        self.signal_strength = 1.0  # 0 to 1
        self.failure_probability = 0.0001
        self.is_failed = False

    def read(self, state: DroneState, current_time: float) -> Dict[str, Any]:
        if self.is_failed:
            return {
                'position': None,
                'velocity': None,
                'status': 'failed',
                'signal_strength': 0.0
            }

        # Update only at specified rate
        if current_time - self.last_update < 1.0/self.update_rate:
            return None

        # Simulate random failures and signal interference
        if np.random.random() < self.failure_probability:
            self.is_failed = True
            return self.read(state, current_time)

        # Simulate signal strength variations
        self.signal_strength = min(1.0, max(0.0, 
            self.signal_strength + np.random.normal(0, 0.05)))

        # Add noise scaled by signal strength
        noise_scale = 1.0 + (1.0 - self.signal_strength) * 2
        position = np.array([
            self.position_noise.apply(state.position[i], self.physics.dt) * noise_scale
            for i in range(3)
        ])
        
        velocity = np.array([
            self.velocity_noise.apply(state.velocity[i], self.physics.dt) * noise_scale
            for i in range(3)
        ])

        self.last_update = current_time

        return {
            'position': position,
            'velocity': velocity,
            'status': 'operational',
            'signal_strength': self.signal_strength
        }

class Barometer:
    def __init__(self):
        self.pressure_noise = SensorNoise(
            std_dev=10.0,  # Pa
            random_walk_scale=1.0,  # Pa/sqrt(s)
        )
        self.temperature_noise = SensorNoise(std_dev=0.1)  # °C
        self.reference_pressure = 101325  # Pa at sea level
        self.failure_probability = 0.0001
        self.is_failed = False

    def read(self, state: DroneState) -> Dict[str, Any]:
        if self.is_failed:
            return {
                'pressure': None,
                'temperature': None,
                'altitude': None,
                'status': 'failed'
            }

        if np.random.random() < self.failure_probability:
            self.is_failed = True
            return self.read(state)

        # Calculate pressure based on altitude using barometric formula
        temperature = 15.0 + self.temperature_noise.apply(0, self.physics.dt)  # Ambient temperature in °C
        pressure = self.reference_pressure * np.exp(-0.0289644 * 9.81 * state.position[2] / 
                                                  (8.31447 * (temperature + 273.15)))
        
        noisy_pressure = self.pressure_noise.apply(pressure, self.physics.dt)
        
        # Calculate altitude from noisy pressure
        altitude = -(8.31447 * (temperature + 273.15) / 
                    (0.0289644 * 9.81)) * np.log(noisy_pressure / self.reference_pressure)

        return {
            'pressure': noisy_pressure,
            'temperature': temperature,
            'altitude': altitude,
            'status': 'operational'
        }

class Camera:
    def __init__(self, resolution=(640, 480), fov=90):
        self.resolution = resolution
        self.fov = fov
        self.focal_length = resolution[0] / (2 * np.tan(np.radians(fov/2)))
        self.failure_probability = 0.0001
        self.is_failed = False
        self.exposure_noise = SensorNoise(std_dev=0.1)
        self.frame_rate = 30  # fps
        self.last_frame_time = 0

    def read(self, state: DroneState, obstacles: List[Obstacle], current_time: float) -> Dict[str, Any]:
        if self.is_failed:
            return {
                'detected_obstacles': [],
                'status': 'failed'
            }

        if current_time - self.last_frame_time < 1.0/self.frame_rate:
            return None

        if np.random.random() < self.failure_probability:
            self.is_failed = True
            return self.read(state, obstacles, current_time)

        # Transform obstacles to camera frame
        R = Rotation.from_euler('xyz', state.orientation).as_matrix()
        detected_obstacles = []

        for obstacle in obstacles:
            # Transform obstacle position to camera frame
            relative_pos = obstacle.position - state.position
            obstacle_cam = R.T @ relative_pos

            # Check if obstacle is in front of camera
            if obstacle_cam[2] <= 0:
                continue

            # Project to image plane
            x = obstacle_cam[0] * self.focal_length / obstacle_cam[2]
            y = obstacle_cam[1] * self.focal_length / obstacle_cam[2]

            # Add noise to detection
            pixel_noise = np.random.normal(0, 2, 2)  # 2 pixel standard deviation
            x += pixel_noise[0]
            y += pixel_noise[1]

            # Check if within image bounds
            if (abs(x) < self.resolution[0]/2 and abs(y) < self.resolution[1]/2):
                detected_obstacles.append({
                    'position': np.array([x + self.resolution[0]/2, y + self.resolution[1]/2]),
                    'depth': obstacle_cam[2],
                    'radius_pixels': obstacle.radius * self.focal_length / obstacle_cam[2]
                })

        self.last_frame_time = current_time

        return {
            'detected_obstacles': detected_obstacles,
            'status': 'operational'
        }

class SensorSuite:
    def __init__(self):
        self.imu = IMUSensor()
        self.gps = GPSSensor()
        self.barometer = Barometer()
        self.camera = Camera()
        self.physics = DronePhysics()
        self.current_time = 0

    def update(self, state: DroneState, obstacles: List[Obstacle], dt: float) -> Dict[str, Any]:
        self.current_time += dt
        
        return {
            'imu': self.imu.read(state),
            'gps': self.gps.read(state, self.current_time),
            'barometer': self.barometer.read(state),
            'camera': self.camera.read(state, obstacles, self.current_time)
        }

    def reset(self):
        self.imu = IMUSensor()
        self.gps = GPSSensor()
        self.barometer = Barometer()
        self.camera = Camera()
        self.current_time = 0
