import numpy as np
import math


class OmnidirectionalRobot:
    def __init__(self, x=10.0, y=10.0, theta=0.0, max_velocity=3.0, max_angular_velocity=2.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
    
    def update(self, dt, control_input):
        vx_cmd = np.clip(control_input[0], -self.max_velocity, self.max_velocity)
        vy_cmd = np.clip(control_input[1], -self.max_velocity, self.max_velocity)
        omega_cmd = np.clip(control_input[2], -self.max_angular_velocity, self.max_angular_velocity)

        self.vx = vx_cmd
        self.vy = vy_cmd
        self.omega = omega_cmd

        # Kinematic update.
        self.x += (self.vx * math.cos(self.theta) - self.vy * math.sin(self.theta)) * dt
        self.y += (self.vx * math.sin(self.theta) + self.vy * math.cos(self.theta)) * dt
        self.theta += self.omega * dt
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
    
    def get_state(self):
        return np.array([self.x, self.y, self.theta])
