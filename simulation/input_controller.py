import pygame

class InputController:
    def __init__(self, key_velocity=1.0, key_angular_velocity=1.0):
        self.key_velocity = key_velocity
        self.key_angular_velocity = key_angular_velocity
    
    def get_control_input(self):
        keys = pygame.key.get_pressed()
        forward_velocity = 0.0
        side_velocity = 0.0
        angular_velocity = 0.0
        
        if keys[pygame.K_w]: forward_velocity = self.key_velocity
        if keys[pygame.K_s]: forward_velocity = -self.key_velocity
        if keys[pygame.K_a]: side_velocity = self.key_velocity
        if keys[pygame.K_d]: side_velocity = -self.key_velocity
        if keys[pygame.K_q]: angular_velocity = -self.key_angular_velocity
        if keys[pygame.K_e]: angular_velocity = self.key_angular_velocity

        return [forward_velocity, side_velocity, angular_velocity]