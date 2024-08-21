import math
import random
import numpy as np

def normalize_angle(angle):
    """
    Convert the given angle to the range [-π, +π].
    """
    # Normalize to [0, 2π)
    angle = angle % (2 * math.pi)
    
    # Convert to [-π, +π]
    if angle > math.pi:
        angle -= 2 * math.pi
    
    return angle

class Robot:
    def __init__(self, x = 0.0, y = 0.0, theta = 0.0, length = 4.0, noise = True):
        self.x = x
        self.y = y
        self.theta = theta
        self.length = length

        self.noise = noise

        #noises
        self.forward_noise = 0.05 if noise else 0.0  # standard deviation for movement noise
        self.steering_noise = 0.05 if noise else 0.0 # standard deviation for steering noise
        self.sensor_noise = 15.0  if noise else 0.0 # standard deviation for sensor noise

        #ориентиры по углам карты
        self.landmarks = [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]]

    def move(self, u):
        # Parsing signal u
        alpha = u[0]
        distance = u[1]

        # Adding noise
        if self.noise:
            alpha += random.gauss(0.0, self.steering_noise)
            distance += random.gauss(0.0, self.forward_noise)

        # Calculations
        beta = (distance / self.length) * math.tan(alpha)

        # Updated robot position
        if abs(beta) < 0.001:  # Straight
            self.x += distance * math.cos(self.theta)
            self.y += distance * math.sin(self.theta)
        else:  # Curved 
            radius = distance / beta
            cx = self.x - math.sin(self.theta) * radius
            cy = self.y + math.cos(self.theta) * radius

            self.x = cx + math.sin(self.theta + beta) * radius
            self.y = cy - math.cos(self.theta + beta) * radius
            self.theta = (self.theta + beta) % (2.0 * math.pi)
    
    def sense(self):
        z = []
        for landmark in self.landmarks:
            pelling = math.atan2(landmark[1] - self.y, landmark[0] - self.x) - self.theta
            if self.noise:
                pelling += random.gauss(0.0, self.sensor_noise)
            pelling %= 2.0 * math.pi
            z.append(normalize_angle(pelling))
        return z
    
    def measurement_prob(self, measurements):
        prob = 1.0
        for i in range(len(self.landmarks)):
            # Calclation true pelling
            true_pelling = math.atan2(self.landmarks[i][1] - self.y, self.landmarks[i][0] - self.x) - self.theta
            true_pelling %= 2.0 * math.pi
            
            # Calculate measurement probability (Gaussian)
            error = (measurements[i] - normalize_angle(true_pelling)) % (2.0 * math.pi)
            prob *= self.gaussian(error, 0.0, self.sensor_noise)
        return prob
    
    def get_distance(self):
        land_distance = []
        for landmark in self.landmarks:
            measurement = math.sqrt((landmark[0] - self.x)**2 + (landmark[1] - self.y)**2)
            if self.noise:
                measurement += random.gauss(0.0, self.sensor_noise)
            land_distance.append(measurement)
        return land_distance
    

    # To calculate particles weights
    @staticmethod
    def gaussian(mu, mu0, sigma):
        # return (1.0 / np.sqrt(2.0 * np.pi * sigma**2)) * np.exp(-0.5 * ((mu - mu0) ** 2) / sigma**2)
        return math.exp(-((mu - mu0) ** 2) / (sigma ** 2)) / math.sqrt(2.0 * math.pi * (sigma ** 2))