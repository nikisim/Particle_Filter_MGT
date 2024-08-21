from Robot import Robot
import numpy as np
import matplotlib.pyplot as plt
import copy

def plot_trajectory(trajectory_noise, theta, particles):
    x_noise, y_noise = zip(*trajectory_noise)

    plt.figure(figsize=(10, 6))
    plt.scatter([p.x for p in particles], [p.y for p in particles], s=1, color='g', label='Particles')
    plt.plot(x_noise, y_noise, 'ro-', label='True position')
    endx = x_noise[-1] + np.cos(theta) * 6.0
    endy = y_noise[-1] + np.sin(theta) * 6.0
    plt.scatter([0,0,100,100], [0,100,0,100], s=15, color='black', label='Landmarks')
    plt.arrow(x_noise[-1], y_noise[-1], endx-x_noise[-1], endy-y_noise[-1], head_width=4.5, head_length=4.5, fc='b', ec='b')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Robot Trajectory')
    plt.xlim(-50,150)
    plt.ylim(-50,150)
    plt.legend()
    plt.grid(True)
    plt.show()

def normalize_weights(weights):
    total_weight = sum(weights)
    if total_weight > 0:
        return [w / total_weight for w in weights]
    else:
        return [1.0 / len(weights) for _ in weights] 

def initialize_particles(num_particles=1000):
    particles = []
    weights = []
    for _ in range(num_particles):
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 100)
        theta = np.random.uniform(-np.pi, np.pi)
        particles.append(Robot(x, y, theta, noise=True))
        weights.append(1.0 / num_particles)
    return particles, weights

def move_particles(particles, u):
    for particle in particles:
        particle.move(u)

def update_particle_weights(particles, measurements, weights):
    for p in range(len(particles)):
        particle_measurements = particles[p].get_distance() 
        weights[p] = 1.0
        for i in range(len(measurements)):
            # Using the Gaussian
            f = Robot.gaussian(particle_measurements[i], measurements[i], particles[p].sensor_noise)
            weights[p] *= f
    
    # Weights norm
    return normalize_weights(weights)

def resample_particles(particles, weights):
    newParticles = []
    newWeights = []
    N = len(particles)
    index = np.random.randint(0, N)
    betta = 0
    for _ in range(N):
        betta = betta + np.random.uniform(0, 2*max(weights))
        while betta > weights[index]:
            betta = betta - weights[index]
            index = (index + 1) % N # индекс изменяется в цикле от 0 до N
        newParticles.append(copy.deepcopy(particles[index]))
        newWeights.append(weights[index])
    
    return newParticles, newWeights

if __name__ == '__main__':
    num_particles = 5000
    particles, weights = initialize_particles(num_particles)

    # Set trajectory
    u = [[0.5, 3.0], [0.0, 40.0], [1.1, 2.0], [0.0, 35.0], [-0.7, 4.0], [0.0, 10.0], [-0.4, 7.0], [-0.0, 20.0], [-1.1, 2.0], [0.0, 12.0]]
    
    robot = Robot(noise=True)

    trajectory_noise = [(robot.x, robot.y)]

    # Move robot
    for tr in u:
        plot_trajectory(trajectory_noise, robot.theta, particles)
        robot.move(tr)

        trajectory_noise.append((robot.x, robot.y))

        measurements = robot.get_distance()

        # Particle filter
        move_particles(particles, tr)
        weights = update_particle_weights(particles, measurements, weights)

        particles, weights = resample_particles(particles, weights)
        x_est = sum(particle.x for particle in particles) / len(particles)
        y_est = sum(particle.y for particle in particles) / len(particles)
        theta_est = sum(particle.theta for particle in particles) / len(particles)

        robot_est_pos = [x_est, y_est, theta_est]

        print("Robot noise state:", robot.x, robot.y, robot.theta)
        print("Robot estimated state:", robot_est_pos[0], robot_est_pos[1], robot_est_pos[2])