import numpy as np
import matplotlib.pyplot as plt
import copy
from Robot import Robot

def plot_trajectory(ax, trajectory_noise, theta, particles):
    x_noise, y_noise = zip(*trajectory_noise)

    ax.cla()  # Clear the axis to update the plot
    ax.scatter([p.x for p in particles], [p.y for p in particles], s=1, color='g', label='Particles')
    ax.plot(x_noise, y_noise, 'ro-', label='True position')
    endx = x_noise[-1] + np.cos(theta) * 6.0
    endy = y_noise[-1] + np.sin(theta) * 6.0
    ax.scatter([0, 0, 100, 100], [0, 100, 0, 100], s=15, color='black', label='Landmarks')
    ax.arrow(x_noise[-1], y_noise[-1], endx - x_noise[-1], endy - y_noise[-1], head_width=4.5, head_length=4.5, fc='b', ec='b')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Robot Trajectory')
    ax.set_xlim(-50, 150)
    ax.set_ylim(-50, 150)
    ax.legend()
    ax.grid(True)
    fig.canvas.draw()  # Ensure the canvas is updated
    fig.canvas.flush_events()  # Force the update to happen

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
    
    # Normalize weights
    return normalize_weights(weights)

def resample_particles(particles, weights):
    new_particles = []
    new_weights = []
    N = len(particles)
    index = np.random.randint(0, N)
    beta = 0
    for _ in range(N):
        beta = beta + np.random.uniform(0, 2 * max(weights))
        while beta > weights[index]:
            beta = beta - weights[index]
            index = (index + 1) % N
        new_particles.append(copy.deepcopy(particles[index]))
        new_weights.append(weights[index])
    
    return new_particles, new_weights

if __name__ == '__main__':
    num_particles = 5000
    particles, weights = initialize_particles(num_particles)

    # Set trajectory
    u = [[0.5, 3.0], [0.0, 40.0], [1.1, 2.0], [0.0, 35.0], [-0.7, 4.0], [0.0, 10.0], [-0.4, 7.0], [-0.0, 20.0], [-1.1, 2.0], [0.0, 12.0]]
    
    robot = Robot(noise=True)
    trajectory_noise = [(robot.x, robot.y)]

    # Initialize figure for drawing
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.ion()

    # Move robot
    for tr in u:
        robot.move(tr)
        trajectory_noise.append((robot.x, robot.y))
        measurements = robot.get_distance()

        # Particle filter
        move_particles(particles, tr)
        weights = update_particle_weights(particles, measurements, weights)
        particles, weights = resample_particles(particles, weights)

        plot_trajectory(ax, trajectory_noise, robot.theta, particles)
        plt.pause(0.001)

        # Estimation
        x_est = sum(particle.x for particle in particles) / len(particles)
        y_est = sum(particle.y for particle in particles) / len(particles)
        theta_est = sum(particle.theta for particle in particles) / len(particles)

        robot_est_pos = [x_est, y_est, theta_est]

        print("Robot state:", robot.x, robot.y, robot.theta)
        print("Robot estimated state:", robot_est_pos[0], robot_est_pos[1], robot_est_pos[2])

    plt.ioff()
    plt.show() 
