# Particle Filter for Estimating Robot Position with Bicycle Model

This repository contains a Python implementation of a Particle Filter used to estimate the position of robot with bicycle model.

### Overview

In this project, the robot follows a predefined trajectory, and the goal is to estimate its position using the Particle Filter algorithm. The robot's motion is modeled using a bicycle model, which simulates realistic robot behavior by adding steering and forward noises.

### Demonstration

Here's a visual demonstration of the Particle Filter in action:

![Robot Trajectory Estimation](Images/Particle_Filter.gif)

### Usage

To run the code, clone this repository and execute the main script:

```bash
git clone https://github.com/nikisim/Particle_Filter_MGT.git
cd Particle_Filter_MGT
python main.py
```

Also in command line you will see true state of the robot ("Robot state") and estimated robot state ("Robot estimated state") at each iteration.
