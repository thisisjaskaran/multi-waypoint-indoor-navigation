# Multiple Waypoint Navigation in Unknown Indoor Environments

This repository is the codebase for the paper Multiple Waypoint Navigation in Unknown Indoor Environments. This was the winning solution of the [The IROS-RSJ Navigation and Manipulation Challenge for Young Students 2021](https://www.iros2021.org/the-iros-rsj-navigation-and-manipulation-challenge-for-young-students).

## Introduction

We present a multiple waypoint path planner and controller stack for navigation in unknown indoor environments where waypoints include the goal along with the intermediary points that the robot must traverse before reaching the goal. Our approach makes use of a global planner (to find the next best waypoint at any instant), a local planner (to plan the path to a specific waypoint) and an adaptive Model Predictive Control strategy (for robust system control and faster maneuvers).

## Installation

```bash
pip3 install -r requirements.txt
```

## Running the Code

```bash
python participant_controller.py
```

This code was benchmarked and tested on the TIAGO Base robot by PAL Robotics, on the Webots simulator.
