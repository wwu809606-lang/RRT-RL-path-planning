# RRT-RL-path-planning

Baseline 3D UAV path planning in urban low-altitude airspace using Rapidly-exploring Random Tree (RRT), with future extensions toward reinforcement learning based path optimization.

## Overview

This project focuses on path planning for UAVs in urban low-altitude environments.  
The current version implements a **baseline 3D RRT planner** using building data as obstacles. It is designed as the first step of a larger research pipeline:

1. urban airspace modeling  
2. baseline path generation with classical planning methods  
3. reinforcement learning based path optimization  

The repository currently emphasizes **environment construction, collision-free path search, and visualization**.

## Main Features

- 3D urban environment construction based on building polygons and heights
- Baseline 3D RRT path planning
- Collision checking against buildings
- 2D visualization of planning results
- 3D visualization of buildings and planned path
- Modular structure for future RL-based optimization

## Project Structure

```text
RRT-RL-path-planning/
├── data/
│   ├── processed/              # processed building data
│   ├── raw/                    # raw data (optional, usually not tracked)
│   └── temp/                   # temporary files
├── results/                    # generated figures and outputs
├── scripts/                    # main scripts
├── requirements.txt            # dependency list
├── README.md
└── .gitignore