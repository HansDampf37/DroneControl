# Drone RL Environment 🚁

A Gymnasium-compatible Reinforcement Learning environment for quadcopter control in Python.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Environment Details](#environment-details)
- [Physics Model](#physics-model)
- [Configuration](#configuration)
- [RL Training](#rl-training)
- [Project Structure](#project-structure)
- [Additional Documentation](#additional-documentation)

## Features

- **Realistic Physics**: Simplified quadcopter physics with 4 independent motors in X-configuration
- **Dynamic Wind**: Ornstein-Uhlenbeck process for realistic wind variations
- **Dense Reward**: `((max_distance - distance) / max_distance) ** 2`
- **Optimized 2-View Visualization**: Top view (XY) + Front view (XZ) like technical drawings
- **Gymnasium-Compatible**: Standard RL interface
- **Crash Detection**: Automatic episode termination on crash

## Quick Start

### Installation
```bash
# Clone repository
git clone <your-repo-url>
cd drone-control

# Install dependencies
pip install -r requirements.txt

# Optional: Developer installation
pip install -e .
```

### Simple Test
```bash
# Minimal rendering test (20 steps)
python tests/test_minimal_render.py

# Comprehensive tests
python tests/test_env.py
```

### Example: Random Agent
```bash
# Without visualization
python examples/random_agent.py --episodes 5

# With visualization
python examples/random_agent.py --episodes 3 --render
```

### Usage in Code
```python
from src.drone_env import DroneEnv

# Create environment
env = DroneEnv(max_steps=1000, render_mode="human")
obs, info = env.reset()

# Main loop
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Environment Details

### Action Space
- **Type**: `Box(4,)` with value range [0, 1]
- **Description**: 4 motors (0-100% thrust)
  - Motor 0: Front-Right
  - Motor 1: Rear-Left  
  - Motor 2: Front-Left
  - Motor 3: Rear-Right

### Observation Space
- **Type**: `Box(15,)` 
- **Components**:
  - `[0:3]` - Relative position to target (x, y, z)
  - `[3:6]` - Linear velocity (vx, vy, vz)
  - `[6:9]` - Orientation (Roll, Pitch, Yaw)
  - `[9:12]` - Angular velocity (wx, wy, wz)
  - `[12:15]` - Wind vector (wx, wy, wz)

**Note**: The drone is always at position (0, 0, 0) in the observation. The target point is given relative to the drone.

### Reward Function
```python
reward = ((max_distance_to_target - distance_to_target) / max_distance_to_target) ** 2
```
- Value range: [0, 1]
- Dense reward for better training

### Termination
- **Truncated**: After fixed `max_steps` (default: 1000)
- **Terminated**: On crash (optional, see Crash Detection)
  - Vertical velocity < -20.0 m/s (falling too fast)
  - Extreme tilt (>80° roll or pitch)
  - Distance to target exceeds maximum

## Physics Model

### Quadcopter X-Configuration
- 4 rotors arranged diagonally (±45° to axes)
- Mass: 1.0 kg
- Arm length: 0.25 m

### Force Calculation
1. **Thrust**: Force vector perpendicular to rotor plane, scaled by motor power
2. **Torque**:
   - Roll: Thrust difference between left/right motors
   - Pitch: Thrust difference between front/rear motors
   - Yaw: Reactive torque from rotor spin directions
3. **Wind**: Ornstein-Uhlenbeck process, configurable
4. **Integration**: Euler integration with 0.01s timestep (100 Hz)

## Configuration

```python
env = DroneEnv(
    max_steps=1000,                      # Episode length
    dt=0.01,                             # Timestep (s)
    target_change_interval=None,         # Target change (None = fixed)
    wind_strength_range=(0.0, 5.0),     # Wind strength (m/s)
    use_wind=False,                      # Enable wind simulation
    render_mode="human",                 # "human", "rgb_array", None
    enable_crash_detection=True,         # Enable crash detection
    crash_z_vel_threshold=-20.0,        # Crash velocity threshold
    crash_tilt_threshold=80.0            # Crash tilt threshold (degrees)
)
```

## RL Training

### Example with Ray RLlib
```bash
# Installation
pip install ray[rllib] torch

# Training
python examples/training.py --mode train --algorithm PPO --timesteps 100000

# Evaluation
python examples/training.py --mode eval --algorithm PPO --model-path models/drone_model
```

### Recommended Algorithms
- **PPO**: Stable, good for beginners, on-policy
- **SAC**: Very good performance with continuous actions, off-policy
- **APPO**: Asynchronous PPO, good for distributed training

### Baseline Performance
- **Hover Agent** (all motors ~25%): Reward ~0.05-0.10
- **Trained Agent**: Reward >0.3-0.5 after 100k steps

## Project Structure

```
drone-control/
├── src/drone_env/          # Main environment
│   ├── __init__.py
│   ├── drone.py           # Drone physics model
│   ├── env.py             # Gymnasium environment
│   └── renderer.py        # Visualization
├── tests/                  # Tests & debugging
│   ├── test_env.py
│   ├── test_rendering.py
│   └── ...
├── examples/               # Example scripts
│   ├── random_agent.py
│   └── training.py
├── docs/                   # Documentation
│   ├── DEVELOPMENT.md      # Developer guide
│   └── TROUBLESHOOTING.md  # Common issues
├── setup.py
├── requirements.txt
└── README.md              # This file
```

## Additional Documentation

- **[Development Guide](docs/DEVELOPMENT.md)** - Developer information, structure, extensions
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues, especially rendering

## Roadmap

- [ ] 3D visualization
- [ ] Recurrent policies (wind inference without direct observation)
- [ ] Multiple target points per episode
- [ ] Obstacles
- [x] Crash detection
- [ ] Energy consumption in reward

## License

MIT

## Author

Adrian - 2025

