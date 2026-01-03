# Drone RL Environment 🚁

A Gymnasium-compatible Reinforcement Learning environment for quadcopter control in Python with realistic physics, dynamic wind simulation, and fast 3D visualization.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Environment Details](#environment-details)
  - [Base Environment (DroneEnv)](#base-environment-droneenv)
  - [Sequential Waypoint Environment](#sequential-waypoint-environment)
  - [Action Wrappers](#action-wrappers)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [Reward Functions](#reward-functions)
- [Physics Model](#physics-model)
- [Configuration](#configuration)
- [Example Scripts](#example-scripts)
- [RL Training](#rl-training)
- [Project Structure](#project-structure)
- [Additional Documentation](#additional-documentation)

## Features

- **Realistic Physics**: Simplified quadcopter physics with 4 independent motors in X-configuration, quaternion-based orientation, and pendulum-like rotor dynamics
- **Dynamic Wind**: Ornstein-Uhlenbeck process for realistic wind variations
- **Multiple Environments**: 
  - `DroneEnv`: Fly to a single target and hover
  - `SequentialWaypointEnv`: Navigate through multiple waypoints sequentially
- **Action Wrappers**: Transform action space for easier learning
  - `ThrustChangeController`: Control thrust changes instead of absolute values
  - `MotionPrimitiveActionWrapper`: Control hover/roll/pitch/yaw primitives
- **Fast 3D Visualization**: High-performance pygame-based 3D renderer (100+ FPS)
- **Alternative 2D Renderer**: Matplotlib-based multi-view renderer for detailed analysis
- **Interactive Controls**: Manual drone control with keyboard, camera controls with mouse/keyboard
- **Gymnasium-Compatible**: Standard RL interface with robust observation/action spaces
- **Crash Detection**: Automatic episode termination on crash (high-speed descent, extreme tilt)
- **Out-of-Bounds Detection**: Episode termination when drone strays too far from target

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

### Interactive Demo
The best way to understand the environment is through the interactive demo:

```bash
# Manual control mode - fly the drone yourself!
python examples/demo_interactive.py --mode manual

# Watch a random agent with wind simulation
python examples/demo_interactive.py --mode random --wind

# Watch a simple hovering controller
python examples/demo_interactive.py --mode hover --fps 60
```

**Controls:**
- **Motors (Manual mode)**: Keys 1-4 increase thrust, 5-8 decrease thrust
- **Camera**: Mouse drag to rotate, WASD/Arrow keys to move, Space/Shift for up/down
- **R**: Reset episode
- **ESC/X**: Exit

### Simple Tests
```bash
# Minimal rendering test (20 steps)
python tests/test_minimal_render.py

# Comprehensive environment tests
python tests/test_env.py
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

### Base Environment (DroneEnv)

The base `DroneEnv` is a single-target hovering task where the drone must fly to a randomly positioned target and maintain position there.

**Key Features:**
- Single target point per episode (optionally changing at intervals)
- Dense reward based on distance and velocity alignment
- Episode ends on crash, out-of-bounds, or max steps
- Configurable crash detection and wind simulation

**Usage:**
```python
from src.drone_env import DroneEnv

env = DroneEnv(
    max_steps=1000,
    render_mode="human",
    use_wind=True,
    enable_crash_detection=True
)

obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        break
env.close()
```

### Sequential Waypoint Environment

`SequentialWaypointEnv` extends `DroneEnv` for multi-waypoint navigation tasks. The drone must visit waypoints in sequence as quickly as possible.

**Key Features:**
- Navigate through multiple waypoints sequentially
- Speed-based reward encourages fast, efficient flight
- Decaying checkpoint bonuses (rewards reaching waypoints quickly)
- Extended observation space includes next waypoint position
- Relaxed out-of-bounds detection for more dynamic flight

**Observation Space Extension:**
- Base observation (33 elements) + next waypoint vector (3) + time since checkpoint (1) = **37 elements**

**Usage:**
```python
from src.drone_env import SequentialWaypointEnv

env = SequentialWaypointEnv(
    max_num_waypoints=15,
    waypoint_reach_threshold_m=0.4,
    checkpoint_bonus=10.0,
    bonus_decay_rate_per_sec=2.0,
    max_steps=600,
    dt=0.05  # 20 Hz
)
```

### Action Wrappers

Wrappers transform the action space to make learning easier or enable different control paradigms.

#### ThrustChangeController

Instead of commanding absolute motor thrusts [0, 1], this wrapper interprets actions as **thrust changes** [-1, 1].

**Benefits:**
- Smoother control transitions
- Better for temporal consistency
- Action space: `Box(4,)` with range [-1, 1]

**Action Transformation:**
```python
new_thrust = current_thrust + dt * action
```

**Usage:**
```python
from src.drone_env import DroneEnv, ThrustChangeController

env = ThrustChangeController(DroneEnv())
# Or with RLlib:
env_config = {
    "wrappers": [ThrustChangeController]
}
```

#### MotionPrimitiveActionWrapper

Maps high-level motion primitives (hover, roll, pitch, yaw) to individual motor commands.

**Action Space:** `Box(4,)` representing [hover, roll, pitch, yaw]

**Mapping to Motors:**
```python
motor[0] = hover + roll + pitch + yaw   # Front-Right
motor[1] = hover - roll - pitch + yaw   # Rear-Left
motor[2] = hover - roll + pitch - yaw   # Front-Left
motor[3] = hover + roll - pitch - yaw   # Rear-Right
```

**Usage:**
```python
from src.drone_env import DroneEnv, MotionPrimitiveActionWrapper

env = MotionPrimitiveActionWrapper(DroneEnv())
```

## Observation Space

### DroneEnv Observation Space

**Type**: `Box(33,)` - All observations are `float32`

The observation provides a complete state of the drone relative to the target:

| Index | Component | Description | Range |
|-------|-----------|-------------|-------|
| 0-2 | **Relative Position** | Vector from drone to target (x, y, z) | [-1.5, 1.5] m |
| 3-5 | **Linear Velocity** | Drone velocity (vx, vy, vz) | [-40, 40] m/s |
| 6-8 | **Linear Acceleration** | Drone acceleration (ax, ay, az) | [-50, 50] m/s² |
| 9-12 | **Orientation (Quaternion)** | Quaternion (w, x, y, z) | [-∞, ∞] |
| 13-15 | **Angular Velocity** | Rotation rates (wx, wy, wz) | [-10, 10] rad/s |
| 16-18 | **Angular Acceleration** | Angular acceleration (awx, awy, awz) | [-100, 100] rad/s² |
| 19-21 | **Normal Vector** | Drone facing direction (unit vector) | [-1, 1] |
| 22-24 | **Wind Vector** | Current wind velocity (wx, wy, wz) | [-5, 5] m/s |
| 25-28 | **Current Motor Thrusts** | Actual motor forces (4 values) | [0, 1] |
| 29-32 | **Target Motor Thrusts** | Commanded motor targets (4 values) | [0, 1] |

**Key Properties:**
- Drone is always at origin in its own reference frame
- Target position is given relative to drone
- Rich state information enables complex control strategies
- Includes both current state and rate of change (velocities & accelerations)

### SequentialWaypointEnv Observation Space

**Type**: `Box(37,)` - Extends DroneEnv observation

Additional observations for waypoint navigation:

| Index | Component | Description | Range |
|-------|-----------|-------------|-------|
| 33-35 | **Next Waypoint Vector** | Vector from drone to next waypoint | [-6, 6] m |
| 36 | **Time Since Last Checkpoint** | Seconds since reaching last waypoint | [0, ∞] |

**Example Usage:**
```python
obs, info = env.reset()

# DroneEnv observation
rel_pos = obs[0:3]          # Where is the target?
velocity = obs[3:6]         # How fast am I moving?
orientation_q = obs[9:13]   # What's my orientation?
normal = obs[19:22]         # Which way am I facing?
wind = obs[22:25]           # What's the wind doing?
motor_thrusts = obs[25:29]  # Current motor forces

# SequentialWaypointEnv additional
next_waypoint = obs[33:36]  # Where is the next waypoint after current target?
time_since_checkpoint = obs[36]  # How long since I hit the last waypoint?
```

## Action Space

### DroneEnv Action Space (Default)

**Type**: `Box(4,)` with range [0, 1]

Direct motor thrust commands for 4 motors in X-configuration:

```
Motor Layout (Top View):
      2 (FL, CCW)
          ○
         / \
        /   \
    0 ○     ○ 3
   (FR,CW)  (RR,CW)
        \   /
         \ /
          ○
      1 (RL, CCW)
```

**Motor Indices:**
- Motor 0: Front-Right (CW rotation)
- Motor 1: Rear-Left (CCW rotation)
- Motor 2: Front-Left (CCW rotation)
- Motor 3: Rear-Right (CW rotation)

**Action Values:**
- `0.0`: Motor off
- `~0.5`: Approximate hover thrust (balances 1kg drone against gravity)
- `1.0`: Maximum thrust

**Example:**
```python
# Hover (approximately)
action = np.array([0.5, 0.5, 0.5, 0.5])

# Increase altitude
action = np.array([0.6, 0.6, 0.6, 0.6])

# Roll right (increase left motors, decrease right motors)
action = np.array([0.4, 0.6, 0.6, 0.4])

# Pitch forward (increase rear motors, decrease front motors)
action = np.array([0.4, 0.6, 0.4, 0.6])
```

### ThrustChangeController Action Space

**Type**: `Box(4,)` with range [-1, 1]

Commands **changes** to motor thrusts rather than absolute values:

```python
new_thrust = current_thrust + dt * action
```

**Benefits:**
- Temporal continuity - smoother control
- Natural for learning incremental adjustments
- Better for fine-tuning around hover state

**Example:**
```python
# Gradually increase all motors
action = np.array([0.5, 0.5, 0.5, 0.5])  # Increases thrust at 50%/sec

# Maintain current thrust
action = np.array([0.0, 0.0, 0.0, 0.0])

# Decrease motor 0, increase motor 2 (roll)
action = np.array([-0.3, 0.0, 0.3, 0.0])
```

### MotionPrimitiveActionWrapper Action Space

**Type**: `Box(4,)` representing [hover, roll, pitch, yaw]

High-level control primitives mapped to motor commands:

| Index | Primitive | Effect |
|-------|-----------|--------|
| 0 | Hover | Overall thrust (altitude control) |
| 1 | Roll | Rotation around forward axis |
| 2 | Pitch | Rotation around lateral axis |
| 3 | Yaw | Rotation around vertical axis |

**Example:**
```python
# Hover in place
action = np.array([0.5, 0.0, 0.0, 0.0])

# Hover + roll right
action = np.array([0.5, 0.3, 0.0, 0.0])

# Hover + pitch forward + yaw left
action = np.array([0.5, 0.0, 0.3, -0.2])
```

## Reward Functions

### DroneEnv Reward Function

The reward combines **position-based** and **velocity-based** components for dense, informative feedback:

```python
distance = norm(target_position - drone_position)
direction_to_target = (target_position - drone_position) / distance
velocity_towards_target = dot(drone_velocity, direction_to_target)

reward_position = exp(-distance)
reward_velocity = tanh(velocity_towards_target / 3.0)
reward = reward_position + 0.5 * (1 - reward_position) * reward_velocity
```

**Components:**

1. **Position Reward** (`exp(-distance)`):
   - `1.0` when exactly at target (distance = 0)
   - `~0.37` at 1 meter from target
   - `~0.05` at 3 meters from target
   - Exponential decay provides strong gradient near target

2. **Velocity Reward** (`tanh(v_correct / 3.0)`):
   - Positive when moving toward target
   - Negative when moving away from target
   - Scaled by (1 - reward_position) so it matters less when already at target
   - Helps agent learn to approach target efficiently

**Reward Range**: Approximately [-1.0, 1.0]

**Properties:**
- Dense reward - informative at every step
- Strong gradient near target for precision
- Velocity term encourages efficient approach
- Balanced between exploration and exploitation

### SequentialWaypointEnv Reward Function

Optimized for **fast waypoint navigation**:

```python
distance = norm(target_position - drone_position)

if distance < waypoint_reach_threshold:
    # Just reached waypoint - give decaying bonus
    reward = max(checkpoint_bonus - bonus_decay_rate * time_since_last, 1.0)
else:
    # Moving toward waypoint - reward correct velocity
    direction_to_target = (target_position - drone_position) / distance
    velocity_towards_target = dot(drone_velocity, direction_to_target)
    reward = tanh(velocity_towards_target / 3.0)
    
    # Small bonus for efficient motor usage
    if reward > 0:
        reward += mean(motor_thrusts) * 0.5
```

**Components:**

1. **Checkpoint Bonus** (when reaching waypoint):
   - Initial bonus: `checkpoint_bonus` (default: 10.0)
   - Decays by `bonus_decay_rate_per_sec` (default: 2.0/sec)
   - Minimum reward: 1.0
   - **Encourages fast waypoint reaching**

2. **Velocity Reward** (while navigating):
   - Same as DroneEnv but without position component
   - Focus on speed rather than hovering
   - Positive for moving toward target, negative for wrong direction

3. **Efficiency Bonus**:
   - Small bonus proportional to motor usage
   - Only when moving in correct direction
   - Encourages aggressive, efficient flight

**Example Scenarios:**
- Reach waypoint in 2 seconds: `10 - 2*2 = 6.0` reward
- Reach waypoint in 4 seconds: `10 - 2*4 = 2.0` reward
- Reach waypoint in 6+ seconds: `1.0` reward (minimum)
- Moving toward waypoint at 5 m/s: `~0.86` reward/step
- Moving away from waypoint: Negative reward

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

### DroneEnv Parameters

```python
from src.drone_env import DroneEnv

env = DroneEnv(
    max_steps=1000,                           # Max steps per episode
    dt=0.01,                                  # Simulation timestep (s) [100 Hz]
    target_change_interval=None,              # Steps between target changes (None = static)
    wind_strength_range=(0.0, 5.0),          # Wind speed range (m/s)
    use_wind=False,                           # Enable wind simulation
    render_mode="human",                      # "human", "rgb_array", or None
    renderer_type="pygame",                   # "pygame" (fast 3D) or "matplotlib" (2D)
    enable_crash_detection=True,              # Detect and terminate on crashes
    enable_out_of_bounds_detection=True,      # Terminate if too far from target
    crash_z_vel_threshold=-20.0,             # Crash if falling faster than this (m/s)
    crash_tilt_threshold=80.0,               # Crash if tilted more than this (degrees)
)
```

### SequentialWaypointEnv Parameters

```python
from src.drone_env import SequentialWaypointEnv

env = SequentialWaypointEnv(
    max_num_waypoints=15,                     # Number of waypoints to generate
    waypoint_reach_threshold_m=0.4,          # Distance to consider waypoint "reached" (m)
    checkpoint_bonus=10.0,                    # Base reward for reaching waypoint
    bonus_decay_rate_per_sec=2.0,            # Reward decay per second (encourages speed)
    max_steps=600,                            # Max steps per episode
    dt=0.05,                                  # Timestep (0.05 = 20 Hz is good for this env)
    # ... all DroneEnv parameters also available
)
```

### RLlib Integration

For Ray RLlib training, use `RLlibDroneEnv` wrapper with configuration dictionary:

```python
from src.drone_env import RLlibDroneEnv, SequentialWaypointEnv, ThrustChangeController

env_config = {
    "env_class": SequentialWaypointEnv,       # Which environment class to use
    "max_steps": 600,
    "dt": 0.05,
    "use_wind": True,
    "wind_strength_range": (0.0, 3.0),
    "enable_crash_detection": False,
    "enable_out_of_bounds_detection": True,
    "wrappers": [ThrustChangeController],     # List of wrapper classes
    # SequentialWaypointEnv specific:
    "max_num_waypoints": 15,
    "waypoint_reach_threshold": 0.4,
    "checkpoint_bonus": 10.0,
    "bonus_decay_rate_per_sec": 2.0,
}

# Use in RLlib config
config = PPOConfig().environment(RLlibDroneEnv, env_config=env_config)
```

## Example Scripts

The project includes two primary example scripts for different use cases:

### 1. Interactive Demo (`examples/demo_interactive.py`)

Explore the environment dynamics with manual control or watch autonomous agents. Best for understanding how the drone behaves and testing the visualization.

**Features:**
- **Manual Mode**: Fly the drone yourself with keyboard controls
- **Random Mode**: Watch a random agent explore
- **Hover Mode**: Watch a simple PD controller try to reach targets
- Interactive 3D camera controls
- Real-time physics visualization

**Usage:**
```bash
# Manual control - fly the drone yourself
python examples/demo_interactive.py --mode manual

# Random agent with wind
python examples/demo_interactive.py --mode random --wind

# Hovering controller at 60 FPS
python examples/demo_interactive.py --mode hover --fps 60
```

**Controls:**
- **Motors (manual mode)**: 1-4 increase thrust, 5-8 decrease thrust
- **Camera**: Mouse drag to rotate, WASD/arrows to move, Space/Shift for vertical
- **R**: Reset episode
- **ESC/X**: Exit

### 2. RL Training (`examples/training.py`)

Train reinforcement learning agents using Ray RLlib with PPO, SAC, or APPO algorithms.

**Features:**
- Multiple RL algorithms (PPO, SAC, APPO)
- Custom metrics logging (TensorBoard)
- Checkpoint saving and loading
- Evaluation during training
- Supports both DroneEnv and SequentialWaypointEnv

**Usage:**
```bash
# Train PPO on sequential waypoint task
python examples/training.py --algorithm PPO --timesteps 1000000

# Train SAC with custom model path
python examples/training.py --algorithm SAC --timesteps 500000 --save-path ./my_model

# Continue training from checkpoint
python examples/training.py --algorithm PPO --timesteps 1000000 --load-from ./my_model

# Evaluate trained model
python examples/training.py --algorithm PPO --evaluate --load-from ./models/drone_model
```

**Example Training Configuration:**
```python
# See examples/training.py for full implementation
env_config = {
    "env_class": SequentialWaypointEnv,
    "max_steps": 600,
    "dt": 1.0/20,  # 20 Hz
    "use_wind": True,
    "wrappers": [ThrustChangeController],
}

config = (
    PPOConfig()
    .environment(RLlibDroneEnv, env_config=env_config)
    .framework("torch")
    .training(
        lr=3e-4,
        gamma=0.99,
        lambda_=0.95,
        # ... other training parameters
    )
)
```

## RL Training

Training details are covered in the [Example Scripts](#example-scripts) section above. Key points:

### Ray RLlib Integration

The environment integrates seamlessly with Ray RLlib through the `RLlibDroneEnv` wrapper:

```python
from ray.rllib.algorithms.ppo import PPOConfig
from src.drone_env import RLlibDroneEnv, SequentialWaypointEnv, ThrustChangeController

env_config = {
    "env_class": SequentialWaypointEnv,
    "max_steps": 600,
    "dt": 1.0/20,
    "use_wind": True,
    "wrappers": [ThrustChangeController],
}

config = (
    PPOConfig()
    .environment(RLlibDroneEnv, env_config=env_config)
    .framework("torch")
    .training(lr=3e-4, gamma=0.99, lambda_=0.95)
    .env_runners(num_env_runners=4)
)

algo = config.build()
for i in range(100):
    result = algo.train()
    print(f"Iteration {i}: reward_mean={result['env_runners']['episode_return_mean']}")
```

### Recommended Algorithms

- **PPO** (Proximal Policy Optimization): 
  - Stable, good default choice
  - On-policy, works well with continuous control
  - Best for SequentialWaypointEnv with ThrustChangeController

- **SAC** (Soft Actor-Critic):
  - Excellent performance with continuous actions
  - Off-policy (more sample efficient)
  - Good for DroneEnv hovering task

- **APPO** (Asynchronous PPO):
  - Asynchronous version of PPO
  - Good for distributed/parallel training
  - Similar performance to PPO but faster wall-clock time

### Training Tips

1. **Use ThrustChangeController wrapper**: Significantly improves learning by providing temporal continuity
2. **Adjust dt based on task**: 
   - DroneEnv: `dt=0.01` (100 Hz) for precise control
   - SequentialWaypointEnv: `dt=0.05` (20 Hz) for faster episodes
3. **Wind simulation**: Start without wind, add it for robustness later
4. **Crash detection**: Disable during early training to allow exploration
5. **Reward engineering**: The default rewards work well, but tune for your specific task

### Custom Metrics

The training script includes custom TensorBoard logging:

```python
class CustomMetricsCallback(DefaultCallbacks):
    def on_episode_end(self, episode, env_runner=None, metrics_logger=None, **kwargs):
        last_info = episode.infos[-1]
        for key, value in last_info.items():
            metrics_logger.log_value(f"custom_logs/{key}", value)
```

Logged metrics include:
- `distance_to_target`: Final distance to target
- `checkpoints_reached`: Waypoints reached (SequentialWaypointEnv)
- `crashed`: Whether episode ended in crash
- `out_of_bounds`: Whether drone went out of bounds

## Visualization

The environment supports two rendering backends:

### Pygame Renderer (Default)

Fast 3D visualization optimized for real-time interaction and training monitoring.

**Features:**
- **100+ FPS** performance
- Interactive camera controls (mouse + keyboard)
- 3D perspective view
- Coordinate axes indicator
- Visual elements:
  - Drone: Central sphere + 4 motor spheres
  - Blue arrow: Drone orientation (normal vector)
  - Yellow arrow: Velocity vector
  - Green sphere: Target position

**Usage:**
```python
env = DroneEnv(render_mode="human", renderer_type="pygame")
```

**Interactive Controls:**
- **Mouse**: Drag to rotate camera
- **WASD/Arrows**: Move camera position
- **Space/Shift**: Move camera up/down

### Matplotlib Renderer

Detailed multi-view renderer for analysis and debugging.

**Features:**
- Three 2D views: Top (XY), Front (XZ), Side (YZ)
- Detailed trajectory visualization
- ~10-12 FPS (slower but more informative)

**Usage:**
```python
env = DroneEnv(render_mode="human", renderer_type="matplotlib")
```

See [Pygame Renderer Documentation](docs/PYGAME_RENDERER.md) for detailed comparison and performance metrics.

## Project Structure

```
drone-control/
├── src/drone_env/              # Main environment package
│   ├── __init__.py            # Package exports and RLlibDroneEnv wrapper
│   ├── drone.py               # Drone physics model (forces, torques, integration)
│   ├── env.py                 # Gymnasium environments (DroneEnv, SequentialWaypointEnv)
│   │                          # and action wrappers (ThrustChangeController, etc.)
│   ├── wind.py                # Wind simulation (Ornstein-Uhlenbeck process)
│   ├── renderer.py            # Matplotlib-based 2D multi-view renderer
│   └── renderer_pygame.py     # Fast pygame-based 3D renderer
│
├── examples/                   # Example scripts
│   ├── demo_interactive.py    # 🎮 Interactive demo (manual/random/hover modes)
│   ├── training.py            # 🤖 RL training with Ray RLlib
│   ├── random_agent.py        # Simple random agent demo
│   ├── manual_control.py      # Manual keyboard control (superseded by demo_interactive.py)
│   ├── demo_pygame_renderer.py # Pygame renderer demo (superseded by demo_interactive.py)
│   └── interactive_camera_demo.py # Camera controls demo (superseded by demo_interactive.py)
│
├── tests/                      # Tests and debugging scripts
│   ├── test_env.py            # Environment tests
│   ├── test_minimal_render.py # Minimal 20-step rendering test
│   ├── test_rendering.py      # Rendering system tests
│   ├── test_crash_detection.py # Crash detection tests
│   └── ...                    # Other test files
│
├── docs/                       # Documentation
│   ├── PYGAME_RENDERER.md     # Pygame renderer guide & performance
│   ├── DEVELOPMENT.md         # Developer guide
│   ├── TROUBLESHOOTING.md     # Common issues and solutions
│   ├── CRASH_DETECTION.md     # Crash detection documentation
│   ├── SEQUENTIAL_WAYPOINT_ENV.md # Sequential waypoint environment
│   ├── VISUALIZATION.md       # Visualization overview
│   └── ...                    # Other documentation
│
├── models/                     # Saved RL models (checkpoints)
│   ├── drone_model/           # Example trained model
│   ├── sequential/            # Sequential waypoint models
│   └── ...
│
├── setup.py                   # Package installation config
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

**Key Files:**
- `src/drone_env/env.py` - Main environments and wrappers (647 lines)
- `src/drone_env/drone.py` - Core physics simulation
- `src/drone_env/renderer_pygame.py` - Fast 3D visualization
- `examples/demo_interactive.py` - **Best starting point for exploration**
- `examples/training.py` - **RL training template**

## Additional Documentation

Detailed documentation is available in the `docs/` directory:

### Environment & Features
- **[Sequential Waypoint Environment](docs/SEQUENTIAL_WAYPOINT_ENV.md)** - Multi-waypoint navigation task
- **[Crash Detection](docs/CRASH_DETECTION.md)** - Crash detection system details
- **[Visualization](docs/VISUALIZATION.md)** - Overview of rendering systems

### Rendering
- **[Pygame Renderer](docs/PYGAME_RENDERER.md)** - Fast 3D renderer guide and performance comparison
- **[Interactive Camera Controls](docs/INTERACTIVE_CAMERA_CONTROLS.md)** - Camera control documentation
- **[Rendering Optimization](docs/RENDERING_OPTIMIZATION.md)** - Performance optimization details

### Development
- **[Development Guide](docs/DEVELOPMENT.md)** - Developer information, architecture, extensions
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Performance Summary](docs/PERFORMANCE_SUMMARY.md)** - Rendering performance benchmarks

## Roadmap & Future Improvements

### Completed ✅
- [x] 3D visualization (pygame-based renderer)
- [x] Crash detection system
- [x] Multiple environment variants (DroneEnv, SequentialWaypointEnv)
- [x] Action wrappers (ThrustChangeController, MotionPrimitiveActionWrapper)
- [x] Interactive manual control
- [x] Camera controls (mouse + keyboard)
- [x] Dynamic wind simulation
- [x] Out-of-bounds detection
- [x] Rich observation space with quaternions
- [x] RLlib integration

### In Progress 🚧
- [ ] Improved reward shaping for faster convergence
- [ ] Recurrent policies (LSTM/GRU) for wind inference without direct observation
- [ ] Energy consumption metrics

### Planned 📋
- [ ] Obstacle avoidance tasks
- [ ] Multiple drones (swarm simulation)
- [ ] Vision-based observations (RGB camera)
- [ ] Real-world physics validation
- [ ] Model-based control baselines (MPC, LQR)
- [ ] Hardware-in-the-loop (HITL) testing support

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT

## Author

Adrian - 2026

