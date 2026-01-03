# Drone Control with Reinforcement Learning

This repository implements two Gymnasium environments for training reinforcement learning agents to control a quadcopter drone with realistic physics simulation.

## Environments

### 1. DroneEnv
A single-target navigation environment where the drone must fly to and maintain position at a target point.

### 2. SequentialWaypointEnv
A sequential waypoint navigation environment where the drone must navigate through a series of waypoints as quickly as possible.

---

## MDP Interface Specification

### Observation Space

**DroneEnv observation space (33 dimensions):**
- **[0:3]** - Relative position to target `(x, y, z)` in meters
- **[3:6]** - Linear velocity `(vx, vy, vz)` in m/s
- **[6:9]** - Linear acceleration `(ax, ay, az)` in m/s²
- **[9:13]** - Orientation as quaternion `(w, x, y, z)` (unit quaternion)
- **[13:16]** - Angular velocity `(wx, wy, wz)` in rad/s
- **[16:19]** - Angular acceleration `(awx, awy, awz)` in rad/s²
- **[19:22]** - Normal vector (drone facing direction, unit vector)
- **[22:25]** - Wind vector `(wx, wy, wz)` in m/s
- **[25:29]** - Current motor thrusts (4 motors, each in [0, 1])
- **[29:33]** - Target motor commands (4 motors, each in [0, 1])

**SequentialWaypointEnv observation space (37 dimensions):**
- **[0:33]** - Base DroneEnv observation
- **[33:36]** - Vector to next waypoint `(x, y, z)` in meters
- **[36]** - Time since last checkpoint in seconds

**Bounds:**
- Position: `[-space_side_length/2, space_side_length/2]` (default: [-1.5, 1.5] m)
- Velocity: `[-40, 40]` m/s per component
- Acceleration: `[-50, 50]` m/s² per component
- Angular velocity: `[-10, 10]` rad/s per component
- Angular acceleration: `[-100, 100]` rad/s² per component
- Wind: `[0, 5]` m/s by default (configurable)

---

### Action Space

**Base action space (4 dimensions):**
- 4 continuous values representing motor thrust commands
- Range: `[0.0, 1.0]` per motor
- Each value controls one of the 4 quadcopter motors (X-configuration)

**Action Wrappers:**

#### 1. ThrustChangeController
Interprets actions as **thrust changes** rather than absolute thrust values.
- **Input:** `[-1.0, 1.0]` per motor (thrust change rate)
- **Output:** Absolute thrust computed as: `thrust = previous_thrust + dt * action`
- **Effect:** Can increase motor command from 0% to 100% in 1 second
- **Benefits:** Smoother control, easier for the agent to learn incremental adjustments

#### 2. MotionPrimitiveActionWrapper
Interprets actions as **motion primitives** (hover, roll, pitch, yaw).
- **Input:** 4 values `[hover, roll, pitch, yaw]` in `[0.0, 1.0]`
- **Output:** Per-motor commands via differential mixing:
  ```
  motor[0] = hover + roll + pitch + yaw   (front-right)
  motor[1] = hover - roll - pitch + yaw   (front-left)
  motor[2] = hover - roll + pitch - yaw   (rear-left)
  motor[3] = hover + roll - pitch - yaw   (rear-right)
  ```
- **Benefits:** More intuitive control space aligned with typical drone control paradigms

---

### Reward Function

#### DroneEnv Reward

The reward function combines position-based and velocity-based components:

```python
distance = ||target_position - drone_position||
direction_to_target = (target_position - drone_position) / (distance + ε)
velocity_toward_target = dot(drone_velocity, direction_to_target)

r_position = exp(-distance)
r_velocity = tanh(velocity_toward_target / velocity_scale)

reward = r_position + 0.5 * (1 - r_position) * r_velocity
```

**Characteristics:**
- Range: approximately `[-1.0, 1.0]`
- At target (distance=0): reward = `1.0`
- Far from target: reward approaches `0.0` (or negative if moving away)
- Encourages both reaching the target and moving in the correct direction
- Dense reward signal for better learning

**Parameters:**
- `velocity_scale = 3.0` m/s (controls velocity reward sensitivity)

#### SequentialWaypointEnv Reward

The reward function adapts based on proximity to the current waypoint:

**When close to waypoint** (`distance < waypoint_reach_threshold_m`):
```python
reward = max(checkpoint_bonus - bonus_decay_rate * time_since_last_checkpoint, 1.0)
```

**When far from waypoint:**
```python
direction_to_target = (target_position - drone_position) / (distance + ε)
velocity_toward_target = dot(drone_velocity, direction_to_target)
reward = tanh(velocity_toward_target / velocity_scale)
```

**Aggressive flying bonus** (when reward > 0):
```python
reward += mean(motor_thrusts) * 0.5  # Encourages aggressive flying
```

**Parameters:**
- `checkpoint_bonus = 10.0` (base reward for reaching waypoint)
- `bonus_decay_rate_per_sec = 2.0` (penalty for slow navigation)
- `waypoint_reach_threshold_m = 0.1` m (distance to consider waypoint reached)

---

### Termination Criteria

Episodes can terminate in two ways:

#### 1. Terminated (True)
The episode ends early due to failure conditions:

**Crash Detection** (if `enable_crash_detection=True`):
- **Vertical velocity threshold:** `z_velocity < -20.0` m/s (falling too fast)
- **Tilt threshold:** `|roll| > 80°` or `|pitch| > 80°` (drone flipped over)

**Out of Bounds Detection** (if `enable_out_of_bounds_detection=True`):
- Distance to target exceeds maximum: `||drone_position - target_position|| > max_distance`
- `max_distance = sqrt(3) * space_side_length` (default: ~5.2 m)
- For SequentialWaypointEnv: `max_distance = 2 * sqrt(3) * space_side_length` (more permissive)

#### 2. Truncated (True)
The episode reaches its natural end:

**DroneEnv:**
- `step_count >= max_steps` (default: 1000 steps)

**SequentialWaypointEnv:**
- `step_count >= max_steps` OR
- `waypoints_reached >= max_num_waypoints` (default: 15 waypoints)

---

### Info Dictionary

The `info` dictionary returned by `step()` and `reset()` contains:

#### DroneEnv Info
```python
{
    'distance_to_target': float,      # Euclidean distance to target (m)
    'position': np.ndarray,            # Drone position [x, y, z] (m)
    'target_position': np.ndarray,     # Target position [x, y, z] (m)
    'step_count': int,                 # Current step number
    'episode_time': float,             # Elapsed time in seconds (step_count * dt)
    'distance_progress': float,        # initial_distance - current_distance (m)
    'crashed': bool,                   # True if crash detected (only in step())
    'out_of_bounds': bool,             # True if drone too far from target (only in step())
}
```

#### SequentialWaypointEnv Additional Info
```python
{
    # ... all DroneEnv info fields, plus:
    'waypoints_reached': int,          # Number of waypoints reached so far
    'current_waypoint_pos': np.ndarray, # Current target waypoint [x, y, z] (m)
    'next_waypoint_pos': np.ndarray,   # Next waypoint after current [x, y, z] (m)
    'steps_since_checkpoint': float,   # Time since last waypoint reached (seconds)
}
```

---

### Rendering

The environment supports three rendering modes:

#### Render Modes

1. **`None` (default)**
   - No rendering (fastest, for training)
   - `render()` returns `None`

2. **`"human"`**
   - Interactive 3D visualization using Pygame
   - Real-time display in a window
   - Interactive camera control:
     - **Mouse drag:** Rotate camera view
     - **WASD keys:** Move camera position
     - **Q/E keys:** Move camera up/down
     - **R key:** Reset camera to default position
   - `render()` returns `None`

3. **`"rgb_array"`**
   - Returns RGB array (numpy array) of the current frame
   - Format: `(height, width, 3)` with dtype `uint8`
   - Suitable for video recording and logging
   - No interactive window

#### Visualization Features

The 3D renderer displays:
- **Drone body:** Blue sphere at center
- **4 Rotors:** Smaller spheres at arm endpoints (X-configuration)
- **Up vector:** Arrow showing drone orientation (normal vector)
- **Velocity vector:** Arrow showing drone velocity direction
- **Wind vector:** Arrow showing wind direction
- **Target position:** Marked sphere at target location
- **Grid:** Ground reference grid
- **Debug info:** Overlayed text showing reward, distance, step count, etc.

#### Usage Example

```python
from src.drone_env import DroneEnv

# Create environment with rendering
env = DroneEnv(render_mode="human")

obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # Display the current state
    
    if terminated or truncated:
        break

env.close()
```

---

## Environment Parameters

### DroneEnv Configuration

```python
DroneEnv(
    max_steps=1000,                          # Episode length limit
    dt=0.01,                                 # Simulation timestep (seconds)
    target_change_interval=None,             # Steps between target position changes
    wind_strength_range=(0.0, 5.0),          # Min/max wind speed (m/s)
    use_wind=True,                           # Enable wind simulation
    render_mode=None,                        # None, "human", or "rgb_array"
    enable_crash_detection=True,             # Enable crash termination
    enable_out_of_bounds_detection=True,     # Enable boundary termination
    crash_z_vel_threshold=-20.0,             # Crash velocity threshold (m/s)
    crash_tilt_threshold=80.0,               # Crash tilt threshold (degrees)
)
```

### SequentialWaypointEnv Configuration

```python
SequentialWaypointEnv(
    max_num_waypoints=15,                    # Number of waypoints per episode
    waypoint_reach_threshold_m=0.1,          # Distance to reach waypoint (m)
    checkpoint_bonus=10.0,                   # Reward for reaching waypoint
    bonus_decay_rate_per_sec=2.0,            # Bonus decay rate
    # ... plus all DroneEnv parameters
)
```

---

## Physics Simulation

The environment implements realistic quadcopter physics:

- **Quadcopter configuration:** X-configuration with 4 motors
- **Mass:** 1.0 kg (default)
- **Arm length:** 0.10 m
- **Gravity:** 9.81 m/s²
- **Inertia tensor:** `[0.01, 0.01, 0.02]` kg⋅m²
- **Motor dynamics:** First-order lag with time constant
- **Thrust model:** Quadratic relationship `thrust ∝ cmd²`
- **Orientation:** Quaternion-based (avoids gimbal lock)
- **Aerodynamic drag:** `F_drag = -0.5 * ρ * C_d * A * |v| * v`
- **Angular damping:** Torque opposing rotation
- **Wind simulation:** Ornstein-Uhlenbeck process for realistic turbulence

---

## Training Example

```python
from src.drone_env import RLlibDroneEnv, ThrustChangeController, SequentialWaypointEnv
from ray.rllib.algorithms.ppo import PPOConfig

# Environment configuration
env_config = {
    "env_class": SequentialWaypointEnv,
    "max_steps": 600,
    "dt": 1.0/20,  # 20 Hz simulation
    "use_wind": True,
    "wrappers": [ThrustChangeController],
}

# Create PPO configuration
config = (
    PPOConfig()
    .environment(RLlibDroneEnv, env_config=env_config)
    .framework("torch")
    .training(
        lr=3e-4,
        train_batch_size=2048,
        gamma=0.96,
    )
    .env_runners(num_env_runners=15)
)

# Train
algo = config.build()
for i in range(100):
    result = algo.train()
    print(f"Iteration {i}: Reward={result['env_runners']['episode_return_mean']:.2f}")
```

See `examples/training.py` for complete training script with RLlib integration.

---

## Installation

```bash
pip install -r requirements.txt
```

**Main dependencies:**
- `gymnasium` - RL environment interface
- `numpy` - Numerical computations
- `pygame` - 3D rendering
- `ray[rllib]` - Distributed RL training (optional, for training)

---

## Usage

### Interactive Demo
```bash
python examples/demo_interactive.py
```

### Training
```bash
python examples/training.py --algorithm PPO --timesteps 1000000
```

### Evaluation
```bash
python examples/training.py --mode eval --model-path models/drone_model --episodes 5
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Adrian Degenkolb - 2025