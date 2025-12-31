# Sequential Waypoint Navigation Environment

## Overview

The `SequentialWaypointEnv` is an extension of the base `DroneEnv` that challenges the drone to navigate through a sequence of waypoints as quickly as possible. This environment is designed to train agents for dynamic flight path following and efficient navigation.

## Key Features

- **Sequential Navigation**: Drone must visit waypoints in order
- **Lookahead Information**: Agent receives information about both current and next waypoint
- **Speed-Based Rewards**: Rewards encourage fast, efficient navigation
- **Automatic Advancement**: Waypoints automatically advance when reached
- **Flexible Configuration**: Customizable number of waypoints, spacing, and thresholds

## Differences from Base DroneEnv

| Aspect | DroneEnv | SequentialWaypointEnv |
|--------|----------|----------------------|
| Goal | Reach and hover at single target | Navigate through sequence of waypoints |
| Observation Space | 33 dimensions | 36 dimensions (adds next waypoint) |
| Episode Termination | Crash or timeout | Crash, timeout, or all waypoints reached |
| Reward | Distance-based exponential | Distance + speed + waypoint completion bonus |
| Target | Static (or periodic change) | Dynamic (advances on reach) |

## Environment Parameters

```python
SequentialWaypointEnv(
    num_waypoints=5,                          # Number of waypoints per episode
    waypoint_reach_threshold=0.4,             # Distance threshold to reach waypoint (m)
    waypoint_spacing_range=(2.0, 4.0),        # Min/max distance between waypoints (m)
    time_bonus_per_waypoint=1.0,              # Bonus reward for reaching waypoint
    speed_reward_weight=0.5,                  # Weight for speed component of reward
    **kwargs                                   # All DroneEnv parameters
)
```

### Parameters Description

- **num_waypoints**: Total number of waypoints to navigate per episode. More waypoints = longer episodes.
- **waypoint_reach_threshold**: Distance in meters within which a waypoint is considered "reached". 
  - Smaller values (0.2-0.3m) require precision
  - Larger values (0.5-0.8m) allow faster passage
- **waypoint_spacing_range**: Tuple of (min, max) distance between consecutive waypoints
  - Affects difficulty and flight patterns
  - Larger spacing = more room for acceleration
- **time_bonus_per_waypoint**: Immediate reward bonus when reaching a waypoint
  - Higher values encourage waypoint completion over hovering
- **speed_reward_weight**: How much to reward progress toward waypoint per timestep
  - Higher values encourage faster movement
  - Lower values encourage stability

## Observation Space

The observation space extends the base `DroneEnv` with 3 additional dimensions:

```
[0:3]   - Relative position to current waypoint (x, y, z)
[3:6]   - Linear velocity (vx, vy, vz)
[6:9]   - Linear acceleration (ax, ay, az)
[9:13]  - Orientation quaternion (qw, qx, qy, qz)
[13:16] - Angular velocity (wx, wy, wz)
[16:19] - Angular acceleration (awx, awy, awz)
[19:22] - Normal vector (drone facing direction)
[22:25] - Wind vector (wx, wy, wz)
[25:29] - Current motor thrusts
[29:33] - Target motor commands
[33:36] - Relative position to next waypoint (x, y, z)  ← NEW
```

Total: **36 dimensions** (33 base + 3 for next waypoint)

## Reward Function

The reward combines multiple components to encourage fast, efficient navigation:

```python
reward = reward_position + reward_speed + reward_vel + waypoint_bonus

where:
  reward_position = exp(-distance_to_waypoint)          # Exponential distance reward
  reward_speed = speed_weight * (prev_dist - curr_dist) # Progress reward
  reward_vel = 0.5 * tanh(velocity_toward_target / 0.3) # Velocity alignment
  waypoint_bonus = bonus_value (if waypoint reached)    # Completion bonus
```

### Reward Components

1. **Position Reward**: Exponential decay with distance (like base env)
2. **Speed Reward**: Direct reward for getting closer each timestep
3. **Velocity Alignment**: Rewards moving in the right direction
4. **Waypoint Bonus**: Large positive reward when waypoint is reached

## Episode Termination

An episode ends when:

1. **All waypoints reached** (success) - `all_waypoints_reached = True`
2. **Crash detected** - Excessive tilt or downward velocity
3. **Out of bounds** - Drone too far from current waypoint
4. **Timeout** - Maximum steps reached

## Info Dictionary

The info dict includes additional waypoint tracking:

```python
info = {
    'distance_to_target': float,           # Distance to current waypoint
    'position': np.array,                  # Drone position
    'target_position': np.array,           # Current waypoint position
    'step_count': int,                     # Steps in episode
    'episode_time': float,                 # Time in seconds
    'distance_progress': float,            # Total distance covered
    'crashed': bool,                       # Crash flag
    'out_of_bounds': bool,                 # Out of bounds flag
    'waypoint_reached': bool,              # Waypoint reached this step
    'all_waypoints_reached': bool,         # All waypoints completed
    'waypoints_reached': int,              # Number of waypoints reached
    'total_waypoints': int,                # Total waypoints in episode
    'waypoint_progress': float,            # Progress ratio [0, 1]
    'current_waypoint': np.array,          # Current target waypoint
    'next_waypoint': np.array,             # Preview of next waypoint
}
```

## Usage Examples

### Basic Usage

```python
from src.drone_env.env import SequentialWaypointEnv

env = SequentialWaypointEnv(
    num_waypoints=5,
    waypoint_reach_threshold_m=0.4,
    max_steps=3000,
)

obs, info = env.reset()

for step in range(3000):
    action = policy(obs)  # Your policy here
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

### Training with RLlib

```python
from ray.rllib.algorithms.ppo import PPOConfig
from src.drone_env.env import SequentialWaypointEnv

config = (
    PPOConfig()
    .environment(
        env=SequentialWaypointEnv,
        env_config={
            "num_waypoints": 5,
            "waypoint_reach_threshold": 0.4,
            "waypoint_spacing_range": (2.0, 4.0),
            "time_bonus_per_waypoint": 2.0,
            "speed_reward_weight": 1.0,
            "max_steps": 3000,
            "use_wind": True,
        }
    )
    .training(lr=3e-4, gamma=0.99)
)

algo = config.build()
algo.train()
```

## Design Rationale

### Why Lookahead Information?

Providing the next waypoint position allows the agent to:
- Plan smoother trajectories
- Anticipate turns and adjust speed accordingly
- Reduce overshoot when approaching waypoints
- Learn more efficient flight patterns

### Waypoint Generation Strategy

Waypoints are generated randomly with constraints:
- First waypoint: Random position in space
- Subsequent waypoints: Random direction, constrained distance from previous
- All waypoints clamped to environment boundaries

This creates varied but reasonable flight paths each episode.

### Reward Design Philosophy

The multi-component reward balances:
1. **Accuracy** (position reward) - Get close to waypoints
2. **Speed** (speed reward) - Move quickly toward goals
3. **Efficiency** (velocity alignment) - Move in the right direction
4. **Completion** (waypoint bonus) - Finish the course

## Performance Considerations

### Computational Cost

- Similar to base `DroneEnv` (same physics simulation)
- Slight overhead for waypoint queue management
- Observation space 9% larger (36 vs 33 dimensions)

### Training Difficulty

Compared to base environment:
- **Harder**: Requires dynamic decision-making
- **More structured**: Clear success criteria (waypoint completion)
- **Better exploration**: Waypoint bonuses provide shaping

Expected learning time: 1.5-2x longer than base environment

## Customization Ideas

### Easy Task
```python
env = SequentialWaypointEnv(
    num_waypoints=3,
    waypoint_reach_threshold=0.8,      # Large threshold
    waypoint_spacing_range=(1.5, 2.5), # Close spacing
    use_wind=False,                     # No wind
)
```

### Hard Task
```python
env = SequentialWaypointEnv(
    num_waypoints=10,
    waypoint_reach_threshold=0.3,      # Precision required
    waypoint_spacing_range=(3.0, 5.0), # Large distances
    use_wind=True,
    wind_strength_range=(0.0, 5.0),    # Strong wind
)
```

### Speed Challenge
```python
env = SequentialWaypointEnv(
    num_waypoints=5,
    waypoint_reach_threshold=0.5,
    time_bonus_per_waypoint=5.0,       # High bonus
    speed_reward_weight=2.0,           # Emphasize speed
    max_steps=1500,                     # Time pressure
)
```

## Future Extensions

Possible enhancements:
- Variable waypoint sizes (gates vs points)
- Time limits per waypoint
- Scoring based on completion time
- Obstacle avoidance between waypoints
- Formation flight with multiple drones
- Dynamic waypoint repositioning

## Troubleshooting

**Agent doesn't reach waypoints**: 
- Increase `waypoint_reach_threshold`
- Increase `time_bonus_per_waypoint`
- Check if agent can reach single target in base env first

**Agent completes too easily**:
- Decrease `waypoint_reach_threshold`
- Increase `num_waypoints`
- Add wind or increase wind strength

**Training is unstable**:
- Decrease `speed_reward_weight` (can cause overshoot)
- Increase `max_steps` to allow more exploration
- Start with fewer waypoints and curriculum learning

## See Also

- [Base DroneEnv Documentation](DEVELOPMENT.md)
- [Training Examples](../examples/train_sequential_waypoint.py)
- [Demo Script](../examples/demo_sequential_waypoint.py)
- [Test Suite](../tests/test_sequential_waypoint_env.py)

