# Crash Detection

The drone now detects crashes and automatically terminates the episode.

## Features

### ✅ Implemented

**1. Low Z-Velocity (Primary)**
- **Most Efficient Method**: O(1) comparison
- **Default Threshold**: vz < -20.0 m/s
- **Clear**: Drone is definitely falling/crashed

**2. Extreme Tilt (Secondary)**
- **For Uncontrolled Drone**: |Roll| > 80° OR |Pitch| > 80°
- **Default Threshold**: 80° (converted to radians)
- **Additional Check**: Catches total loss of control

**3. Configurable**
- Crash detection can be disabled
- All thresholds are adjustable

## Usage

### Default (Crash Detection Enabled)
```python
from src.drone_env import DroneEnv

env = DroneEnv(
    enable_crash_detection=True,     # Default
    crash_z_vel_threshold=-20.0,     # Default (m/s)
    crash_tilt_threshold=80.0,       # Default (degrees)
)
```

### Custom Thresholds
```python
env = DroneEnv(
    crash_z_vel_threshold=-30.0,     # Lower = more tolerance
    crash_tilt_threshold=85.0,       # Higher = more tolerance
)
```

### Disabled (Like Before)
```python
env = DroneEnv(
    enable_crash_detection=False
)
```

## Behavior

### Episode Ends on Crash
```python
obs, info = env.reset()

for step in range(1000):
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        if info['crashed']:
            print("Drone crashed!")
        break
    
    if truncated:
        print("Max steps reached")
        break
```

### Info Dictionary
```python
info = {
    'distance_to_target': float,
    'position': np.ndarray,
    'target_position': np.ndarray,
    'step_count': int,
    'crashed': bool,  # ← NEW
}
```

## Crash Criteria

### 1. Z-Velocity
```python
if self.velocity[2] < self.crash_z_vel_threshold:
    return True  # Crash!
```

**Typical Values:**
- vz = 0.0 m/s: Hovering
- vz = -5.0 m/s: Controlled descent
- vz = -20.0 m/s: Crash! (Default Threshold)
- vz = -30.0 m/s: Definitely crashed

### 2. Tilt (Orientation)
```python
roll, pitch, _ = self.orientation

if abs(roll) > self.crash_tilt_threshold:
    return True  # Completely tilted sideways

if abs(pitch) > self.crash_tilt_threshold:
    return True  # Completely tilted forward/backward
```

**Typical Values:**
- 0°-30°: Normal maneuvers
- 30°-60°: Aggressive maneuvers
- 60°-80°: Very aggressive, but still controllable
- >80°: Uncontrolled, Crash! (Default Threshold)

## Tests

```bash
python tests/test_crash_detection.py
```

**Test Scenarios:**
1. ✅ Crash due to low z-velocity
2. ✅ Crash due to extreme tilt
3. ✅ No detection when disabled
4. ✅ No false positives during normal hover
5. ✅ 'crashed' flag present in info

## Method Comparison

| Method | Efficiency | Clarity | False Positives | Early Detection |
|---------|-----------|---------|-----------------|-----------------|
| **Z-Velocity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Extreme Tilt** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Z-Coordinate (not impl.) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**Why Z-Velocity as Primary?**
- ✅ Simple calculation (1 comparison)
- ✅ Few false positives
- ✅ Early detection (before hitting ground)
- ✅ Works for rapid descents

**Why Extreme Tilt as Secondary?**
- ✅ Detects total loss of control
- ✅ Catches flips/rolls
- ✅ Additional safety check

## Impact on Training

### Before (Without Crash Detection)
```
Episode: 1000 Steps → truncated
→ Agent receives no negative signal for crash
```

### Now (With Crash Detection)
```
Episode: 127 Steps → terminated (crashed)
→ Episode ends early
→ Agent learns: Crash = bad
```

### Training Recommendations

**Option 1: With Crash Penalty (Recommended)**
```python
def _compute_reward(self) -> float:
    distance = np.linalg.norm(self.target_position - self.drone.position)
    reward = ((self.max_dist_to_target - distance) / self.max_dist_to_target) ** 2
    
    # Large penalty on crash
    if self._check_crash():
        reward -= 10.0
    
    return float(reward)
```

**Option 2: Without Extra Penalty**
```python
# Simply end episode
# Agent learns through shorter episode = less total reward
```

**Option 3: Disabled for Exploration**
```python
# At start of training
env = DroneEnv(enable_crash_detection=False)

# Enable later
env = DroneEnv(enable_crash_detection=True)
```

## Example: Crash-Safe Policy

```python
from src.drone_env import DroneEnv
import numpy as np

env = DroneEnv(enable_crash_detection=True)
obs, info = env.reset()

total_crashes = 0
total_episodes = 0

for episode in range(100):
    obs, info = env.reset()
    done = False
    
    while not done:
        # Your policy here
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if terminated and info['crashed']:
            total_crashes += 1
            print(f"Episode {episode}: CRASHED at step {info['step_count']}")
            break
    
    total_episodes += 1

crash_rate = total_crashes / total_episodes * 100
print(f"\nCrash Rate: {crash_rate:.1f}%")
```

## Adjustments

### More Tolerant Crash Detection (For Training)
```python
env = DroneEnv(
    crash_z_vel_threshold=-30.0,   # Lower (more negative)
    crash_tilt_threshold=85.0,     # Higher
)
```

### Stricter Crash Detection (For Evaluation)
```python
env = DroneEnv(
    crash_z_vel_threshold=-15.0,   # Higher (less negative)
    crash_tilt_threshold=70.0,     # Lower
)
```

### Only Z-Velocity (Simplest Variant)
```python
env = DroneEnv(
    crash_z_vel_threshold=-20.0,
    crash_tilt_threshold=180.0,    # Effectively disabled
)
```

## Future Extensions

Possible additional crash criteria:
- [ ] Ground collision (z-position < threshold)
- [ ] High angular velocity (out of control spinning)
- [ ] Collision with obstacles (when implemented)
- [ ] Timeout without movement (stuck/jammed)

---

**Status:** ✅ Fully implemented and tested
**Tests:** 5/5 passed

