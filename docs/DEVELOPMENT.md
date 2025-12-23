# Development Guide

Developer documentation for the Drone RL Environment.

## Project Structure

```
drone-control/
│
├── src/                              # Production code
│   └── drone_env/
│       ├── __init__.py               # Exports DroneEnv
│       ├── drone.py                  # Drone physics model
│       ├── env.py                    # Main environment
│       └── renderer.py               # Visualization
│
├── tests/                            # Tests & debugging
│   ├── __init__.py
│   ├── test_env.py                   # Comprehensive tests (4 test suites)
│   ├── test_rendering.py             # Rendering tests (200 steps)
│   ├── test_minimal_render.py        # Minimal test (20 steps)
│   └── debug_render.py               # Debug information
│
├── examples/                         # Example scripts
│   ├── __init__.py
│   ├── random_agent.py               # Random/Hover agent demo
│   └── training.py                   # Ray RLlib training
│
├── docs/                             # Documentation
│   ├── DEVELOPMENT.md                # This file
│   ├── TROUBLESHOOTING.md            # Problem solutions
│   ├── CRASH_DETECTION.md            # Crash detection system
│   └── VISUALIZATION.md              # Rendering details
│
├── setup.py                          # Package installation
├── requirements.txt                  # Dependencies
└── README.md                         # Main documentation
```

## Installation for Development

### Editable Mode (recommended)
```bash
# Changes to code take effect immediately
pip install -e .

# With RL support
pip install -e ".[rl]"

# With developer tools
pip install -e ".[dev]"
```

## Code Organization

### Design Principles

1. **Separation of Concerns**
   - `src/` - Production code only
   - `tests/` - All tests
   - `examples/` - User examples

2. **Package Structure**
   - Each directory has `__init__.py`
   - Import via package names
   - Installable via `setup.py`

3. **Imports**
   ```python
   # In tests/examples
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   from src.drone_env import DroneEnv
   
   # After installation
   from src.drone_env import DroneEnv
   ```

## Running Tests

### All Tests
```bash
# Comprehensive test suite
python tests/test_env.py
```

Output:
- Test 1: Basic functionality
- Test 2: Physics simulation (hover, full thrust, roll)
- Test 3: Wind dynamics
- Test 4: Reward function
- Demo with visualization (optional)

### Rendering Tests
```bash
# Quick test (20 steps)
python tests/test_minimal_render.py

# Full test (200 steps with various actions)
python tests/test_rendering.py

# Debug information (figure tracking)
python tests/debug_render.py
```

## Adding a New Test

1. **Create file** in `tests/`
2. **Use import template**:
   ```python
   """Description of the test."""
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   
   import numpy as np
   from src.drone_env import DroneEnv
   
   def test_my_feature():
       env = DroneEnv()
       obs, info = env.reset()
       # ... Test code ...
       env.close()
   
   if __name__ == "__main__":
       test_my_feature()
   ```

## Adding a New Example

1. **Create file** in `examples/`
2. **Use import structure** as in tests
3. **CLI arguments** with `argparse` (optional)
4. **Document** in README.md

Example:
```python
"""Example: My Agent."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.drone_env import DroneEnv

def run_my_agent():
    env = DroneEnv(render_mode="human")
    # ... Agent code ...
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5)
    args = parser.parse_args()
    run_my_agent()
```

## Extending the Environment

### Adding New Features to DroneEnv

Edit `src/drone_env/env.py`:

#### Adding New Parameters
```python
def __init__(
    self,
    max_steps: int = 1000,
    my_new_parameter: float = 1.0,  # NEW
    # ...
):
    self.my_new_parameter = my_new_parameter
```

#### Changing Physics Parameters
Lines ~85-95 in DroneEnv.__init__():

```python
self.drone = Drone(
    mass=1.0,               # kg
    arm_length=0.25,        # m
    thrust_coef=10.0,       # Thrust = coeff * motor_power
    torque_coef=0.1,        # Torque = coeff * motor_power
)
self.gravity = 9.81  # m/s^2
```

#### Modifying Observation Space
Lines ~270-280 in _get_observation():
```python
# Example: Remove wind from observation
observation = np.concatenate([
    vect_to_target,
    self.drone.velocity,
    self.drone.orientation,
    self.drone.angular_velocity,
    # self.wind_vector,  # <-- Comment out to remove
])
```

**Important**: Adjust observation space bounds in `__init__()` accordingly!

### Changing the Reward Function

Line ~290 in _compute_reward():
```python
def _compute_reward(self) -> float:
    """Computes the reward."""
    distance = np.linalg.norm(self.target_position - self.drone.position)
    
    # Current: Dense Reward
    reward = ((self.max_dist_to_target - distance) / self.max_dist_to_target) ** 2
    
    # Alternative: Sparse Reward
    # reward = 1.0 if distance < 1.0 else 0.0
    
    # Alternative: With Stability Bonus
    # stability_penalty = np.linalg.norm(self.drone.angular_velocity)
    # reward = ((self.max_dist_to_target - distance) / self.max_dist_to_target) ** 2
    # reward -= 0.1 * stability_penalty
    
    return float(reward)
```

## Code Style

### Formatting (optional)
```bash
pip install black flake8 mypy

# Format code
black src/ tests/ examples/

# Linting
flake8 src/ tests/ examples/

# Type-Checking
mypy src/
```

### Conventions
- Docstrings for all classes/functions
- Type hints where possible
- Comments for complex logic
- Constants in UPPERCASE

## Debugging

### Print Debugging
```python
# In env.py, e.g., in step()
print(f"Action: {action}")
print(f"Position: {self.drone.position}")
print(f"Thrust: {thrusts}")
```

### Interactive Debugging
```python
# In your script
import pdb

env = DroneEnv()
obs, info = env.reset()

pdb.set_trace()  # Breakpoint
obs, reward, term, trunc, info = env.step(action)
```

### Rendering Debug
```bash
# Shows figure numbers, matplotlib state
python tests/debug_render.py
```

## Performance Optimization

### Profiling
```python
import cProfile
import pstats

env = DroneEnv()
profiler = cProfile.Profile()

profiler.enable()
# ... Run code ...
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Rendering Performance
- `render_mode=None` for training (no rendering)
- Reduce rendering frequency: `if step % 10 == 0: env.render()`

## Versioning

### Git Workflow
```bash
# Feature Branch
git checkout -b feature/my-new-feature

# Commit changes
git add src/drone_env/env.py
git commit -m "Add: New feature description"

# Push
git push origin feature/my-new-feature
```

### Semantic Versioning
- `MAJOR.MINOR.PATCH` (e.g., 0.1.0)
- MAJOR: Breaking Changes
- MINOR: New Features (backwards compatible)
- PATCH: Bug Fixes

Update in:
- `src/drone_env/__init__.py`
- `setup.py`

## Dependency Management

### Updating requirements.txt
```bash
# After new pip install:
pip freeze > requirements.txt

# Or edit manually (recommended):
# Only direct dependencies with version ranges
```

### Updating setup.py
Edit `install_requires` for new dependencies:
```python
install_requires=[
    "gymnasium>=0.29.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "scipy>=1.10.0",
    "your-new-package>=1.0.0",
],
```

## Continuous Integration (future)

### GitHub Actions Example
```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - run: pip install -e .
      - run: python tests/test_env.py
```

## Common Developer Tasks

### Reset Environment to Defaults
```python
env = DroneEnv()  # Uses all default values
```

### Deterministic Episodes
```python
env.reset(seed=42)  # Same starting conditions
```

### Set Custom Target Point
```python
env.reset()
env.target_position = np.array([10.0, 5.0, 0.0])
```

### Change Physics Parameters at Runtime

```python
env = DroneEnv()
env.drone.mass = 1.5  # Heavier drone
env.drone.thrust_coef = 12.0  # Stronger motors
```

## Contact & Contribution

For questions or suggestions:
- Create issues in the GitHub repository
- Pull requests are welcome!
- Follow the code style guidelines
- Add tests for new features
- Update documentation

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`python tests/test_env.py`)
5. Commit your changes (`git commit -m 'Add: Amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## See Also

- [README.md](../README.md) - Main documentation
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
- [CRASH_DETECTION.md](CRASH_DETECTION.md) - Crash detection system details
- [VISUALIZATION.md](VISUALIZATION.md) - Rendering and visualization guide

