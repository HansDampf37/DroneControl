# Pygame 3D Renderer

## Overview

The pygame-based 3D renderer is a high-performance alternative to the original matplotlib renderer. It provides significantly better rendering performance (typically 5-10x faster) while maintaining clear visualization of the drone's state.

## Performance Comparison

| Renderer | Typical FPS | Best Use Case |
|----------|-------------|---------------|
| Matplotlib | 10-12 FPS | Detailed multi-view analysis, debugging |
| Pygame | 60+ FPS | Real-time visualization, training monitoring |

## Features

The pygame renderer displays the drone environment in a single 3D perspective view with:

### Visual Elements

1. **Drone Representation**
   - Central blue sphere: Main body (8cm radius)
   - 4 colored spheres: Rotors (4cm radius each)
     - Red spheres: Clockwise-rotating motors (M0, M2)
     - Green spheres: Counter-clockwise motors (M1, M3)
   - Gray lines: Rotor arms connecting body to rotors

2. **Orientation Indicators**
   - **Orange arrow**: "Up" direction (drone's local Z-axis)
     - Shows which way is "up" for the drone
     - Length scales with orientation
   - **Magenta arrow**: Velocity direction
     - Only visible when velocity > 0.1 m/s
     - Shows the direction the drone is moving

3. **Target**
   - Green sphere with crosshair
   - Dashed line connecting drone to target

4. **Grid and Ground**
   - Brown grid on the ground plane (Z=0)
   - Grid spacing: 0.5 meters
   - Helps visualize drone position and height

5. **Wind Indicator**
   - Red arrow in the corner (when wind > 0.1 m/s)
   - Shows wind direction and relative strength

6. **Info Panel** (top-left)
   - Time and step count
   - Distance to target and altitude
   - Velocity magnitude and wind strength
   - Orientation angles (roll, pitch, yaw)
   - Custom metrics from kwargs

7. **Motor Thrust Bars** (bottom-right)
   - 4 vertical bars showing thrust level for each motor
   - Color gradient: red (low) to yellow (high)
   - Percentage display above each bar

### Camera View

The renderer uses an isometric-like 3D projection with:
- **Horizontal angle**: 45° (adjustable via `camera_angle_h`)
- **Vertical angle**: 30° (adjustable via `camera_angle_v`)
- **Distance**: 2.5x the space side length

This provides a clear view of all three spatial dimensions.

## Usage

### Basic Usage

```python
from src.drone_env import DroneEnv

# Create environment with pygame renderer (default)
env = DroneEnv(
    render_mode="human",
    renderer_type="pygame"  # This is now the default
)

obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Using Matplotlib Renderer

If you prefer the original matplotlib multi-view renderer:

```python
env = DroneEnv(
    render_mode="human",
    renderer_type="matplotlib"
)
```

### RGB Array Mode (for Recording)

```python
env = DroneEnv(
    render_mode="rgb_array",
    renderer_type="pygame"
)

obs, info = env.reset()
frames = []

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Get RGB array
    frame = env.render()
    frames.append(frame)

# Save frames as video using cv2, imageio, etc.
```

## Running the Demo

A demo script is provided to showcase the pygame renderer:

```bash
# Random agent demo
python examples/demo_pygame_renderer.py

# Hovering agent demo (more stable flight)
python examples/demo_pygame_renderer.py hover
```

## Performance Testing

To benchmark the performance difference between renderers:

```bash
python tests/test_pygame_renderer.py
```

This will run both renderers for a set number of steps and compare their FPS.

## Implementation Details

### 3D Projection

The renderer uses a simple 3D-to-2D projection pipeline:

1. **Camera rotation**: Apply horizontal and vertical rotations
2. **Orthographic projection**: Map 3D coordinates to 2D screen space
3. **Scaling**: 200 pixels per meter

### Rendering Order

Objects are drawn in the following order (back to front):
1. Grid and ground plane
2. Connection line (drone to target)
3. Target sphere and crosshair
4. Drone rotors (with connecting arms)
5. Drone body
6. Orientation arrows
7. UI elements (info panel, motor bars)

### Anti-aliasing

The renderer uses pygame's gfxdraw for anti-aliased circles, providing smooth sphere rendering without performance degradation.

### Shading

Simple highlight shading is applied to spheres to give a 3D appearance:
- Highlight offset: -25% radius in X and Y
- Highlight color: +50 brightness to base color

## Customization

### Adjusting Camera View

You can modify the camera angles in the renderer after initialization:

```python
env = DroneEnv(render_mode="human", renderer_type="pygame")
env.renderer.camera_angle_h = 30  # Horizontal angle (degrees)
env.renderer.camera_angle_v = 45  # Vertical angle (degrees)
env.renderer.camera_distance = 8  # Distance from origin
```

### Changing Colors

Colors are defined as class attributes and can be modified:

```python
env.renderer.DRONE_BODY_COLOR = (255, 0, 0)  # Red drone body
env.renderer.BG_COLOR = (0, 0, 0)  # Black background
```

### Window Size

Adjust the rendering resolution:

```python
env.renderer.screen_width = 1600
env.renderer.screen_height = 1200
```

Note: These changes should be made before the first render() call.

## Troubleshooting

### No Display Window Appears

Make sure you're using `render_mode="human"`:
```python
env = DroneEnv(render_mode="human", renderer_type="pygame")
```

### Low FPS

- Check if other applications are using significant GPU resources
- Try reducing the window size
- Ensure pygame is properly installed: `pip install pygame --upgrade`

### Window Closes Immediately

Make sure you're calling `env.render()` in your loop and not calling `env.close()` too early.

### Black Screen

This may occur if the drone is far from the origin. The camera is centered at (0, 0, 0), so ensure your drone spawns within the observable space.

## Future Enhancements

Potential improvements for the pygame renderer:

- [ ] Interactive camera controls (mouse drag to rotate)
- [ ] Zoom controls (mouse wheel)
- [ ] Trail visualization (show drone's path)
- [ ] Multiple camera views (split screen)
- [ ] Lighting and better shading
- [ ] Actual 3D rendering using PyOpenGL
- [ ] Recording functionality built-in
- [ ] Performance statistics overlay

## Technical Limitations

1. **Projection**: Currently uses a simple isometric-like projection, not true perspective
2. **Z-ordering**: Objects don't use proper depth sorting, may have rendering artifacts in some angles
3. **Lighting**: No real lighting model, just simple highlights
4. **Collision**: No visual feedback for collisions/crashes beyond termination

Despite these limitations, the renderer provides excellent real-time visualization performance for training and debugging.

