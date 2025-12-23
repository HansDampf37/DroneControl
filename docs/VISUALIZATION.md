# Visualization Update

## New Features in 2D Visualization

### Rotor Representation with Complete 3D Rotation
The 4 rotors of the drone (X-configuration) are now correctly transformed:

- **4 colored circles**: Rotor positions (projected onto XY plane)
  - 🔴 **Red**: CW-rotating rotors (Motor 0, 1)
  - 🟢 **Green**: CCW-rotating rotors (Motor 2, 3)
- **Gray lines**: Connections from center to rotors
- **3D Transformation**: Rotor positions are transformed with **Roll, Pitch AND Yaw**
- **XY Projection**: The transformed 3D positions are projected onto the XY plane

### How It Works

**Body-Frame → World-Frame → XY-Projection:**

1. **Body-Frame**: Rotors at fixed positions (X-configuration, ±45°, arm length 0.25m)
2. **Rotation**: Complete 3D rotation with Roll, Pitch, Yaw
3. **Projection**: XY components of rotated positions are drawn

```python
# For each rotor:
rotor_pos_world = R @ rotor_pos_body  # 3D rotation
rotor_xy = [rotor_pos_world[0], rotor_pos_world[1]]  # XY projection
```

### Visible Effects

#### Yaw Only (Rotation around Z-axis)
```
Yaw = 0°:           Yaw = 45°:
    ○                   ○
    |                  / \
 ○--●--○            ○--●--○
    |                  \ /
    ○                   ○
```
The X rotates but remains symmetrical.

#### Roll (Sideways Tilt)
```
Roll = 0°:          Roll > 0°:
    ○                   ○
    |                   |
 ○--●--○             ○-●  ○
    |                   |
    ○                   ○
```
Right arms appear shorter (tilted away from viewer).

#### Pitch (Forward/Backward Tilt)
```
Pitch = 0°:         Pitch < 0°:
    ○                   ○
    |                   |
 ○--●--○             ○--●--○
    |                   ○
    ○
```
Front arms appear shorter (tilted forward).

#### Combined (Roll + Pitch + Yaw)
The X-shape becomes asymmetric - different arm lengths show the 3D tilt!

### Tilt Indicator
An **orange arrow** shows the tilt direction of the drone:

- **Calculation**: Projection of the drone's normal vector onto the XY plane
- **Meaning**: Shows which direction the drone is "tipping"
- **Visibility**: Only when tilt is significant (>0.01 rad)

### Extended Info Box
The info box now also displays:
- **Roll**: Rotation around X-axis (in degrees)
- **Pitch**: Rotation around Y-axis (in degrees)  
- **Yaw**: Rotation around Z-axis (in degrees)

## Visualization Elements

```
     Rotor 2 (Green, CCW)
           ○
           |
    Motor 0 ○----●----○ Motor 3
   (Red,CW)      |      (Red,CW)
                 |
                 ○
           Rotor 1 (Green, CCW)

    ● = Drone center (blue)
    ○ = Rotor (red/green)
    → = Tilt arrow (orange)
```

## Interpretation

### No Tilt (Hover)
- No orange arrow visible
- All rotors evenly spaced from center
- Roll ≈ 0°, Pitch ≈ 0°

### Roll to the Right
- Orange arrow points right
- Roll > 0°
- Left rotors (2, 1) higher, right rotors (0, 3) lower

### Pitch Forward
- Orange arrow points forward (in flight direction)
- Pitch < 0°
- Rear rotors (1, 3) higher, front rotors (0, 2) lower

### Combined Tilt
- Orange arrow points in diagonal direction
- Roll ≠ 0°, Pitch ≠ 0°

## Testing

```bash
python tests/test_visualization.py
```

This test demonstrates various maneuvers:
1. Hover (no tilt)
2. Roll right
3. Pitch forward
4. Combined
5. Return to hover

## Technical Details

### Rotor Positions (X-Configuration)
```python
# Body-frame angles (before Yaw rotation)
Motor 0: +45°  (front-right, CW)
Motor 1: -135° (rear-left, CW)
Motor 2: +135° (front-left, CCW)
Motor 3: -45°  (rear-right, CCW)
```

### Tilt Calculation
```python
# Normal in body-frame
normal_body = [0, 0, 1]

# Rotation to world-frame
R = get_rotation_matrix(roll, pitch, yaw)
normal_world = R @ normal_body

# Projection onto XY
tilt_x = normal_world[0]
tilt_y = normal_world[1]
```

### Color Scheme
- **Drone Center**: Blue (#0066cc)
- **CW Rotors**: Red (#ff6666)
- **CCW Rotors**: Green (#66ff66)
- **Rotor Arms**: Gray (#666666)
- **Tilt Arrow**: Orange (#ff9900)
- **Target**: Green (#00cc00)
- **Wind**: Red (#cc0000)

## Future Extensions

Possible additional visualization features:
- [ ] Motor thrust as circle size or color intensity
- [ ] Trajectory (path of last N positions)
- [ ] 3D view with actual rotor heights
- [ ] Wind effect as particles
- [ ] Velocity vector

