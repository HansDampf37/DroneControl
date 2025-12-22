# Drohnen-RL Environment 🚁

Ein Gymnasium-kompatibles Reinforcement Learning Environment für Quadcopter-Steuerung.

## Features

- **Realistische Physik**: Vereinfachte Quadcopter-Physik mit 4 unabhängigen Motoren in X-Konfiguration
- **Dynamischer Wind**: Ornstein-Uhlenbeck-Prozess für realistische Windänderungen
- **Dense Reward**: `1/(1 + distance)` für effizientes Training
- **Visualisierung**: 2D Top-Down-Ansicht mit matplotlib
- **Gymnasium-kompatibel**: Standard RL-Interface für einfache Integration

## Installation

### Einfache Installation
```bash
pip install -r requirements.txt
```

### Entwickler-Installation (empfohlen)
```bash
pip install -e .
```

### Mit RL-Training-Unterstützung
```bash
pip install -e ".[rl]"
```

## Schnellstart

### Basis-Test
```python
from src.drone_env import DroneEnv

env = DroneEnv(max_steps=1000, render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Zufällige Aktion
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        break

env.close()
```

### Tests ausführen
```bash
# Alle Tests
python tests/test_env.py

# Rendering-Test
python tests/test_rendering.py

# Minimaler Test
python tests/test_minimal_render.py

# Debug-Informationen
python tests/debug_render.py
```

### Beispiele ausführen
```bash
# Random Agent (ohne Visualisierung)
python examples/random_agent.py --episodes 5 --steps 500

# Random Agent (mit Visualisierung)
python examples/random_agent.py --episodes 3 --steps 500 --render

# Hover Agent (Baseline)
python examples/random_agent.py --agent hover --episodes 3 --render

# RL-Training (benötigt stable-baselines3)
python examples/training.py --mode train --algorithm PPO --timesteps 100000

# Modell evaluieren
python examples/training.py --mode eval --model-path models/drone_model
```

## Projektstruktur

```
drone-control/
├── src/
│   └── drone_env/
│       ├── __init__.py          # Package-Initialisierung
│       └── env.py               # DroneEnv Klasse
│
├── tests/
│   ├── __init__.py
│   ├── test_env.py              # Umfassende Tests
│   ├── test_rendering.py        # Rendering-Tests
│   ├── test_minimal_render.py   # Minimaler Rendering-Test
│   └── debug_render.py          # Debug-Informationen
│
├── examples/
│   ├── __init__.py
│   ├── random_agent.py          # Random/Hover Agent Demo
│   └── training.py              # SB3 Training & Evaluation
│
├── setup.py                     # Package-Setup
├── requirements.txt             # Dependencies
├── README.md                    # Diese Datei
└── .gitignore                   # Git Ignore
```

## Environment Details

### Action Space
- **Typ**: `Box(4,)` 
- **Wertebereich**: [0, 1] pro Motor (0% - 100% Thrust)
- **Beschreibung**: 
  - Motor 0: Vorne-Rechts
  - Motor 1: Hinten-Links
  - Motor 2: Vorne-Links
  - Motor 3: Hinten-Rechts

### Observation Space
- **Typ**: `Box(15,)`
- **Komponenten**:
  - `[0:3]` - Position relativ zum Ziel (x, y, z)
  - `[3:6]` - Lineare Geschwindigkeit (vx, vy, vz)
  - `[6:9]` - Orientierung (Roll, Pitch, Yaw in Radiant)
  - `[9:12]` - Winkelgeschwindigkeit (wx, wy, wz)
  - `[12:15]` - Windvektor absolut (wx, wy, wz)

**Hinweis**: Die Drohne ist in der Beobachtung immer bei Position (0, 0, 0). Der Zielpunkt wird relativ angegeben.

### Reward
```python
reward = 1.0 / (1.0 + distance_to_target)
```
- **Wertebereich**: (0, 1]
- **Maximum**: 1.0 (Drohne am Ziel)
- **Eigenschaften**: Dense, kontinuierlich, differenzierbar

### Termination
- **Terminated**: Immer `False` (keine Crash-Detektion)
- **Truncated**: `True` nach `max_steps` Schritten
- **Standard**: 1000 Steps (~10 Sekunden bei 100 Hz)

## Physik-Modell

### Quadcopter-Konfiguration
- **X-Konfiguration**: Rotoren diagonal angeordnet (±45° zu Achsen)
- **Masse**: 1.0 kg
- **Arm-Länge**: 0.25 m
- **Trägheitsmomente**: [0.01, 0.01, 0.02] kg·m²

### Kraftberechnung
1. **Thrust**: Kraftvektor senkrecht zur Rotorebene, skaliert mit Motor-Power
2. **Drehmoment**:
   - **Roll**: Thrust-Differenz zwischen linken/rechten Motoren
   - **Pitch**: Thrust-Differenz zwischen vorderen/hinteren Motoren
   - **Yaw**: Reaktives Drehmoment aus Rotor-Drehrichtungen
3. **Wind**: Kraftvektor proportional zur Windgeschwindigkeit
4. **Gravitation**: -9.81 m/s² in Z-Richtung

### Integration
- **Methode**: Euler-Integration
- **Zeitschritt**: 0.01 s (100 Hz Standard)

## Konfigurationsparameter

```python
from src.drone_env import DroneEnv

env = DroneEnv(
    max_steps=1000,                          # Episode-Länge
    dt=0.01,                                 # Zeitschritt in Sekunden
    target_change_interval=None,             # Ziel-Änderung (None = fix)
    wind_strength_range=(0.0, 5.0),         # Wind-Stärke in m/s
    render_mode="human"                      # "human", "rgb_array", oder None
)
```

## Entwicklung

### Paket installieren (editable mode)
```bash
pip install -e .
```

### Tests ausführen
```bash
# Alle Tests
python tests/test_env.py

# Spezifischer Test
python tests/test_minimal_render.py
```

### Code formatieren (optional)
```bash
pip install -e ".[dev]"
black src/ tests/ examples/
flake8 src/ tests/ examples/
```

## Zukünftige Erweiterungen

- [ ] 3D-Visualisierung (PyVista oder Pygame)
- [ ] Optionale Windrichtung aus Observation Space entfernen (für recurrent policies)
- [ ] Mehrere Zielpunkte pro Episode
- [ ] Hindernisse
- [ ] Crash-Detektion und -Penalties
- [ ] Energieverbrauch als Teil der Reward-Funktion
- [ ] Aerodynamische Effekte (Luftwiderstand, Downwash)

## Trainings-Tipps

### Baseline
Ein einfacher Hover-Agent (alle Motoren auf ~25%) erreicht einen durchschnittlichen Reward von ~0.05-0.10, abhängig von der initialen Ziel-Distanz.

### Empfohlene Algorithmen
- **SAC** (Soft Actor-Critic): Gut für kontinuierliche Actions
- **PPO** (Proximal Policy Optimization): Stabil und sample-effizient
- **TD3** (Twin Delayed DDPG): Für deterministische Policies

### Recurrent Policies
Um die Drohne Wind "spüren" zu lassen:
1. Entferne `wind_vector` aus Observation Space (Index 12:15)
2. Verwende LSTM/GRU-basierte Policy
3. Drohne muss Wind aus Positions-/Geschwindigkeits-Historie inferieren

## Lizenz

MIT

## Autor

Adrian - 2025

