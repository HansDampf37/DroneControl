# Drohnen-RL Environment 🚁

Ein Gymnasium-kompatibles Reinforcement Learning Environment für Quadcopter-Steuerung in Python.

## Inhaltsverzeichnis

- [Features](#features)
- [Schnellstart](#schnellstart)
- [Environment Details](#environment-details)
- [Physik-Modell](#physik-modell)
- [Konfiguration](#konfiguration)
- [RL-Training](#rl-training)
- [Projektstruktur](#projektstruktur)
- [Weitere Dokumentation](#weitere-dokumentation)

## Features

- **Realistische Physik**: Vereinfachte Quadcopter-Physik mit 4 unabhängigen Motoren in X-Konfiguration
- **Dynamischer Wind**: Ornstein-Uhlenbeck-Prozess für realistische Windänderungen
- **Dense Reward**: `1/(1 + distance)` für effizientes Training
- **2D-Visualisierung**: Top-Down-Ansicht mit matplotlib
- **Gymnasium-kompatibel**: Standard RL-Interface

## Schnellstart

### Installation
```bash
# Clone Repository
git clone <your-repo-url>
cd drone-control

# Installiere Dependencies
pip install -r requirements.txt

# Optional: Entwickler-Installation
pip install -e .
```

### Einfacher Test
```bash
# Minimaler Rendering-Test (20 Steps)
python tests/test_minimal_render.py

# Umfassende Tests
python tests/test_env.py
```

### Beispiel: Random Agent
```bash
# Ohne Visualisierung
python examples/random_agent.py --episodes 5

# Mit Visualisierung
python examples/random_agent.py --episodes 3 --render
```

### Verwendung im Code
```python
from src.drone_env import DroneEnv

# Environment erstellen
env = DroneEnv(max_steps=1000, render_mode="human")
obs, info = env.reset()

# Haupt-Loop
for _ in range(1000):
    action = env.action_space.sample()  # Zufällige Aktion
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Environment Details

### Action Space
- **Typ**: `Box(4,)` mit Wertebereich [0, 1]
- **Beschreibung**: 4 Motoren (0-100% Thrust)
  - Motor 0: Vorne-Rechts
  - Motor 1: Hinten-Links  
  - Motor 2: Vorne-Links
  - Motor 3: Hinten-Rechts

### Observation Space
- **Typ**: `Box(15,)`
- **Komponenten**:
  - `[0:3]` - Relative Position zum Ziel (x, y, z)
  - `[3:6]` - Lineare Geschwindigkeit (vx, vy, vz)
  - `[6:9]` - Orientierung (Roll, Pitch, Yaw)
  - `[9:12]` - Winkelgeschwindigkeit (wx, wy, wz)
  - `[12:15]` - Windvektor absolut (wx, wy, wz)

**Hinweis**: Die Drohne ist in der Beobachtung immer bei Position (0, 0, 0). Der Zielpunkt wird relativ angegeben.

### Reward Function
```python
reward = 1.0 / (1.0 + distance_to_target)
```
- Wertebereich: (0, 1]
- Dense Reward für besseres Training

### Termination
- Episode endet nach festen `max_steps` (Standard: 1000)
- Keine Crash-Detektion

## Physik-Modell

### Quadcopter X-Konfiguration
- 4 Rotoren diagonal angeordnet (±45° zu Achsen)
- Masse: 1.0 kg
- Arm-Länge: 0.25 m

### Kraftberechnung
1. **Thrust**: Kraftvektor senkrecht zur Rotorebene, skaliert mit Motor-Power
2. **Drehmoment**:
   - Roll: Thrust-Differenz zwischen linken/rechten Motoren
   - Pitch: Thrust-Differenz zwischen vorderen/hinteren Motoren
   - Yaw: Reaktives Drehmoment aus Rotor-Drehrichtungen
3. **Wind**: Ornstein-Uhlenbeck-Prozess, anpassbar
4. **Integration**: Euler-Integration mit 0.01s Zeitschritt (100 Hz)

## Konfiguration

```python
env = DroneEnv(
    max_steps=1000,                      # Episode-Länge
    dt=0.01,                             # Zeitschritt (s)
    target_change_interval=None,         # Ziel-Änderung (None = fix)
    wind_strength_range=(0.0, 5.0),     # Wind-Stärke (m/s)
    render_mode="human"                  # "human", "rgb_array", None
)
```

## RL-Training

### Beispiel mit Stable-Baselines3
```bash
# Installation
pip install stable-baselines3[extra]

# Training
python examples/training.py --mode train --algorithm PPO --timesteps 100000

# Evaluation
python examples/training.py --mode eval --model-path models/drone_model
```

### Empfohlene Algorithmen
- **PPO**: Stabil, gut für Einstieg
- **SAC**: Sehr gute Performance bei kontinuierlichen Actions
- **TD3**: Alternative zu SAC

### Baseline Performance
- **Hover Agent** (alle Motoren ~25%): Reward ~0.05-0.10
- **Trainierter Agent**: Reward >0.3-0.5 nach 100k Steps

## Projektstruktur

```
drone-control/
├── src/drone_env/          # Haupt-Environment
│   ├── __init__.py
│   └── env.py
├── tests/                  # Tests & Debugging
│   ├── test_env.py
│   ├── test_rendering.py
│   └── ...
├── examples/               # Beispiel-Skripte
│   ├── random_agent.py
│   └── training.py
├── docs/                   # Dokumentation
│   ├── DEVELOPMENT.md      # Entwickler-Guide
│   └── TROUBLESHOOTING.md  # Häufige Probleme
├── setup.py
├── requirements.txt
└── README.md              # Diese Datei
```

## Weitere Dokumentation

- **[Development Guide](docs/DEVELOPMENT.md)** - Entwickler-Informationen, Struktur, Erweiterungen
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Häufige Probleme, insbesondere Rendering

## Roadmap

- [ ] 3D-Visualisierung
- [ ] Recurrent Policies (Wind-Inferenz ohne direkte Observation)
- [ ] Mehrere Zielpunkte pro Episode
- [ ] Hindernisse
- [ ] Crash-Detektion
- [ ] Energieverbrauch in Reward

## Lizenz

MIT

## Autor

Adrian - 2025

