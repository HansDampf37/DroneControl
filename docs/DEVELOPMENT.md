# Development Guide

Entwickler-Dokumentation für das Drohnen-RL Environment.

## Projektstruktur

```
drone-control/
│
├── src/                              # Produktivcode
│   └── drone_env/
│       ├── __init__.py               # Exportiert DroneEnv
│       └── env.py                    # Haupt-Environment (481 Zeilen)
│
├── tests/                            # Tests & Debugging
│   ├── __init__.py
│   ├── test_env.py                   # Umfassende Tests (4 Test-Suites)
│   ├── test_rendering.py             # Rendering-Tests (200 Steps)
│   ├── test_minimal_render.py        # Minimaler Test (20 Steps)
│   └── debug_render.py               # Debug-Informationen
│
├── examples/                         # Beispiel-Skripte
│   ├── __init__.py
│   ├── random_agent.py               # Random/Hover Agent Demo
│   └── training.py                   # Stable-Baselines3 Training
│
├── docs/                             # Dokumentation
│   ├── DEVELOPMENT.md                # Diese Datei
│   └── TROUBLESHOOTING.md            # Problemlösungen
│
├── setup.py                          # Package-Installation
├── requirements.txt                  # Dependencies
└── README.md                         # Haupt-Dokumentation
```

## Installation für Entwicklung

### Editable Mode (empfohlen)
```bash
# Änderungen am Code werden sofort wirksam
pip install -e .

# Mit RL-Support
pip install -e ".[rl]"

# Mit Entwickler-Tools
pip install -e ".[dev]"
```

## Code-Organisation

### Design-Prinzipien

1. **Trennung von Concerns**
   - `src/` - Nur Produktivcode
   - `tests/` - Alle Tests
   - `examples/` - Nutzer-Beispiele

2. **Package-Struktur**
   - Jedes Verzeichnis hat `__init__.py`
   - Import via Package-Namen
   - Installierbar via `setup.py`

3. **Imports**
   ```python
   # In Tests/Beispielen
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   from src.drone_env import DroneEnv
   
   # Nach Installation
   from src.drone_env import DroneEnv
   ```

## Tests ausführen

### Alle Tests
```bash
# Umfassende Test-Suite
python tests/test_env.py
```

Output:
- Test 1: Grundlegende Funktionalität
- Test 2: Physik-Simulation (Hover, Full-Thrust, Roll)
- Test 3: Wind-Dynamik
- Test 4: Reward-Funktion
- Demo mit Visualisierung (optional)

### Rendering-Tests
```bash
# Schneller Test (20 Steps)
python tests/test_minimal_render.py

# Vollständiger Test (200 Steps mit verschiedenen Actions)
python tests/test_rendering.py

# Debug-Informationen (Figure-Tracking)
python tests/debug_render.py
```

## Neuen Test hinzufügen

1. **Datei erstellen** in `tests/`
2. **Import-Template** verwenden:
   ```python
   """Beschreibung des Tests."""
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   
   import numpy as np
   from src.drone_env import DroneEnv
   
   def test_my_feature():
       env = DroneEnv()
       obs, info = env.reset()
       # ... Test-Code ...
       env.close()
   
   if __name__ == "__main__":
       test_my_feature()
   ```

## Neues Beispiel hinzufügen

1. **Datei erstellen** in `examples/`
2. **Import-Struktur** wie bei Tests
3. **CLI-Argumente** mit `argparse` (optional)
4. **Dokumentieren** in README.md

Beispiel:
```python
"""Beispiel: Mein Agent."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.drone_env import DroneEnv

def run_my_agent():
    env = DroneEnv(render_mode="human")
    # ... Agent-Code ...
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5)
    args = parser.parse_args()
    run_my_agent()
```

## Environment erweitern

### Neue Features in DroneEnv

Editiere `src/drone_env/env.py`:

#### Neue Parameter hinzufügen
```python
def __init__(
    self,
    max_steps: int = 1000,
    my_new_parameter: float = 1.0,  # NEU
    # ...
):
    self.my_new_parameter = my_new_parameter
```

#### Physik-Parameter ändern
Zeilen ~45-60:
```python
self.mass = 1.0              # kg
self.arm_length = 0.25       # m
self.thrust_coeff = 10.0     # Thrust = coeff * motor_power
self.torque_coeff = 0.1      # Torque = coeff * motor_power
self.gravity = 9.81          # m/s^2
```

#### Observation Space modifizieren
Zeilen ~90-115:
```python
# Beispiel: Wind aus Observation entfernen
observation = np.concatenate([
    relative_target,
    self.velocity,
    self.orientation,
    self.angular_velocity,
    # self.wind_vector,  # <-- Auskommentieren
])
```

**Wichtig**: Observation Space Grenzen anpassen!

### Reward-Funktion ändern

Zeile ~310:
```python
def _compute_reward(self) -> float:
    """Berechnet den Reward."""
    distance = np.linalg.norm(self.target_position - self.position)
    
    # Aktuell: Dense Reward
    reward = 1.0 / (1.0 + distance)
    
    # Alternative: Sparse Reward
    # reward = 1.0 if distance < 1.0 else 0.0
    
    # Alternative: Mit Stability-Bonus
    # stability_penalty = np.linalg.norm(self.angular_velocity)
    # reward = 1.0 / (1.0 + distance) - 0.1 * stability_penalty
    
    return float(reward)
```

## Code-Style

### Formatierung (optional)
```bash
pip install black flake8 mypy

# Formatieren
black src/ tests/ examples/

# Linting
flake8 src/ tests/ examples/

# Type-Checking
mypy src/
```

### Konventionen
- Docstrings für alle Klassen/Funktionen
- Type Hints wo möglich
- Kommentare für komplexe Logik
- Konstanten in UPPERCASE

## Debugging

### Print-Debugging
```python
# In env.py, z.B. in step()
print(f"Action: {action}")
print(f"Position: {self.position}")
print(f"Thrust: {thrusts}")
```

### Interactive Debugging
```python
# In deinem Script
import pdb

env = DroneEnv()
obs, info = env.reset()

pdb.set_trace()  # Breakpoint
obs, reward, term, trunc, info = env.step(action)
```

### Rendering Debug
```bash
# Zeigt Figure-Nummern, matplotlib State
python tests/debug_render.py
```

## Performance-Optimierung

### Profiling
```python
import cProfile
import pstats

env = DroneEnv()
profiler = cProfile.Profile()

profiler.enable()
# ... Code ausführen ...
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Rendering Performance
- `render_mode=None` für Training (kein Rendering)
- Rendering-Frequenz reduzieren: `if step % 10 == 0: env.render()`

## Versionierung

### Git Workflow
```bash
# Feature Branch
git checkout -b feature/my-new-feature

# Änderungen committen
git add src/drone_env/env.py
git commit -m "Add: Neue Feature-Beschreibung"

# Pushen
git push origin feature/my-new-feature
```

### Semantic Versioning
- `MAJOR.MINOR.PATCH` (z.B. 0.1.0)
- MAJOR: Breaking Changes
- MINOR: Neue Features (backwards compatible)
- PATCH: Bug Fixes

Aktualisiere in:
- `src/drone_env/__init__.py`
- `setup.py`

## Dependency Management

### requirements.txt aktualisieren
```bash
# Nach neuen pip install:
pip freeze > requirements.txt

# Oder manuell editieren (empfohlen):
# Nur direkte Dependencies mit Version-Ranges
```

### setup.py aktualisieren
Editiere `install_requires` für neue Dependencies:
```python
install_requires=[
    "gymnasium>=0.29.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "scipy>=1.10.0",
    "your-new-package>=1.0.0",
],
```

## Continuous Integration (zukünftig)

### GitHub Actions Beispiel
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

## Häufige Entwickler-Aufgaben

### Environment zurücksetzen zu Defaults
```python
env = DroneEnv()  # Nutzt alle Default-Werte
```

### Deterministische Episoden
```python
env.reset(seed=42)  # Gleiche Startbedingungen
```

### Custom Zielpunkt setzen
```python
env.reset()
env.target_position = np.array([10.0, 5.0, 0.0])
```

### Physik-Parameter zur Laufzeit ändern
```python
env = DroneEnv()
env.mass = 1.5  # Schwerere Drohne
env.thrust_coeff = 12.0  # Stärkere Motoren
```

## Kontakt & Contribution

Bei Fragen oder Vorschlägen:
- Issues erstellen im GitHub Repository
- Pull Requests willkommen!

## Siehe auch

- [README.md](../README.md) - Haupt-Dokumentation
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problemlösungen

