# Quick Reference - Drohnen-RL Environment

## Schnellstart

### Tests ausführen
```bash
# Minimaler Test (schnell)
python tests/test_minimal_render.py

# Alle Tests
python tests/test_env.py

# Rendering-Test
python tests/test_rendering.py

# Debug-Informationen
python tests/debug_render.py
```

### Beispiele
```bash
# Random Agent (5 Episoden, ohne Rendering)
python examples/random_agent.py --episodes 5

# Mit Visualisierung
python examples/random_agent.py --episodes 3 --render

# Hover Agent (Baseline)
python examples/random_agent.py --agent hover --render

# Training (benötigt: pip install stable-baselines3[extra])
python examples/training.py --mode train --algorithm PPO --timesteps 100000

# Evaluation
python examples/training.py --mode eval --model-path models/drone_model
```

## Import

### In Python-Skript
```python
import sys
from pathlib import Path

# Pfad zum Repository
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.drone_env import DroneEnv

# Environment erstellen
env = DroneEnv(max_steps=1000, render_mode="human")
obs, info = env.reset()

# Haupt-Loop
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        break

env.close()
```

## Installation (optional)

### Development Mode
```bash
pip install -e .
```

Nach Installation:
```python
from src.drone_env import DroneEnv
```

### Mit RL-Unterstützung
```bash
pip install -e ".[rl]"
```

## Datei-Übersicht

| Pfad | Beschreibung |
|------|--------------|
| `src/drone_env/env.py` | Haupt-Environment-Klasse |
| `tests/test_env.py` | Umfassende Tests |
| `tests/test_minimal_render.py` | Schneller Rendering-Test |
| `examples/random_agent.py` | Random/Hover Agent Demo |
| `examples/training.py` | SB3 Training |
| `README.md` | Vollständige Dokumentation |
| `STRUCTURE.md` | Projekt-Struktur |

## Häufige Aufgaben

### Neuen Test hinzufügen
1. Datei in `tests/` erstellen
2. Import-Template verwenden:
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   from src.drone_env import DroneEnv
   ```

### Neues Beispiel hinzufügen
1. Datei in `examples/` erstellen
2. Gleiche Import-Struktur wie Tests
3. In README.md dokumentieren

### Environment-Parameter ändern
Editiere `src/drone_env/env.py`

### Physik-Parameter anpassen
In `src/drone_env/env.py`, Zeilen ~45-60:
```python
self.mass = 1.0
self.arm_length = 0.25
self.thrust_coeff = 10.0
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"
```bash
# Stelle sicher, dass du im richtigen Verzeichnis bist
cd /pfad/zum/drone-control

# Oder installiere das Package
pip install -e .
```

### Rendering-Probleme
```bash
# Teste mit minimal test
python tests/test_minimal_render.py

# Debug-Informationen
python tests/debug_render.py
```

### matplotlib Backend-Fehler
Editiere `src/drone_env/env.py`, Zeile 5-6:
```python
import matplotlib
matplotlib.use('TkAgg')  # Oder 'Qt5Agg'
```

