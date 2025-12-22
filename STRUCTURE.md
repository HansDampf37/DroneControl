# Projektstruktur

```
drone-control/
│
├── src/                          # Produktivcode
│   ├── __init__.py
│   └── drone_env/               # Haupt-Package
│       ├── __init__.py          # Exportiert DroneEnv
│       └── env.py               # DroneEnv Klasse (Gymnasium Interface)
│
├── tests/                       # Tests und Debugging
│   ├── __init__.py
│   ├── test_env.py              # Umfassende Environment-Tests
│   ├── test_rendering.py        # Rendering-Tests (200 Steps)
│   ├── test_minimal_render.py   # Minimaler Rendering-Test (20 Steps)
│   └── debug_render.py          # Debug-Informationen (Figure-Tracking)
│
├── examples/                    # Beispiel-Skripte
│   ├── __init__.py
│   ├── random_agent.py          # Random/Hover Agent Demo
│   └── training.py              # Training mit Stable-Baselines3
│
├── setup.py                     # Package-Installation
├── requirements.txt             # Python-Dependencies
├── README.md                    # Haupt-Dokumentation
├── .gitignore                   # Git Ignore File
│
└── docs/                        # Zusätzliche Dokumentation
    ├── RENDERING_FIX.md
    └── RENDERING_DEBUG.md

```

## Verzeichnisse im Detail

### `src/drone_env/`
**Produktivcode - sollte sauber und getestet sein**

- **`env.py`**: Hauptklasse `DroneEnv`
  - Gymnasium Interface (reset, step, render, close)
  - Physik-Simulation (Kräfte, Drehmomente, Integration)
  - Wind-Dynamik (Ornstein-Uhlenbeck-Prozess)
  - Reward-Berechnung
  - 2D-Visualisierung

- **`__init__.py`**: Package-Initialisierung
  - Exportiert `DroneEnv` für einfache Imports
  - Version-Information

### `tests/`
**Test- und Debug-Skripte**

- **`test_env.py`**: Umfassende Tests
  - Test 1: Grundlegende Funktionalität
  - Test 2: Physik-Simulation
  - Test 3: Wind-Dynamik
  - Test 4: Reward-Funktion
  - Demo mit Visualisierung

- **`test_rendering.py`**: Rendering-Test
  - 200 Steps mit wechselnden Actions
  - Überprüft Visualisierung

- **`test_minimal_render.py`**: Schneller Test
  - Nur 20 Steps
  - Für schnelle Validierung

- **`debug_render.py`**: Debug-Informationen
  - Tracked Figure-Nummern
  - Überprüft matplotlib State

### `examples/`
**Beispiele für Nutzer**

- **`random_agent.py`**: Demo-Agents
  - Random Agent
  - Hover Agent (Baseline)
  - CLI mit Argumenten
  - Statistiken über Episoden

- **`training.py`**: RL-Training
  - Integration mit Stable-Baselines3
  - Unterstützt PPO, SAC, TD3
  - Training mit Callbacks
  - Model Evaluation

## Installation

### Entwickler-Modus (empfohlen)
```bash
pip install -e .
```

Dies installiert das Package im "editable mode", sodass Änderungen am Code sofort wirksam werden.

### Standard-Installation
```bash
pip install .
```

### Mit RL-Support
```bash
pip install -e ".[rl]"
```

## Verwendung

### Import im Code
```python
# Nach Installation
from src.drone_env import DroneEnv

# Oder direkt (wenn nicht installiert)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.drone_env import DroneEnv
```

### Tests ausführen
```bash
# Aus dem Root-Verzeichnis
python tests/test_env.py
python tests/test_rendering.py
python tests/test_minimal_render.py
python tests/debug_render.py
```

### Beispiele ausführen
```bash
# Random Agent
python examples/random_agent.py --episodes 5

# Mit Rendering
python examples/random_agent.py --episodes 3 --render

# Training
python examples/training.py --mode train --algorithm PPO --timesteps 100000
```

## Design-Prinzipien

### 1. Trennung von Produktiv- und Test-Code
- **`src/`**: Nur produktiver Code, keine Tests
- **`tests/`**: Alle Tests und Debug-Skripte
- **`examples/`**: Beispiele für Nutzer

### 2. Package-Struktur
- Klare Package-Hierarchie mit `__init__.py`
- Imports über Package-Namen
- Installation via `setup.py`

### 3. Entwickler-Freundlichkeit
- Editable Installation (`pip install -e .`)
- Konsistente Import-Pfade
- Klare Verzeichnisstruktur

## Nächste Schritte

1. **Package installieren**:
   ```bash
   pip install -e .
   ```

2. **Tests ausführen**:
   ```bash
   python tests/test_env.py
   ```

3. **Beispiele testen**:
   ```bash
   python examples/random_agent.py --episodes 1 --render
   ```

4. **Entwickeln**:
   - Code in `src/drone_env/` bearbeiten
   - Tests in `tests/` hinzufügen
   - Beispiele in `examples/` erweitern

