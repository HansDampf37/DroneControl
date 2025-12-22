# Troubleshooting Guide

Lösungen für häufige Probleme beim Drohnen-RL Environment.

## Rendering-Probleme

### Problem: Leere oder weiße Fenster

**Symptome:**
- Matplotlib-Fenster öffnet sich, ist aber leer/weiß
- Keine Drohne oder Ziel sichtbar
- Nur gelegentlich erscheint etwas

**Lösung 1: Backend explizit setzen**

Editiere `src/drone_env/env.py`, Zeile 5-6:
```python
import matplotlib
matplotlib.use('TkAgg')  # Explizit TkAgg verwenden
import matplotlib.pyplot as plt
```

Alternative Backends:
- `'TkAgg'` (Standard, meist vorinstalliert)
- `'Qt5Agg'` (benötigt `pip install PyQt5`)
- `'QtAgg'` (neuere Qt-Version)

**Lösung 2: Backend-Dependencies installieren**

```bash
# Für TkAgg (Linux)
sudo apt-get install python3-tk

# Für Qt5Agg
pip install PyQt5
```

**Lösung 3: Test mit Debug-Skript**

```bash
# Zeigt matplotlib-Informationen
python tests/debug_render.py
```

Erwartete Ausgabe:
```
plt.get_fignums(): [1]  # Nur EINE Figure!
env.fig.number: 1       # Bleibt konstant
```

Wenn mehrere Figures erstellt werden → Backend-Problem!

### Problem: Mehrere Fenster werden geöffnet

**Symptome:**
- Bei jedem `render()` öffnet sich ein neues Fenster
- `plt.get_fignums()` zeigt `[1, 2, 3, ...]`

**Ursache:**
- Matplotlib erstellt neue Figures statt die bestehende zu aktualisieren

**Lösung:**

Überprüfe in `src/drone_env/env.py` Zeile ~330:
```python
if self.fig is None:
    if self.render_mode == "human":
        plt.ion()  # MUSS vor subplot sein!
    self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
```

Stelle sicher, dass `self.fig` wiederverwendet wird.

### Problem: Rendering zu langsam

**Symptome:**
- Animation ruckelt
- Große Verzögerung zwischen Steps

**Lösungen:**

1. **Rendering-Frequenz reduzieren:**
   ```python
   for step in range(1000):
       obs, reward, term, trunc, info = env.step(action)
       if step % 5 == 0:  # Nur jeden 5. Step rendern
           env.render()
   ```

2. **Pause-Zeit reduzieren** (in `env.py` Zeile ~455):
   ```python
   plt.pause(0.001)  # Statt 0.01
   ```

3. **Figure-Größe reduzieren** (Zeile ~335):
   ```python
   self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))  # Statt 10x10
   ```

### Problem: "RuntimeError: main thread is not in main loop"

**Ursache:**
- Matplotlib/Tkinter Threading-Probleme

**Lösung:**
```python
# Vor import matplotlib
import matplotlib
matplotlib.use('Agg')  # Non-interactive Backend
```

Oder verwende `render_mode=None` für Training (kein Rendering).

## Import-Probleme

### Problem: "ModuleNotFoundError: No module named 'src'"

**Ursache:**
- Falsches Working Directory
- Package nicht installiert

**Lösungen:**

1. **Richtiges Directory:**
   ```bash
   cd /pfad/zum/drone-control
   python tests/test_env.py
   ```

2. **Package installieren:**
   ```bash
   pip install -e .
   ```

3. **Manueller Path (in Skript):**
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   from src.drone_env import DroneEnv
   ```

### Problem: "ImportError: cannot import name 'DroneEnv'"

**Ursache:**
- `__init__.py` fehlt oder ist fehlerhaft

**Lösung:**

Überprüfe `src/drone_env/__init__.py`:
```python
from .env import DroneEnv
__all__ = ['DroneEnv']
```

## Physik-Probleme

### Problem: Drohne fällt sofort zu Boden

**Ursache:**
- Hover-Thrust zu niedrig
- Gravitation zu stark

**Debug:**
```python
env = DroneEnv()
obs, info = env.reset()

# Teste verschiedene Thrust-Werte
for thrust in [0.2, 0.25, 0.3]:
    print(f"\nTesting thrust: {thrust}")
    for _ in range(100):
        action = np.array([thrust] * 4)
        obs, reward, term, trunc, info = env.step(action)
    print(f"Final Z-Position: {info['position'][2]:.2f}m")
```

**Lösung:**

Editiere `src/drone_env/env.py`, Zeile ~50:
```python
self.thrust_coeff = 12.0  # Erhöhen für mehr Thrust
```

Oder passe Masse/Gravitation an:
```python
self.mass = 0.8  # Leichter
self.gravity = 9.81  # Standard
```

### Problem: Drohne rotiert unkontrolliert

**Ursache:**
- Drehmoment zu stark
- Trägheitsmomente zu niedrig

**Lösung:**

Zeile ~51 in `env.py`:
```python
self.inertia = np.array([0.02, 0.02, 0.04])  # Erhöhen
```

Oder Torque-Koeffizient reduzieren:
```python
self.torque_coeff = 0.05  # Statt 0.1
```

### Problem: Wind zu stark/schwach

**Lösung:**

Beim Environment-Init:
```python
# Schwacher Wind
env = DroneEnv(wind_strength_range=(0.0, 2.0))

# Kein Wind
env = DroneEnv(wind_strength_range=(0.0, 0.0))

# Starker Wind
env = DroneEnv(wind_strength_range=(2.0, 10.0))
```

## Training-Probleme

### Problem: Agent lernt nicht

**Mögliche Ursachen & Lösungen:**

1. **Reward zu sparse:**
   ```python
   # In env.py, _compute_reward()
   # Nutze Dense Reward (bereits default)
   reward = 1.0 / (1.0 + distance)
   ```

2. **Episode zu kurz:**
   ```python
   env = DroneEnv(max_steps=2000)  # Statt 1000
   ```

3. **Hyperparameter:**
   ```python
   # Für PPO (in examples/training.py)
   model = PPO(
       'MlpPolicy',
       env,
       learning_rate=3e-4,  # Reduzieren bei Instabilität
       n_steps=2048,        # Erhöhen für bessere Exploration
   )
   ```

4. **Observation Normalisierung:**
   ```python
   from stable_baselines3.common.vec_env import VecNormalize
   
   env = DroneEnv()
   env = VecNormalize(env, norm_obs=True, norm_reward=True)
   ```

### Problem: Training zu langsam

**Lösungen:**

1. **Kein Rendering:**
   ```python
   env = DroneEnv(render_mode=None)
   ```

2. **Timestep erhöhen:**
   ```python
   env = DroneEnv(dt=0.02)  # Statt 0.01, weniger Steps
   ```

3. **Vectorized Environments:**
   ```python
   from stable_baselines3.common.vec_env import SubprocVecEnv
   
   def make_env():
       return DroneEnv(render_mode=None)
   
   env = SubprocVecEnv([make_env for _ in range(4)])
   ```

## Allgemeine Probleme

### Problem: "UserWarning: overflow encountered"

**Ursache:**
- Numerische Instabilität in Physik-Berechnung

**Lösung:**

Begrenze Werte in `env.py`:
```python
# In _update_physics(), nach Integration
self.velocity = np.clip(self.velocity, -20.0, 20.0)
self.angular_velocity = np.clip(self.angular_velocity, -10.0, 10.0)
```

### Problem: Tests schlagen fehl

**Debug-Schritte:**

1. **Einzelnen Test ausführen:**
   ```bash
   python tests/test_minimal_render.py
   ```

2. **Fehler-Output lesen:**
   ```bash
   python tests/test_env.py 2>&1 | less
   ```

3. **Dependencies überprüfen:**
   ```bash
   pip list | grep -E 'gymnasium|numpy|matplotlib'
   ```

4. **Python-Version:**
   ```bash
   python --version  # Sollte ≥3.8 sein
   ```

## Quick Debug-Checks

### Minimaler Funktionstest
```python
from src.drone_env import DroneEnv
import numpy as np

env = DroneEnv(render_mode=None)
obs, info = env.reset()
print(f"✓ Reset OK, obs shape: {obs.shape}")

action = np.array([0.25, 0.25, 0.25, 0.25])
obs, reward, term, trunc, info = env.step(action)
print(f"✓ Step OK, reward: {reward:.4f}")

env.close()
print("✓ Environment funktioniert!")
```

### Rendering-Test
```bash
# Einfachster Test
python tests/test_minimal_render.py

# Wenn das nicht funktioniert:
python tests/debug_render.py
```

### Physik-Test
```python
env = DroneEnv()
obs, info = env.reset()

# Hover-Test (sollte ~bei Z=0 bleiben)
for _ in range(100):
    env.step(np.array([0.25] * 4))

print(f"Z-Position nach Hover: {env.position[2]:.2f}m")
# Erwartung: -0.5 bis +0.5
```

## Noch Probleme?

### Logs sammeln
```bash
python tests/test_env.py > test_output.txt 2>&1
python tests/debug_render.py > debug_output.txt 2>&1
```

### System-Info
```bash
python --version
pip list | grep -E 'gymnasium|numpy|matplotlib|scipy'
uname -a  # Linux
```

### Minimales Reproduktions-Beispiel
```python
# minimal_bug.py
from src.drone_env import DroneEnv

env = DroneEnv(render_mode="human")
obs, info = env.reset()
env.render()
input("Press Enter...")
env.close()
```

## Siehe auch

- [README.md](../README.md) - Haupt-Dokumentation
- [DEVELOPMENT.md](DEVELOPMENT.md) - Entwickler-Guide

