# Rendering Debugging Guide

## Problem
- Zwei Plots werden erstellt statt einem
- Beide Plots sind leer
- Rendering funktioniert nicht pro Zeitschritt

## Diagnostik

### Schritt 1: Debug-Skript ausführen
```bash
python debug_render.py
```

**Was zu erwarten ist:**
- `plt.get_fignums()` sollte nach dem ersten `render()` genau `[1]` sein
- Bei jedem weiteren `render()` sollte es `[1]` bleiben (NICHT `[1, 2]`!)
- `env.fig.number` sollte konstant `1` bleiben

**Wenn mehrere Figures erstellt werden:**
- Das ist das Problem! Matplotlib erstellt neue Fenster
- Mögliche Ursache: `plt.ion()` Verhalten oder subplot-Aufrufe

### Schritt 2: Minimaler Test
```bash
python test_minimal_render.py
```

**Was zu sehen sein sollte:**
- EIN Fenster öffnet sich
- Drohne (blauer Kreis) ist sichtbar
- Ziel (grüner Kreis) ist sichtbar  
- Die Werte in der Info-Box aktualisieren sich
- Das Fenster bleibt offen und zeigt 20 Updates

### Schritt 3: Vollständiger Test
```bash
python test_rendering.py
```

## Fixes die bereits gemacht wurden

### Fix 1: Import-Pfad korrigiert
**Vorher:**
```python
sys.path.insert(0, 'src')
from src.env import DroneEnv  # FALSCH - doppelter Pfad!
```

**Nachher:**
```python
sys.path.insert(0, 'src')
from env import DroneEnv  # RICHTIG
```

### Fix 2: Matplotlib Ion-Modus
```python
if self.fig is None:
    if self.render_mode == "human":
        plt.ion()  # VOR dem subplot!
    self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
```

### Fix 3: Rendering mit plt.draw()
```python
if self.render_mode == "human":
    plt.draw()  # Statt fig.canvas.draw()
    plt.pause(0.01)  # Ausreichende Pause
```

## Mögliche weitere Probleme

### Problem A: Backend
Manche matplotlib Backends funktionieren nicht gut mit `ion()`.

**Lösung:** Backend explizit setzen in `src/env.py` GANZ OBEN:
```python
import matplotlib
matplotlib.use('TkAgg')  # Oder 'Qt5Agg' 
import matplotlib.pyplot as plt
```

### Problem B: Figure wird nicht angezeigt
**Lösung:** Nach dem ersten subplot explizit anzeigen:
```python
if self.fig is None:
    if self.render_mode == "human":
        plt.ion()
    self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
    self.fig.set_facecolor('white')
    plt.show(block=False)  # HINZUFÜGEN
```

### Problem C: Leere Plots
Wenn Plots erstellt werden aber leer bleiben:
- Drohne könnte außerhalb des sichtbaren Bereichs sein
- Check: `info['position']` sollte nahe [0, 0, 0] sein
- Check: `info['target_position']` sollte im Bereich [-30, 30] sein

## Temporärer Workaround

Falls nichts hilft, verwende nicht-interaktives Rendering und zeige manuell:

```python
# In test_rendering.py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ...nach env.step():
env.render()
if step == 0:
    plt.show(block=False)
```

## Nächste Schritte

1. **Führe `debug_render.py` aus** - kopiere die Ausgabe
2. **Überprüfe ob Objekte sichtbar sind:**
   - Print statements zeigen Position/Ziel
   - Sind diese im Bereich -30 bis 30?
3. **Backend testen:**
   - Füge `matplotlib.use('TkAgg')` ganz oben in env.py hinzu
   - Teste nochmal

## Quick Fix zum Testen

Editiere `src/env.py`, Zeile 1-6:
```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import matplotlib
matplotlib.use('TkAgg')  # <-- DIESE ZEILE HINZUFÜGEN
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
```

Dann teste nochmal mit:
```bash
python test_minimal_render.py
```

