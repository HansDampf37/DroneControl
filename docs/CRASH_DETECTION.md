# Crash-Detektion

Die Drohne erkennt jetzt Abstürze und beendet die Episode automatisch.

## Features

### ✅ Implementiert

**1. Low Z-Coordinate (Primär)**
- **Effizienteste Methode**: O(1) Vergleich
- **Standard-Threshold**: z < -5.0m
- **Eindeutig**: Drohne ist definitiv abgestürzt

**2. Extreme Tilt (Sekundär)**
- **Für unkontrollierte Drohne**: |Roll| > 80° ODER |Pitch| > 80°
- **Standard-Threshold**: 80° (in Radiant umgerechnet)
- **Zusätzlicher Check**: Fängt Totalverlust der Kontrolle ab

**3. Konfigurierbar**
- Crash-Detektion kann deaktiviert werden
- Alle Thresholds sind anpassbar

## Verwendung

### Standard (Crash-Detektion aktiviert)
```python
from src.drone_env import DroneEnv

env = DroneEnv(
    enable_crash_detection=True,  # Standard
    crash_z_threshold=-5.0,        # Standard
    crash_tilt_threshold=80.0,     # Standard (Grad)
)
```

### Angepasste Thresholds
```python
env = DroneEnv(
    crash_z_threshold=-10.0,       # Tiefer = mehr Toleranz
    crash_tilt_threshold=85.0,     # Höher = mehr Toleranz
)
```

### Deaktiviert (wie vorher)
```python
env = DroneEnv(
    enable_crash_detection=False
)
```

## Verhalten

### Episode endet bei Crash
```python
obs, info = env.reset()

for step in range(1000):
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        if info['crashed']:
            print("Drohne abgestürzt!")
        break
    
    if truncated:
        print("Max Steps erreicht")
        break
```

### Info-Dictionary
```python
info = {
    'distance_to_target': float,
    'position': np.ndarray,
    'target_position': np.ndarray,
    'step_count': int,
    'crashed': bool,  # ← NEU
}
```

## Crash-Kriterien

### 1. Z-Coordinate
```python
if self.position[2] < self.crash_z_threshold:
    return True  # Crash!
```

**Typische Werte:**
- z = 0.0m: Start-Höhe
- z = -2.0m: Noch OK (leichter Abstieg)
- z = -5.0m: Crash! (Standard-Threshold)
- z = -10.0m: Definitiv gecrasht

### 2. Tilt (Neigung)
```python
roll, pitch, _ = self.orientation

if abs(roll) > self.crash_tilt_threshold:
    return True  # Komplett seitlich gekippt

if abs(pitch) > self.crash_tilt_threshold:
    return True  # Komplett vorwärts/rückwärts gekippt
```

**Typische Werte:**
- 0°-30°: Normale Manöver
- 30°-60°: Aggressive Manöver
- 60°-80°: Sehr aggressiv, aber noch kontrollierbar
- >80°: Unkontrolliert, Crash! (Standard-Threshold)

## Tests

```bash
python tests/test_crash_detection.py
```

**Test-Szenarien:**
1. ✅ Crash durch zu niedrige Z-Koordinate
2. ✅ Crash durch extreme Neigung
3. ✅ Keine Detektion wenn deaktiviert
4. ✅ Kein False Positive bei normalem Hover
5. ✅ 'crashed' Flag in info vorhanden

## Vergleich der Methoden

| Methode | Effizienz | Eindeutigkeit | False Positives | Frühzeitige Detektion |
|---------|-----------|---------------|-----------------|----------------------|
| **Z-Coordinate** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Extreme Tilt** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Z-Velocity (nicht impl.) | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**Warum Z-Coordinate als Primär?**
- ✅ Einfachste Berechnung (1 Vergleich)
- ✅ Keine False Positives
- ✅ Eindeutige physikalische Grenze
- ✅ Stabil über verschiedene Szenarien

**Warum Extreme Tilt als Sekundär?**
- ✅ Erkennt Totalverlust der Kontrolle
- ✅ Früher als Z-Threshold (Drohne noch oben aber kippt um)
- ✅ Zusätzlicher Sicherheits-Check

## Auswirkungen auf Training

### Vorher (ohne Crash-Detektion)
```
Episode: 1000 Steps → truncated
→ Agent bekommt kein negatives Signal für Crash
```

### Jetzt (mit Crash-Detektion)
```
Episode: 127 Steps → terminated (crashed)
→ Episode endet früher
→ Agent lernt: Crash = schlecht
```

### Empfehlungen für Training

**Option 1: Mit Crash-Penalty (empfohlen)**
```python
def _compute_reward(self) -> float:
    distance = np.linalg.norm(self.target_position - self.position)
    reward = 1.0 / (1.0 + distance)
    
    # Großer Penalty bei Crash
    if self._check_crash():
        reward -= 10.0
    
    return float(reward)
```

**Option 2: Ohne extra Penalty**
```python
# Einfach Episode beenden
# Agent lernt durch kürzere Episode = weniger Gesamt-Reward
```

**Option 3: Deaktiviert für Exploration**
```python
# Am Anfang des Trainings
env = DroneEnv(enable_crash_detection=False)

# Später aktivieren
env = DroneEnv(enable_crash_detection=True)
```

## Beispiel: Crash-sichere Policy

```python
from src.drone_env import DroneEnv
import numpy as np

env = DroneEnv(enable_crash_detection=True)
obs, info = env.reset()

total_crashes = 0
total_episodes = 0

for episode in range(100):
    obs, info = env.reset()
    done = False
    
    while not done:
        # Deine Policy hier
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if terminated and info['crashed']:
            total_crashes += 1
            print(f"Episode {episode}: CRASHED at step {info['step_count']}")
            break
    
    total_episodes += 1

crash_rate = total_crashes / total_episodes * 100
print(f"\nCrash Rate: {crash_rate:.1f}%")
```

## Anpassungen

### Tolerantere Crash-Detektion (für Training)
```python
env = DroneEnv(
    crash_z_threshold=-10.0,   # Tiefer
    crash_tilt_threshold=85.0, # Höher
)
```

### Strengere Crash-Detektion (für Evaluation)
```python
env = DroneEnv(
    crash_z_threshold=-3.0,    # Höher
    crash_tilt_threshold=70.0, # Niedriger
)
```

### Nur Z-Coordinate (einfachste Variante)
```python
env = DroneEnv(
    crash_z_threshold=-5.0,
    crash_tilt_threshold=180.0,  # Effektiv deaktiviert
)
```

## Zukünftige Erweiterungen

Mögliche weitere Crash-Kriterien:
- [ ] Hohe Z-Velocity (vz < -15 m/s)
- [ ] Hohe Winkelgeschwindigkeit (außer Kontrolle)
- [ ] Kollision mit Hindernissen (wenn implementiert)
- [ ] Timeout ohne Bewegung (festgeklemmt)

---

**Status:** ✅ Vollständig implementiert und getestet
**Tests:** 5/5 bestanden

