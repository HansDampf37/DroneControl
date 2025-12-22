# Visualisierungs-Update

## Neue Features in der 2D-Visualisierung

### Rotor-Darstellung mit vollständiger 3D-Rotation
Die 4 Rotoren der Drohne (X-Konfiguration) werden jetzt korrekt transformiert:

- **4 farbige Kreise**: Rotor-Positionen (projiziert auf XY-Ebene)
  - 🔴 **Rot**: CW-drehende Rotoren (Motor 0, 1)
  - 🟢 **Grün**: CCW-drehende Rotoren (Motor 2, 3)
- **Graue Linien**: Verbindungen vom Zentrum zu den Rotoren
- **3D-Transformation**: Rotor-Positionen werden mit **Roll, Pitch UND Yaw** transformiert
- **XY-Projektion**: Die transformierten 3D-Positionen werden auf die XY-Ebene projiziert

### Wie es funktioniert

**Body-Frame → World-Frame → XY-Projektion:**

1. **Body-Frame**: Rotoren bei festen Positionen (X-Konfiguration, ±45°, Arm-Länge 0.25m)
2. **Rotation**: Vollständige 3D-Rotation mit Roll, Pitch, Yaw
3. **Projektion**: XY-Komponenten der rotierten Positionen werden gezeichnet

```python
# Für jeden Rotor:
rotor_pos_world = R @ rotor_pos_body  # 3D-Rotation
rotor_xy = [rotor_pos_world[0], rotor_pos_world[1]]  # XY-Projektion
```

### Sichtbare Effekte

#### Nur Yaw (Drehung um Z-Achse)
```
Yaw = 0°:           Yaw = 45°:
    ○                   ○
    |                  / \
 ○--●--○            ○--●--○
    |                  \ /
    ○                   ○
```
Das X dreht sich, bleibt aber symmetrisch.

#### Roll (Neigung zur Seite)
```
Roll = 0°:          Roll > 0°:
    ○                   ○
    |                   |
 ○--●--○             ○-●  ○
    |                   |
    ○                   ○
```
Rechte Arme erscheinen kürzer (vom Betrachter weg geneigt).

#### Pitch (Neigung vorwärts/rückwärts)
```
Pitch = 0°:         Pitch < 0°:
    ○                   ○
    |                   |
 ○--●--○             ○--●--○
    |                   ○
    ○
```
Vordere Arme erscheinen kürzer (nach vorne geneigt).

#### Kombiniert (Roll + Pitch + Yaw)
Die X-Form wird asymmetrisch - verschiedene Arm-Längen zeigen die 3D-Neigung!

### Neigungs-Indikator
Ein **oranger Pfeil** zeigt die Neigungsrichtung der Drohne:

- **Berechnung**: Projektion der Drohnen-Normalen auf die XY-Ebene
- **Bedeutung**: Zeigt in welche Richtung die Drohne "kippt"
- **Sichtbarkeit**: Nur bei nennenswerter Neigung (>0.01 rad)

### Erweiterte Info-Box
Die Info-Box zeigt jetzt auch:
- **Roll**: Drehung um X-Achse (in Grad)
- **Pitch**: Drehung um Y-Achse (in Grad)  
- **Yaw**: Drehung um Z-Achse (in Grad)

## Visualisierungs-Elemente

```
     Rotor 2 (Grün, CCW)
           ○
           |
    Motor 0 ○----●----○ Motor 3
   (Rot,CW)      |      (Rot,CW)
                 |
                 ○
           Rotor 1 (Grün, CCW)

    ● = Drohnen-Zentrum (blau)
    ○ = Rotor (rot/grün)
    → = Neigungspfeil (orange)
```

## Interpretation

### Keine Neigung (Hover)
- Kein oranger Pfeil sichtbar
- Alle Rotoren gleichmäßig vom Zentrum entfernt
- Roll ≈ 0°, Pitch ≈ 0°

### Roll nach rechts
- Oranger Pfeil zeigt nach rechts
- Roll > 0°
- Linke Rotoren (2, 1) höher, rechte Rotoren (0, 3) niedriger

### Pitch vorwärts
- Oranger Pfeil zeigt vorwärts (in Flugrichtung)
- Pitch < 0°
- Hintere Rotoren (1, 3) höher, vordere Rotoren (0, 2) niedriger

### Kombinierte Neigung
- Oranger Pfeil zeigt in diagonale Richtung
- Roll ≠ 0°, Pitch ≠ 0°

## Test

```bash
python tests/test_visualization.py
```

Dieser Test zeigt verschiedene Manöver:
1. Hover (keine Neigung)
2. Roll rechts
3. Pitch vorwärts
4. Kombiniert
5. Zurück zu Hover

## Technische Details

### Rotor-Positionen (X-Konfiguration)
```python
# Body-Frame Winkel (vor Yaw-Rotation)
Motor 0: +45°  (vorne-rechts, CW)
Motor 1: -135° (hinten-links, CW)
Motor 2: +135° (vorne-links, CCW)
Motor 3: -45°  (hinten-rechts, CCW)
```

### Neigungsberechnung
```python
# Normale im Body-Frame
normal_body = [0, 0, 1]

# Rotation ins World-Frame
R = get_rotation_matrix(roll, pitch, yaw)
normal_world = R @ normal_body

# Projektion auf XY
tilt_x = normal_world[0]
tilt_y = normal_world[1]
```

### Farb-Schema
- **Drohnen-Zentrum**: Blau (#0066cc)
- **CW-Rotoren**: Rot (#ff6666)
- **CCW-Rotoren**: Grün (#66ff66)
- **Rotor-Arme**: Grau (#666666)
- **Neigungspfeil**: Orange (#ff9900)
- **Ziel**: Grün (#00cc00)
- **Wind**: Rot (#cc0000)

## Zukünftige Erweiterungen

Mögliche weitere Visualisierungs-Features:
- [ ] Motor-Thrust als Kreis-Größe oder Farb-Intensität
- [ ] Trajektorie (Pfad der letzten N Positionen)
- [ ] 3D-Ansicht mit echten Rotor-Höhen
- [ ] Wind-Effekt als Partikel
- [ ] Geschwindigkeits-Vektor

