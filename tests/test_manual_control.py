"""
Manueller Test der Drohnen-Steuerung.

Steuerung:
- Tasten 1-4: Motor 1-4 auf 100% setzen/toggeln
- Tasten Q/W/E/R: Motor 1-4 auf 50% setzen/toggeln
- Taste 0: Alle Motoren aus
- Taste SPACE: Alle Motoren auf 25% (Hover-Versuch)
- Taste ESC oder X: Beenden
- Taste R: Reset

Die Visualisierung zeigt die Drohne von oben.
Motor-Konfiguration (X-Formation):
  Motor 2 (vorne-links, CCW)
         ○
        / \
       /   \
Motor 0 ○   ○ Motor 3
(vorne-  \   / (hinten-
rechts,   \ /  rechts,
CW)        ○   CW)
    Motor 1 (hinten-links, CW)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.drone_env import DroneEnv
import matplotlib.pyplot as plt
import time


class ManualDroneController:
    """Interaktiver Controller für manuelle Drohnen-Steuerung."""

    def __init__(self):
        # dt=0.1 entspricht 10 FPS (1/0.01 = 100 Hz)
        self.dt = 0.1  # Zeitschritt in Sekunden

        self.env = DroneEnv(
            max_steps=10000,
            render_mode="human",
            enable_crash_detection=False,  # Kein Crash während Test
            dt=self.dt,
        )

        # Motor-States
        self.motor_powers = np.array([0.0, 0.0, 0.0, 0.0])
        self.motor_active = np.array([False, False, False, False])

        # Simulation State
        self.running = True
        self.paused = False

        # FPS-Kontrolle
        self.target_fps = 1.0 / self.dt  # Ziel-FPS basierend auf dt
        self.target_frame_time = self.dt  # Zeit pro Frame in Sekunden
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps_display_time = time.time()
        self.actual_fps = 0.0

        # Reset Environment
        self.obs, self.info = self.env.reset()

        print("=" * 60)
        print("MANUELLE DROHNEN-STEUERUNG")
        print("=" * 60)
        print(f"\nSIMULATIONS-PARAMETER:")
        print(f"  Zeitschritt (dt): {self.dt}s")
        print(f"  Ziel-FPS: {self.target_fps:.0f}")
        print(f"  Frame-Zeit: {self.target_frame_time*1000:.1f}ms")
        print("\nSTEUERUNG:")
        print("  1-4     : Motor 1-4 auf 100% setzen/toggeln")
        print("  Q/W/E/R : Motor 1-4 auf 50% setzen/toggeln")
        print("  0       : Alle Motoren AUS")
        print("  SPACE   : Alle Motoren auf 25% (Hover)")
        print("  X oder ESC : Beenden")
        print("  R       : Reset (Position & Orientierung)")
        print("\nMOTOR-KONFIGURATION (X-Formation):")
        print("     2(vL)")
        print("      ○")
        print("     / \\")
        print("  0 ○   ○ 3")
        print("   (vR)  (hR)")
        print("     \\ /")
        print("      ○")
        print("     1(hL)")
        print("\nvL=vorne-links(CCW), vR=vorne-rechts(CW)")
        print("hL=hinten-links(CW), hR=hinten-rechts(CW)")
        print("=" * 60)
        print("\nGravitation ist DEAKTIVIERT!")
        print("Drücke Tasten im Matplotlib-Fenster!\n")

    def on_key_press(self, event):
        """Behandle Tastatur-Eingaben."""
        key = event.key

        # Motor 1-4 auf 100%
        motor_power = 1
        percent = f"{int(motor_power*100)}%"
        if key == '1':
            self.motor_active[0] = not self.motor_active[0]
            self.motor_powers[0] = motor_power if self.motor_active[0] else 0.0
            print(f"Motor 0 (vorne-rechts): {'EIN (' + percent + f')' if self.motor_active[0] else 'AUS'}")

        elif key == '2':
            self.motor_active[1] = not self.motor_active[1]
            self.motor_powers[1] = motor_power if self.motor_active[1] else 0.0
            print(f"Motor 1 (hinten-links): {'EIN (' + percent + f')' if self.motor_active[1] else 'AUS'}")

        elif key == '3':
            self.motor_active[2] = not self.motor_active[2]
            self.motor_powers[2] = motor_power if self.motor_active[2] else 0.0
            print(f"Motor 2 (vorne-links):  {'EIN (' + percent + f')' if self.motor_active[2] else 'AUS'}")

        elif key == '4':
            self.motor_active[3] = not self.motor_active[3]
            self.motor_powers[3] = motor_power if self.motor_active[3] else 0.0
            print(f"Motor 3 (hinten-rechts): {'EIN (' + percent + f')' if self.motor_active[3] else 'AUS'}")

        # Motor 1-4 auf 50%
        elif key == 'q':
            self.motor_active[0] = not self.motor_active[0]
            self.motor_powers[0] = 0.5 if self.motor_active[0] else 0.0
            print(f"Motor 0 (vorne-rechts): {'EIN (50%)' if self.motor_active[0] else 'AUS'}")

        elif key == 'w':
            self.motor_active[1] = not self.motor_active[1]
            self.motor_powers[1] = 0.5 if self.motor_active[1] else 0.0
            print(f"Motor 1 (hinten-links): {'EIN (50%)' if self.motor_active[1] else 'AUS'}")

        elif key == 'e':
            self.motor_active[2] = not self.motor_active[2]
            self.motor_powers[2] = 0.5 if self.motor_active[2] else 0.0
            print(f"Motor 2 (vorne-links):  {'EIN (50%)' if self.motor_active[2] else 'AUS'}")

        elif key == 'r':
            # Spezialfall: R kann entweder Motor 4 auf 50% oder Reset sein
            # Verwende Shift+R für Reset
            if event.key == 'R':  # Shift+R
                self.reset()
            else:  # Kleinbuchstabe r
                self.motor_active[3] = not self.motor_active[3]
                self.motor_powers[3] = 0.5 if self.motor_active[3] else 0.0
                print(f"Motor 3 (hinten-rechts): {'EIN (50%)' if self.motor_active[3] else 'AUS'}")

        # Alle Motoren aus
        elif key == '0':
            self.motor_powers[:] = 0.0
            self.motor_active[:] = False
            print("Alle Motoren AUS")

        # Hover (alle auf 25%)
        elif key == ' ':
            self.motor_powers[:] = 0.25
            self.motor_active[:] = True
            print("Alle Motoren auf 25% (Hover-Versuch)")

        # Reset
        elif key == 'R':  # Shift+R
            self.reset()

        # Beenden
        elif key in ['x', 'X', 'escape']:
            print("\nBeende...")
            self.running = False
            plt.close('all')

    def reset(self):
        """Reset der Drohne."""
        self.obs, self.info = self.env.reset()
        self.motor_powers[:] = 0.0
        self.motor_active[:] = False
        print("\n>>> RESET: Position und Orientierung zurückgesetzt <<<\n")

    def run(self):
        """Haupt-Loop."""
        # Verbinde Keyboard-Event
        if self.env.fig is not None:
            self.env.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Initial render um Figure zu erstellen
        self.env.render()

        # Verbinde nach erstem Render
        if self.env.fig is not None:
            self.env.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        step = 0
        last_info_time = time.time()
        self.last_frame_time = time.time()

        try:
            while self.running:
                frame_start_time = time.time()

                # Step mit aktuellen Motor-Powers
                self.obs, reward, terminated, truncated, self.info = self.env.step(self.motor_powers)

                # Rendering (Environment hat eigene FPS-Kontrolle in render())
                self.env.render()

                # Berechne wie lange wir warten müssen für korrekte FPS
                frame_elapsed = time.time() - frame_start_time
                time_to_wait = self.target_frame_time - frame_elapsed

                if time_to_wait > 0:
                    time.sleep(time_to_wait)

                # FPS-Tracking
                self.frame_count += 1
                current_time = time.time()

                # Berechne tatsächliche FPS alle Sekunde
                if current_time - self.fps_display_time >= 1.0:
                    self.actual_fps = self.frame_count / (current_time - self.fps_display_time)
                    self.frame_count = 0
                    self.fps_display_time = current_time

                # Info alle 2 Sekunden
                if current_time - last_info_time > 2.0:
                    pos = self.info['position']
                    roll_deg = np.rad2deg(self.env.orientation[0])
                    pitch_deg = np.rad2deg(self.env.orientation[1])
                    yaw_deg = np.rad2deg(self.env.orientation[2])
                    angular_vel = np.rad2deg(self.env.angular_velocity)
                    vel = self.env.velocity

                    print(self.obs)

                    print(f"\nStatus (Step {step}):")
                    print(f"  Position: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}]")
                    print(f"  Roll:  {roll_deg:6.1f}°")
                    print(f"  Pitch: {pitch_deg:6.1f}°")
                    print(f"  Yaw:   {yaw_deg:6.1f}°")
                    print(f"  Velocity: [{vel[0]:6.2f}, {vel[1]:6.2f}, {vel[2]:6.2f}]m/s")
                    print(f"  Angular Velocity: [{angular_vel[0]:6.2f}, {angular_vel[1]:6.2f}, {angular_vel[2]:6.2f}]deg/s")
                    print(f"  Motors: [{self.motor_powers[0]:.2f}, {self.motor_powers[1]:.2f}, "
                          f"{self.motor_powers[2]:.2f}, {self.motor_powers[3]:.2f}]")
                    print(f"  FPS: {self.actual_fps:.1f} / {self.target_fps:.0f} (Ziel)")

                    last_info_time = current_time

                step += 1


        except KeyboardInterrupt:
            print("\nUnterbrochen durch Benutzer")

        finally:
            self.env.close()
            print("\nManuelle Steuerung beendet.")


if __name__ == "__main__":
    controller = ManualDroneController()
    controller.run()
