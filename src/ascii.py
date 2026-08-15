import select
import sys
import termios
import time
import tty

# A basic terminal ASCII playback tool for HEIST diagnostic evaluation.
# Reads state/observation arrays and renders them with configurable speed and pause toggle.


class ASCIIRenderer:
    def __init__(self, fps=5.0):
        self.fps = fps
        self.paused = False

        # Mapping constants to ASCII chars
        self.char_map = {
            -1: "░",  # FOG
            0: " ",  # EMPTY
            1: "█",  # WALL
            2: "T",  # TERMINAL
            3: "$",  # LOOT
            4: "E",  # EXTRACT
            5: "G",  # GUARD
            6: "A",  # ALLY (generic)
            7: "C",  # CAMERA
            8: "D",  # DOOR
            9: "x",  # WAYPOINT
        }

    def _get_key(self):
        """Non-blocking key press detection."""
        dr, dw, de = select.select([sys.stdin], [], [], 0)
        if dr:
            return sys.stdin.read(1)
        return None

    def render_step(self, grid, step_idx, alarm, terminal_status, extract_status):
        """Renders the discrete grid to the terminal."""
        # Clear screen and move cursor to top left
        sys.stdout.write("\033[2J\033[H")

        header = (
            f"=== HEIST Step: {step_idx:04d} | Alarm: {alarm:03.0f}/100 ===\n"
            f"Terminal: {terminal_status} | Extraction: {extract_status}\n"
            f"[SPACE] Play/Pause | [f] Faster | [s] Slower | [q] Quit\n"
            f"FPS: {self.fps:.1f}\n\n"
        )
        sys.stdout.write(header)

        for row in grid:
            line = "".join([self.char_map.get(cell, "?") for cell in row])
            sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def run_playback(self, episode_history):
        """
        Takes a list of episode states:
        dict(grid=..., step=..., alarm=..., terminal=..., extract=...)
        """
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        idx = 0
        total_steps = len(episode_history)

        try:
            while idx < total_steps:
                state = episode_history[idx]

                self.render_step(
                    state["grid"],
                    state["step"],
                    state["alarm"],
                    state["terminal"],
                    state["extract"],
                )

                # Wait loop for frame timing & input handling
                start_time = time.time()
                frame_duration = 1.0 / self.fps

                while True:
                    key = self._get_key()
                    if key:
                        if key == " ":
                            self.paused = not self.paused
                        elif key == "f":
                            self.fps = min(60.0, self.fps + 2.0)
                        elif key == "s":
                            self.fps = max(1.0, self.fps - 2.0)
                        elif key == "q":
                            return

                    if self.paused:
                        time.sleep(0.1)
                    else:
                        elapsed = time.time() - start_time
                        if elapsed >= frame_duration:
                            break

                if not self.paused:
                    idx += 1
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            print("\nPlayback finished.")


if __name__ == "__main__":
    # Example usage:
    # from src.env import HeistEnv
    # env = HeistEnv(DEFAULT_CONFIG)
    # env.reset()
    # history = []
    # ... rollout loop collecting env.grid and state into history ...
    # ASCIIRenderer().run_playback(history)
    pass
