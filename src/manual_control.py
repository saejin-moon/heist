# src/manual_control.py
# Manual (keyboard) control of HEIST for debugging and demos.
#
# Controls:
#   1/2/3/4  : select Scout / Hacker / Muscle / Extractor
#   W/A/S/D  : move up / left / down / right
#   E        : role-specific action (tag / hack / neutralize / extract)
#   SPACE    : wait
#
# The terminal prints the active agent's observation matrix, action mask,
# and reward delta on every step.  The pygame window shows the fog of war,
# cameras, doors, guards, and the global alarm meter.

import pygame as pg

from env import HeistEnv
from constants import TILE_SIZE, WAIT, INTERACT


def run_manual(env):
    env.reset()
    active_agent = "scout"

    pg.init()
    screen = pg.display.set_mode((env.map_w * TILE_SIZE, env.map_h * TILE_SIZE + 40))
    pg.display.set_caption("heist")
    font = pg.font.SysFont("monospace", 16)
    clock = pg.time.Clock()

    agent_select = {pg.K_1: "scout", pg.K_2: "hacker", pg.K_3: "muscle", pg.K_4: "extractor"}
    pg_key_map = {
        pg.K_w: 0, pg.K_s: 1, pg.K_a: 2, pg.K_d: 3,
        pg.K_SPACE: WAIT, pg.K_e: INTERACT,
    }

    running = True
    while running:
        action = WAIT
        step_env = False

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key in agent_select:
                    active_agent = agent_select[event.key]
                elif event.key in pg_key_map:
                    action = pg_key_map[event.key]
                    step_env = True

        if step_env:
            actions = {agent: WAIT for agent in env.agents}
            actions[active_agent] = action
            obs, rewards, terms, truncs, infos = env.step(actions)

            print(f"[ {active_agent.upper()} ] Reward: {rewards.get(active_agent, 0):.2f} "
                  f"| Mask: {obs[active_agent]['action_mask']}")
            print(f"Observation: {obs[active_agent]['observation']}")

            if any(terms.values()) or any(truncs.values()):
                print("Episode Ended.")
                env.reset()

        env.render_pygame(screen, active_agent, font)
        pg.display.flip()
        clock.tick(15)

    pg.quit()


if __name__ == "__main__":
    run_manual(HeistEnv())
