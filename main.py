import pygame
import sys
import time
import random
import numpy as np
import os
from src.environment.aegis_env import AegisEnv
from src.engine.fuzzy_director import FuzzyDirector
from src.environment.sensors import SensoryLayer
from src.agents.search import TacticalSearch
from src.agents.learning import QLearningBoss

# Constants
CELL_SIZE = 64
GRID_DIM = 10
SIDE_PANEL = 320
WIDTH, HEIGHT = (GRID_DIM * CELL_SIZE) + SIDE_PANEL, GRID_DIM * CELL_SIZE
FPS = 15

# Color Palette
COLORS = {
    "Tank": (50, 120, 255),      # Blue
    "Assassin": (150, 50, 200),  # Purple
    "Wizard": (0, 255, 255),     # Cyan
    "Cleric": (255, 215, 0),     # Gold
    "Boss": (255, 0, 0),         # Red
    "Minion": (139, 0, 0),       # Dark Red
    "Wall": (60, 60, 65),
    "Lava": (255, 69, 0),
    "Fog": (10, 10, 15)
}

class AegisDM_Final:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("AEGIS-DM: Squad Tactics Simulator")
        self.font = pygame.font.SysFont("Consolas", 16)
        self.big_font = pygame.font.SysFont("Consolas", 36, bold=True)
        self.clock = pygame.time.Clock()
        
        self.env = AegisEnv(GRID_DIM)
        self.fuzzy = FuzzyDirector()
        self.sensors = SensoryLayer(GRID_DIM)
        self.search_ai = TacticalSearch(self.env.validator, self.env.semantic)
        self.boss_ai = QLearningBoss(action_space_size=4)
        
        # Load Squad Policy
        policy_path = "data/replay_buffer/boss_policy.npy"
        if os.path.exists(policy_path):
            self.boss_ai.q_table.update(np.load(policy_path, allow_pickle=True).item())
        
        self.log = []
        self.intensity = 0.0
        self.game_over = False
        self.winner_banner = ""
        self.reset_sim()

    def add_log(self, msg, flavor="SYSTEM"):
        prefix = f"[{flavor}] " if flavor != "LOG" else " > "
        self.log.append(f"{prefix}{msg}")
        if len(self.log) > 18: self.log.pop(0)

    def reset_sim(self):
        self.env.reset()
        self.game_over = False
        self.winner_banner = ""
        self.log = ["--- New Simulation Initialized ---"]
        self.add_log("DM is generating tactical layout...", "DM")

    def draw_dashboard(self):
        pygame.draw.rect(self.screen, (25, 25, 30), (GRID_DIM * CELL_SIZE, 0, SIDE_PANEL, HEIGHT))
        
        # Intensity Meter
        self.screen.blit(self.font.render(f"DM TENSION: {self.intensity:.2f}", True, (255, 255, 255)), (GRID_DIM * CELL_SIZE + 20, 20))
        pygame.draw.rect(self.screen, (50, 50, 50), (GRID_DIM * CELL_SIZE + 20, 45, 250, 15))
        meter_w = int(max(0, min(10, self.intensity)) * 25)
        pygame.draw.rect(self.screen, (255, 50, 0) if self.intensity > 7 else (0, 200, 255), (GRID_DIM * CELL_SIZE + 20, 45, meter_w, 15))

        # Squad Status
        offset = 80
        self.screen.blit(self.font.render("SQUAD STATUS:", True, (150, 150, 150)), (GRID_DIM * CELL_SIZE + 20, offset))
        for name, data in self.env.players.items():
            offset += 25
            hp_val = int(max(0, data['hp']))
            t_color = (100, 100, 100) if hp_val <= 0 else COLORS.get(name, (255,255,255))
            suffix = "[KIA]" if hp_val <= 0 else f"HP: {hp_val:2}"
            self.screen.blit(self.font.render(f"{name:8} | {suffix}", True, t_color), (GRID_DIM * CELL_SIZE + 20, offset))

        # Enemy Status
        offset += 30
        self.screen.blit(self.font.render("ENEMIES:", True, (150, 150, 150)), (GRID_DIM * CELL_SIZE + 20, offset))
        for name, data in self.env.enemies.items():
            offset += 25
            hp_val = int(max(0, data['hp']))
            e_color = COLORS["Boss"] if name == "Boss" else COLORS["Minion"]
            if hp_val <= 0:
                e_color = (80, 40, 40)
                e_status = "[SLAYED]"
            else: e_status = f"HP: {hp_val:2}"
            self.screen.blit(self.font.render(f"{name:8} | {e_status}", True, e_color), (GRID_DIM * CELL_SIZE + 20, offset))

        # Combat Log
        log_y = 360
        pygame.draw.line(self.screen, (80, 80, 80), (GRID_DIM * CELL_SIZE + 10, log_y - 10), (WIDTH - 10, log_y - 10))
        for i, entry in enumerate(self.log):
            c = (255, 140, 0) if "DM" in entry else (180, 180, 180)
            self.screen.blit(self.font.render(entry, True, c), (GRID_DIM * CELL_SIZE + 10, log_y + (i * 18)))

    def run(self):
        running = True
        while running:
            self.screen.fill((15, 15, 20))
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN and self.game_over:
                    if event.key == pygame.K_r: self.reset_sim()

            if not self.game_over:
                active_name = self.env.turn_queue[self.env.current_turn_idx]
                
                # --- TURN LOGIC ---
                if active_name in self.env.players:
                    p = self.env.players[active_name]
                    if p['hp'] > 0:
                        # Character uses Aggression-Aware Heuristic via TacticalSearch
                        path = self.search_ai.a_star_path(tuple(p['pos']), tuple(self.env.enemies['Boss']['pos']), self.env)
                        move = path[0] if path else p['pos']
                        self.env.step(move)
                        
                        if self.env.is_last_stand(active_name) and random.random() < 0.3:
                            self.add_log(f"{active_name} FIGHTS DESPERATELY!", "DM")
                    else: self.env._advance_turn()
                
                else:
                    e = self.env.enemies[active_name]
                    if e['hp'] > 0:
                        if active_name == "Boss":
                            # --- ENRAGED BOSS LOGIC ---
                            is_enraged = self.boss_ai.get_desperation_factor(self.env)
                            state_obs = np.array([e['pos']])
                            legal_actions = [0, 1, 2, 3] # UP, DOWN, RIGHT, LEFT
                            
                            # Choose action with reduced Epsilon if Enraged
                            action_idx = self.boss_ai.choose_action(state_obs, legal_actions, is_enraged=is_enraged)
                            
                            d = {0:(-1,0), 1:(1,0), 2:(0,1), 3:(0,-1)}[action_idx]
                            target = (e['pos'][0]+d[0], e['pos'][1]+d[1])
                            
                            # Perform Step
                            prev_obs = state_obs
                            next_state_dict, reward, _ = self.env.step(target)
                            next_obs = np.array([e['pos']])
                            
                            # Online Learning Update
                            self.boss_ai.learn(prev_obs, action_idx, reward, next_obs, is_last_stand=is_enraged)
                            
                            if is_enraged: self.add_log("THE BOSS IS ENRAGED!", "DM")
                        else:
                            # Minion movement
                            m_moves = [(e['pos'][0]+dx, e['pos'][1]+dy) for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]]
                            valid_m = [m for m in m_moves if 0<=m[0]<10 and 0<=m[1]<10]
                            if valid_m: self.env.step(random.choice(valid_m))
                            else: self.env._advance_turn()
                    else: self.env._advance_turn()

                # --- WIN CONDITION ---
                self.game_over, self.winner_banner = self.env._check_done()

                # --- FUZZY DM ---
                avg_hp = np.mean([p['hp'] for p in self.env.players.values()])
                hp_pct = max(0, min(100, (avg_hp / 25.0) * 100))
                try:
                    self.intensity = self.fuzzy.calculate_hazard_level(hp_pct, self.env.turn_count)
                except: self.intensity = 5.0
                
                if self.intensity > 8.5 and random.random() < 0.05:
                    if self.env.spawn_hazard(random.randint(0,9), random.randint(0,9)):
                        self.add_log("The DM intervenes with LAVA!", "DM")

            # --- RENDERING ---
            for x in range(GRID_DIM):
                for y in range(GRID_DIM):
                    rect = (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    has_los = self.sensors.get_line_of_sight(self.env.players['Tank']['pos'], (x, y), self.env.obstacles)
                    color = COLORS["Fog"] if not has_los else \
                            (COLORS["Wall"] if self.env.grid_matrix[x, y] == 1 else \
                             (COLORS["Lava"] if self.env.grid_matrix[x, y] == 2 else (30, 30, 35)))
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, (20, 20, 25), rect, 1)

            for name, p in self.env.players.items():
                if p['hp'] > 0:
                    pygame.draw.rect(self.screen, COLORS[name], (p['pos'][1]*CELL_SIZE+14, p['pos'][0]*CELL_SIZE+14, 36, 36), border_radius=6)
            for name, e in self.env.enemies.items():
                if e['hp'] > 0:
                    c = COLORS["Boss"] if name == "Boss" else COLORS["Minion"]
                    pygame.draw.circle(self.screen, c, (e['pos'][1]*CELL_SIZE+32, e['pos'][0]*CELL_SIZE+32), 22)

            self.draw_dashboard()

            if self.game_over:
                overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 210))
                self.screen.blit(overlay, (0,0))
                msg = self.big_font.render(self.winner_banner, True, (255, 255, 255))
                self.screen.blit(msg, msg.get_rect(center=(WIDTH//2, HEIGHT//2 - 40)))
                retry = self.font.render("PRESS 'R' TO RESTART", True, (0, 255, 180))
                self.screen.blit(retry, retry.get_rect(center=(WIDTH//2, HEIGHT//2 + 30)))

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

if __name__ == "__main__":
    AegisDM_Final().run()