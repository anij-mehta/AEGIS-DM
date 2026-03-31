import numpy as np
import random
from src.engine.logic import ActionValidator
from src.engine.semantic_network import SemanticEngine

class AegisEnv:
    """
    Enhanced Multi-Agent Environment.
    Manages 4 Players vs. 3 Enemies with dynamic map generation and turn-based initiative.
    """
    def __init__(self, grid_dim=10, attributes_path="data/static_attributes.json"): ## EDIT THIS LINE
        self.grid_dim = grid_dim
        self.validator = ActionValidator()
        self.semantic = SemanticEngine(attributes_path)
        self.reset()

    def map_generator(self):
        """Generates 10-15 random obstacles and ensures paths remain possible."""
        self.grid_matrix = np.zeros((self.grid_dim, self.grid_dim), dtype=int)
        num_obstacles = random.randint(10, 15)
        self.obstacles = []
        
        while len(self.obstacles) < num_obstacles:
            obs = (random.randint(0, self.grid_dim-1), random.randint(0, self.grid_dim-1))
            # Don't place obstacles in corners (spawn zones) or duplicates
            if obs not in self.obstacles and obs not in [(0,0), (0,1), (1,0), (9,9), (9,8), (8,9)]:
                self.obstacles.append(obs)
                self.grid_matrix[obs] = 1

    def reset(self):
        """Initializes 4 Players and 3 Enemies (Boss + 2 Minions)."""
        self.map_generator()
        self.turn_count = 0
        
        # 1. Initialize Player Party
        self.players = {
            "Tank": {"pos": [1, 1], "hp": 35, "class": "Fighter", "element": "Physical"},
            "Assassin": {"pos": [0, 1], "hp": 20, "class": "Rogue", "element": "Physical"},
            "Wizard": {"pos": [1, 0], "hp": 16, "class": "Wizard", "element": "Fire"},
            "Cleric": {"pos": [0, 0], "hp": 22, "class": "Cleric", "element": "Life"}
        }

        # 2. Initialize Boss Team
        self.enemies = {
            "Boss": {"pos": [8, 8], "hp": 85, "type": "Dragon", "element": "Fire"},
            "Minion_1": {"pos": [9, 8], "hp": 25, "type": "Undead", "element": "Cold"},
            "Minion_2": {"pos": [8, 9], "hp": 25, "type": "Undead", "element": "Cold"}
        }

        # 3. Initiative Queue (Simple Round-Robin: Players then Enemies)
        self.turn_queue = list(self.players.keys()) + list(self.enemies.keys())
        self.current_turn_idx = 0
        
        return self._get_obs()

    def _get_obs(self):
        """Returns the full state vector for all 7 entities."""
        return {
            "players": {name: d.copy() for name, d in self.players.items()},
            "enemies": {name: d.copy() for name, d in self.enemies.items()},
            "grid": self.grid_matrix.copy(),
            "active_entity": self.turn_queue[self.current_turn_idx]
        }

    def step(self, action):
        """
        Executes a turn for the active entity with Boundary Protection.
        """
        active_name = self.turn_queue[self.current_turn_idx]
        is_player = active_name in self.players
        entity_data = self.players[active_name] if is_player else self.enemies[active_name]
        
        if entity_data["hp"] <= 0:
            self._advance_turn()
            return self._get_obs(), 0, self._check_done()

        # --- SAFETY FIX: Coordinate Clipping ---
        # Ensure target is within [0, grid_dim-1]
        target_x = max(0, min(self.grid_dim - 1, action[0]))
        target_y = max(0, min(self.grid_dim - 1, action[1]))
        safe_action = (target_x, target_y)

        # 1. Execute Movement / Action
        if self.validator.validate_movement(entity_data["pos"], safe_action, speed=6):
            # Check for Walls
            if self.grid_matrix[safe_action] == 0:
                # Check for Occupancy (don't step on teammates/enemies)
                all_pos = [tuple(p["pos"]) for p in self.players.values()] + \
                          [tuple(e["pos"]) for e in self.enemies.values()]
                
                if safe_action not in all_pos:
                    entity_data["pos"] = list(safe_action)

        # 2. Combat Resolution
        self.resolve_squad_combat(active_name)

        # 3. Environmental Hazards
        # Use tuple(entity_data["pos"]) to ensure correct indexing
        if self.grid_matrix[tuple(entity_data["pos"])] == 2: # Lava
            entity_data["hp"] -= 5

        # 4. Advance State
        done = self._check_done()
        self._advance_turn()
        self.turn_count += 1
        
        return self._get_obs(), 0, done

    def resolve_squad_combat(self, attacker_name):
        """Checks for adjacency and resolves damage using Semantic multipliers."""
        is_player = attacker_name in self.players
        attacker = self.players[attacker_name] if is_player else self.enemies[attacker_name]
        targets = self.enemies if is_player else self.players

        for t_name, target in targets.items():
            if target["hp"] > 0 and self._is_adjacent(attacker["pos"], target["pos"]):
                # Get elements for Semantic Engine
                atk_element = attacker.get("element", "Physical")
                tgt_type = target.get("type", target.get("class", "Unknown"))
                
                modifier = self.semantic.get_interaction_modifier(atk_element, tgt_type)
                damage = 4 * modifier
                target["hp"] -= damage
                return  # One attack per turn

    def _is_adjacent(self, p1, p2):
        return abs(p1[0]-p2[0]) <= 1 and abs(p1[1]-p2[1]) <= 1

    def _advance_turn(self):
        self.current_turn_idx = (self.current_turn_idx + 1) % len(self.turn_queue)

    def _check_done(self):
        """Returns (is_done, winner_message)"""
        players_alive = [n for n, p in self.players.items() if p['hp'] > 0]
        enemies_alive = [n for n, e in self.enemies.items() if e['hp'] > 0]

        if not players_alive:
            return True, "BOSS WINS - SQUAD ELIMINATED"
        if "Boss" not in enemies_alive:
            return True, "PLAYERS WIN - THE BEAST IS SLAIN"
        if not enemies_alive:
            return True, "PLAYERS WIN - ALL FOES PURGED"
            
        return False, ""

    def is_last_stand(self, entity_name):
        """Returns True if the entity's teammates are all dead."""
        if entity_name in self.players:
            return sum(1 for p in self.players.values() if p['hp'] > 0) == 1
        else:
            return sum(1 for e in self.enemies.values() if e['hp'] > 0) == 1

    def spawn_hazard(self, x, y):
        """DM Manipulation."""
        occupied = [tuple(p["pos"]) for p in self.players.values()] + \
                   [tuple(e["pos"]) for e in self.enemies.values()]
        if (x, y) not in occupied and (x, y) not in self.obstacles:
            self.grid_matrix[x, y] = 2
            return True
        return False