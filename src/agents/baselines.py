import random

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, state, legal_actions):
        return random.choice(legal_actions)

class GreedyAgent:
    """An agent that always moves toward the nearest opposing team member."""
    def __init__(self, agent_type="player"):
        self.agent_type = agent_type

    def choose_action(self, state, legal_actions):
        # 1. Identify the opposing team
        if self.agent_type == "player":
            enemies = state['enemies']
        else:
            enemies = state['players']

        # 2. Get the current position of the entity currently taking its turn
        # We use the 'active_entity' key provided by aegis_env.py
        active_name = state.get('active_entity')
        all_players = state.get('players', {})
        all_enemies = state.get('enemies', {})
        
        # Merge dictionaries to find the active entity's data
        combined_entities = {**all_players, **all_enemies}
        
        if active_name in combined_entities:
            current_pos = combined_entities[active_name]['pos']
        else:
            # Fallback if state is missing names: use first legal action
            current_pos = legal_actions[0] if legal_actions else (0,0)
        
        # 3. Find the closest living enemy
        living_enemies = [data['pos'] for data in enemies.values() if data['hp'] > 0]
        
        if not living_enemies:
            return random.choice(legal_actions) if legal_actions else None

        # 4. Pick the move that minimizes Manhattan distance to the nearest enemy
        def dist_to_nearest_enemy(move):
            if isinstance(move, (tuple, list)):
                return min([abs(move[0] - e[0]) + abs(move[1] - e[1]) for e in living_enemies])
            return 999

        return min(legal_actions, key=dist_to_nearest_enemy)