import heapq
from src.agents.heuristics import tank_heuristic, assassin_heuristic, glass_cannon_heuristic

class TacticalSearch:
    def __init__(self, validator, semantic_engine):
        self.validator = validator
        self.semantic = semantic_engine
        # Map classes to their base heuristic functions
        self.base_heuristics = {
            "Tank": tank_heuristic,
            "Assassin": assassin_heuristic,
            "Wizard": glass_cannon_heuristic,
            "Cleric": self.healer_heuristic
        }

    def get_combined_heuristic(self, state, agent_name, env):
        """
        Main entry point for AI decision making. 
        Switches to 'Aggression Mode' if the character is the last one standing.
        """
        agent_data = env.players[agent_name]
        base_func = self.base_heuristics.get(agent_name, lambda s, a: 0)
        
        # Calculate base score (Role-based)
        score = base_func(state, agent_data)
        
        # --- LAST STAND LOGIC ---
        if env.is_last_stand(agent_name):
            # 1. Identify the Boss
            boss_pos = env.enemies['Boss']['pos']
            curr_pos = agent_data['pos']
            
            # 2. Calculate Distance to Boss
            dist_to_boss = abs(curr_pos[0] - boss_pos[0]) + abs(curr_pos[1] - boss_pos[1])
            
            # 3. Aggression Boost:
            # We subtract the distance from a high constant so that 
            # getting closer to the Boss results in a much higher score.
            aggression_score = (20 - dist_to_boss) * 5 
            
            # 4. Double the final value
            score = (score + aggression_score) * 2
            
        return score

    def healer_heuristic(self, state, agent):
        """Standard Cleric: Stay near allies."""
        score = agent['hp'] * 2
        allies = [p for name, p in state['players'].items() if name != "Cleric" and p['hp'] > 0]
        
        if allies:
            # Find nearest ally
            min_dist = min([abs(agent['pos'][0] - a['pos'][0]) + abs(agent['pos'][1] - a['pos'][1]) for a in allies])
            score -= min_dist * 3
        return score

    def a_star_path(self, start, goal, env):
        """Enhanced A*: Treats other agents as temporary obstacles."""
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: self._dist(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))

        # Identify all occupied squares (except the goal itself)
        occupied = [tuple(p['pos']) for p in env.players.values() if p['hp'] > 0] + \
                   [tuple(e['pos']) for e in env.enemies.values() if e['hp'] > 0]

        while oheap:
            current = heapq.heappop(oheap)[1]
            if current == goal: return self._reconstruct(came_from, current)
            close_set.add(current)

            for i, j in neighbors:
                neighbor = (current[0] + i, current[1] + j)
                if 0 <= neighbor[0] < env.grid_dim and 0 <= neighbor[1] < env.grid_dim:
                    # Block if Wall or Occupied by another living entity
                    if (env.grid_matrix[neighbor] == 1 or neighbor in occupied) and neighbor != goal:
                        continue
                else: continue

                tentative_g = gscore[current] + 1
                if neighbor in close_set and tentative_g >= gscore.get(neighbor, float('inf')):
                    continue
                
                if tentative_g < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + self._dist(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return []

    def _dist(self, a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    def _reconstruct(self, came_from, current):
        p = []
        while current in came_from:
            p.append(current); current = came_from[current]
        return p[::-1]