import numpy as np
import collections
import random

class QLearningBoss:
    def __init__(self, action_space_size, alpha=0.1, gamma=0.9, epsilon=0.6):
        self.q_table = collections.defaultdict(lambda: np.zeros(action_space_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon  # Higher starting epsilon for better exploration
        self.action_space_size = action_space_size

    def get_desperation_factor(self, env):
        """Returns True if all minions are defeated."""
        minions = [name for name in env.enemies if "Minion" in name]
        if not minions: return True
        return all(env.enemies[m]['hp'] <= 0 for m in minions)

    def calculate_commander_reward(self, env, base_reward):
        """
        REWARD SHAPING:
        1. Proximity to Wizard (Breadcrumbs)
        2. Proximity to Minions (Pack Logic)
        """
        shaped_reward = base_reward
        boss_pos = env.enemies['Boss']['pos']
        wiz_pos = env.players['Wizard']['pos']
        
        # 1. Proximity to Wizard: Reward getting closer, penalize staying far
        # Manhattan distance
        dist_to_wiz = abs(boss_pos[0] - wiz_pos[0]) + abs(boss_pos[1] - wiz_pos[1])
        # Max distance on 10x10 is 18. We reward being within 5 tiles.
        if dist_to_wiz < 5:
            shaped_reward += 2.0  # Encouragement for closing in
        elif dist_to_wiz > 8:
            shaped_reward -= 0.5  # Penalty for "cowardice" or wandering off
            
        # 2. Pack Logic: Reward staying near living Minions
        living_minions = [m for m in env.enemies if "Minion" in m and env.enemies[m]['hp'] > 0]
        for m_name in living_minions:
            m_pos = env.enemies[m_name]['pos']
            dist_to_minion = abs(boss_pos[0] - m_pos[0]) + abs(boss_pos[1] - m_pos[1])
            if dist_to_minion <= 2:
                shaped_reward += 1.5  # "Strength in numbers" bonus
                
        return shaped_reward

    def choose_action(self, state_obs, legal_actions, is_enraged=False):
        """Epsilon-Greedy selection with Enraged override."""
        state_key = tuple(state_obs.flatten())
        
        # Reduce exploration if Enraged to focus on lethal moves
        current_epsilon = self.epsilon / 2.0 if is_enraged else self.epsilon
        
        if random.random() < current_epsilon:
            return random.choice(legal_actions)
        
        qs = self.q_table[state_key]
        # Mask illegal actions with very low value
        mask = np.full(qs.shape, -999.0)
        for a in legal_actions:
            mask[a] = qs[a]
        
        return np.argmax(mask)

    def learn(self, state, action, reward, next_state, is_last_stand=False):
        """Standard Q-Update with Last Stand multiplier."""
        s_key = tuple(state.flatten())
        ns_key = tuple(next_state.flatten())
        
        # Double rewards during Last Stand to reinforce aggressive end-game
        actual_reward = reward * 2.0 if is_last_stand else reward
        
        # Q-Learning Formula
        best_next = np.max(self.q_table[ns_key])
        current_q = self.q_table[s_key][action]
        
        # Update rule
        self.q_table[s_key][action] += self.alpha * (actual_reward + self.gamma * best_next - current_q)