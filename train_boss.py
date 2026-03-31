import numpy as np
import os
import random
from src.environment.aegis_env import AegisEnv
from src.agents.learning import QLearningBoss
from src.agents.baselines import GreedyAgent

def train_squad():
    save_dir = "data/replay_buffer"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "boss_policy.npy")

    env = AegisEnv(grid_dim=10)
    boss_agent = QLearningBoss(action_space_size=4, alpha=0.5, gamma=0.9, epsilon=0.9)
    
    player_baselines = GreedyAgent(agent_type="player")
    minion_baselines = GreedyAgent(agent_type="enemy")

    num_episodes = 5000
    print("Starting Training with Survival Buff...")

    for episode in range(num_episodes):
        state = env.reset()
        
        # --- TRAINING BUFFS ---
        # 1. Give Boss massive HP so it doesn't die before learning
        env.enemies['Boss']['hp'] = 1000 
        # 2. Randomize Turn Order so Boss gets to go first sometimes
        random.shuffle(env.turn_queue)
        
        done = False
        episode_reward = 0
        boss_turns = 0
        boss_agent.epsilon = max(0.1, 0.9 * (0.9995 ** episode))

        while not done:
            active_name = env.turn_queue[env.current_turn_idx]
            
            # --- BOSS TURN ---
            if active_name == "Boss":
                boss_turns += 1
                e_data = env.enemies['Boss']
                state_obs = np.array([e_data['pos']])
                
                # Force exploration
                action_idx = boss_agent.choose_action(state_obs, [0,1,2,3], False)
                dir_map = {0:(-1,0), 1:(1,0), 2:(0,1), 3:(0,-1)}
                target_pos = (e_data['pos'][0]+dir_map[action_idx][0], e_data['pos'][1]+dir_map[action_idx][1])
                
                wiz_hp_before = env.players['Wizard']['hp']
                _, _, done = env.step(target_pos)
                
                # Reward: Big points for hits, tiny points for moving
                step_reward = 100 if env.players['Wizard']['hp'] < wiz_hp_before else -0.05
                shaped_reward = boss_agent.calculate_commander_reward(env, step_reward)
                
                boss_agent.learn(state_obs, action_idx, shaped_reward, np.array([e_data['pos']]), False)
                episode_reward += shaped_reward

            # --- PLAYERS ---
            elif active_name in env.players:
                p_data = env.players[active_name]
                if p_data['hp'] > 0:
                    p_moves = [(p_data['pos'][0]+dx, p_data['pos'][1]+dy) for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]]
                    valid_p = [m for m in p_moves if 0<=m[0]<10 and 0<=m[1]<10]
                    if valid_p:
                        p_action = player_baselines.choose_action(state, valid_p)
                        env.step(p_action)
                    else: env._advance_turn()
                else: env._advance_turn()

            # --- MINIONS ---
            else:
                env._advance_turn() # Skip minions to speed up Boss learning

            state = env._get_obs()
            if env.turn_count > 100: done = True
            if env.players['Wizard']['hp'] <= 0: env.players['Wizard']['hp'] = 25 # Respawn/Heal Wizard

        if (episode + 1) % 100 == 0:
            print(f"Ep {episode+1} | Reward: {episode_reward:.1f} | Boss Turns: {boss_turns} | Eps: {boss_agent.epsilon:.2f}")

    np.save(save_path, dict(boss_agent.q_table))
    print("Training Complete!")

if __name__ == "__main__":
    train_squad()