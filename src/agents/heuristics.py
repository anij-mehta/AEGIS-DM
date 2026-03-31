import math

def manhattan_dist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def tank_heuristic(state, agent):
    """
    Prioritizes high HP retention and positioning between the Boss and glass-cannon allies.
    """
    score = agent.hp * 2  # High value on staying alive
    
    # Identify the 'Boss' and 'Wizard' (Glass Cannon) from state
    boss = state.get('boss_pos')
    wizard = state.get('wizard_pos') # Assuming a multi-agent party state
    
    if boss and wizard:
        # Ideal position for a tank is the midpoint between boss and wizard
        midpoint = ((boss[0] + wizard[0]) // 2, (boss[1] + wizard[1]) // 2)
        dist_to_ideal = manhattan_dist(agent.pos, midpoint)
        score -= dist_to_ideal * 3  # Penalty for being out of position
        
    return score

def assassin_heuristic(state, agent):
    """
    Prioritizes flanking and targeting the enemy with the lowest HP.
    """
    score = 0
    boss_hp = state.get('boss_hp', 100)
    boss_pos = state.get('boss_pos')
    
    # Reward for Boss having low HP
    score += (100 - boss_hp) * 2
    
    # Reward for being close to the target
    if boss_pos:
        dist = manhattan_dist(agent.pos, boss_pos)
        if dist <= 1:
            score += 50  # Strike range bonus
        else:
            score -= dist
            
    return score

def glass_cannon_heuristic(state, agent, sensors):
    """
    Prioritizes maximum distance from the Boss while maintaining Line-of-Sight (LoS).
    """
    score = agent.hp * 5  # Very sensitive to taking damage
    boss_pos = state.get('boss_pos')
    
    if boss_pos:
        dist = manhattan_dist(agent.pos, boss_pos)
        has_los = sensors.get_line_of_sight(agent.pos, boss_pos, state.get('obstacles', []))
        
        if has_los:
            score += dist * 2  # Reward for staying far away but visible
        else:
            score -= 20  # Penalty for losing the target (can't shoot)
            
        if dist < 3:
            score -= 50  # Heavy penalty for being in melee range
            
    return score