from z3 import Bool, And, Implies, Int, Solver, sat

class ActionValidator:
    """Uses Z3 for Predicate Calculus to validate moves."""
    
    def __init__(self):
        # Define Z3 Variables
        self.is_turn = Bool("is_turn")
        self.has_action = Bool("has_action")
        self.is_alive = Bool("is_alive")
        self.in_range = Bool("in_range")

    def validate_attack(self, agent_status):
        s = Solver()
        
        # State Constraints
        state_constraints = And(
            self.is_turn == agent_status['is_turn'],
            self.has_action == agent_status['has_action'],
            self.is_alive == (agent_status['hp'] > 0),
            self.in_range == agent_status['in_range']
        )
        
        # Rule: Success requires all preconditions
        rule = Implies(state_constraints, And(self.is_turn, self.has_action, self.is_alive, self.in_range))
        
        s.add(rule)
        s.add(state_constraints) # Check if this specific state is valid
        return s.check() == sat

    def validate_movement(self, current_pos, target_pos, speed):
        s = Solver()
        dist = abs(current_pos[0] - target_pos[0]) + abs(current_pos[1] - target_pos[1])
        
        move_dist = Int("move_dist")
        max_speed = Int("max_speed")
        
        s.add(move_dist == dist)
        s.add(max_speed == speed)
        s.add(move_dist <= max_speed)
        
        return s.check() == sat