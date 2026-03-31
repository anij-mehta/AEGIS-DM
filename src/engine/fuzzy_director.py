import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyDirector:
    def __init__(self):
        # Antecedents (Inputs)
        self.party_health = ctrl.Antecedent(np.arange(0, 101, 1), 'party_health')
        self.turn_count = ctrl.Antecedent(np.arange(0, 51, 1), 'turn_count')
        
        # Consequent (Output)
        self.hazard_intensity = ctrl.Consequent(np.arange(0, 11, 1), 'hazard_intensity')

        # Membership Functions
        self.party_health['critical'] = fuzz.trimf(self.party_health.universe, [0, 0, 30])
        self.party_health['healthy'] = fuzz.trimf(self.party_health.universe, [40, 100, 100])
        
        self.turn_count['early'] = fuzz.trimf(self.turn_count.universe, [0, 0, 10])
        self.turn_count['stagnant'] = fuzz.trimf(self.turn_count.universe, [15, 50, 50])

        self.hazard_intensity['low'] = fuzz.trimf(self.hazard_intensity.universe, [0, 0, 5])
        self.hazard_intensity['high'] = fuzz.trimf(self.hazard_intensity.universe, [5, 10, 10])

        # Rules 
        rule1 = ctrl.Rule(self.party_health['healthy'] & self.turn_count['stagnant'], self.hazard_intensity['high'])
        rule2 = ctrl.Rule(self.party_health['critical'], self.hazard_intensity['low'])

        self.director_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem([rule1, rule2]))

    def calculate_hazard_level(self, current_hp_percent, current_turn):
        self.director_sim.input['party_health'] = current_hp_percent
        self.director_sim.input['turn_count'] = current_turn
        self.director_sim.compute()
        return self.director_sim.output['hazard_intensity']