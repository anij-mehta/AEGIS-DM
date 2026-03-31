import networkx as nx
import json

class SemanticEngine:
    """Manages the Knowledge Graph for entity hierarchies and elemental interactions."""
    
    def __init__(self, data_path="data/static_attributes.json"):
        self.graph = nx.DiGraph()
        self._build_hierarchy()
        self.load_attributes(data_path)

    def _build_hierarchy(self):
        """Defines RDF-style triples for entity relationships."""
        # Entity Types [cite: 48]
        self.graph.add_edge("Skeleton", "Undead", relation="is_a")
        self.graph.add_edge("Zombie", "Undead", relation="is_a")
        self.graph.add_edge("Undead", "Monster", relation="is_a")
        
        # Elemental Interactions [cite: 53]
        self.graph.add_edge("Fire", "Ice", relation="melts")
        self.graph.add_edge("Fire", "Undead", relation="weakness_of")
        self.graph.add_edge("Radiant", "Undead", relation="effective_against")
        self.graph.add_edge("Life", "Undead", relation="damages") 

    def load_attributes(self, path):
        """Loads static attributes into the engine context."""
        with open(path, 'r') as f:
            self.attributes = json.load(f)

    def get_interaction_modifier(self, source_element, target_entity_type):
        """Queries the Semantic Network to find damage modifiers."""
        try:
            # Check for direct relationship in graph
            if self.graph.has_edge(source_element, target_entity_type):
                relation = self.graph[source_element][target_entity_type]['relation']
                if relation in ["effective_against", "damages", "weakness_of"]:
                    return 2.0  # Double damage for weaknesses
            return 1.0
        except Exception:
            return 1.0

    def get_entity_stats(self, entity_name, category="classes"):
        """Retrieves raw data from the JSON library[cite: 47]."""
        return self.attributes.get(category, {}).get(entity_name, {})