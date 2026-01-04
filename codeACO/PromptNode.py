class PromptNode:
    def __init__(self, name: str, text: str = "", optional: bool = True):
        self.name = name
        self.text = text
        self.optional = optional
        self.children = []

    def add_child(self, child: "PromptNode", initial_pheromone: float = 0.01):
        if child not in self.children:
            self.children.append(child)