import random
from copy import deepcopy

class MutationOperator:
    def __init__(self):
        self.operator_map = {
            "KCS": [self.lexical_overlap],
            "BERTScore": [self.semantic_fidelity],
            "LengthScore": [self.length_control],
        }

    def apply(self, prompt, target_metric):

        print(f"Prompt: {prompt.prompt} (SHOULD BE UPDATED IN NEXT GENERATION)")

        new_prompt = deepcopy(prompt)
        fn = random.choice(self.operator_map[target_metric])
        new_prompt.prompt = fn(prompt.prompt)
        new_prompt.fitness = None
        new_prompt.metric_values = {}



        return new_prompt

    def lexical_overlap(self, p):
        return p + "\nUse exact wording and phrases from the article."

    def semantic_fidelity(self, p):
        return p + "\nPreserve meaning exactly, avoid paraphrasing errors."

    def length_control(self, p):
        return p + "\nWrite a summary with similar length to the reference."
