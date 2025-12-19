import random



class Population:

    """
    Contains the population of prompts

    Attributes:

    -   prompts: a list of the prompts

    """

    def __init__(self, prompts):
        self.prompts = prompts

    def evaluate_all(self, model, tokenizer, samples, metric_evaluator,
                     judge_model, judge_tokenizer):
        


        #Evaluate all the prompts in the population and calculate the objective score using three metrics
        #The goal of this score is to try to capture the objective part of a summary
        for p in self.prompts:
            p.compute_objective_score(model, tokenizer, samples, metric_evaluator)

        #Evaluate all the prompts in the population and calculate the pairwise score
        #The goal of this score is to try to evaluate the subjective part of a summary
        for p in self.prompts:
            print(f"PAIRWISE CALCULATION")
            opponents = [opp for opp in self.prompts if opp.prompt != p.prompt]
            p.compute_pairwise_score(opponents, samples, judge_model, judge_tokenizer)
            p.compute_fitness(delta=0.5)


        #Remove all the generated summaries of a prompt in the generation
        for p in self.prompts:
            p.generated_summaries = []


    def select_elite(self, k):
        #Select top k prompts by fitness

        self.prompts.sort(key=lambda p: p.fitness, reverse=True)
        return self.prompts[:k]

    def generate_next_generation(self, mutation_operator, metric_evaluator, elite_k=1):

        elites = self.select_elite(elite_k)
        new_population = elites.copy()

        #Generate the population for the next generation, it starts from the top k prompts
        while len(new_population) < len(self.prompts):
            parent = random.choice(elites)
            child = parent.mutate(mutation_operator, metric_evaluator)
            new_population.append(child)

        #New population
        self.prompts = new_population
