from population import Population
from metrics import MetricEvaluator
from mutation import MutationOperator
from transformers import T5Tokenizer, T5ForConditionalGeneration


DEVICE = "mps"
JUDGE_MODEL_NAME = "google/flan-t5-base" #(240M)

class PromptEvolutionPipeline:
    """
    Coordinates the process of prompt evolution for NLP models.

    Attributes:
       
    -   model: The main language model used for evaluating prompts.
    -   tokenizer: Tokenizer associated with the main model.
    -   dataset: Dataset object containing train and test samples.
    -   population: Population object managing the current set of prompts.
    -   metric_evaluator: MetricEvaluator instance used to assess prompt performance.
    -   mutation_operator: MutationOperator instance used to generate new prompt variations.
    -   num_generations: Total number of generations for the evolution process.
    """


    def __init__(self, model, tokenizer, dataset, initial_prompts, num_generations):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.population = Population(initial_prompts)
        self.metric_evaluator = MetricEvaluator()
        self.mutation_operator = MutationOperator()
        self.num_generations = num_generations

    def run(self):
        """
        Executes the evolutionary prompt optimization process. 
        Steps per generation:

            1. Evaluate all prompts in the current population on training samples.
            2. Select elite prompts based on fitness.
            3. Generate a new population via mutation and optionally crossover.
            4. Repeat for the defined number of generations.

        """

        for gen in range(self.num_generations):
            print(f"\n=== Generation {gen+1} ===\n")

            #Evaluate all the prompts in the current population
            self.population.evaluate_all(
                model=self.model,
                tokenizer=self.tokenizer,
                samples=self.dataset.train_samples,
                metric_evaluator=self.metric_evaluator,
                judge_model= T5ForConditionalGeneration.from_pretrained(JUDGE_MODEL_NAME).to(DEVICE),
                judge_tokenizer= T5Tokenizer.from_pretrained(JUDGE_MODEL_NAME)
            )

            #Select the best K prompts of the population
            elite_size = max(1, len(self.population.prompts) // 2)
            #elite_prompts = self.population.select_elite(k=elite_size)
            print(f"Selected {elite_size} elite prompts")


            #Print the best prompt 
            best_prompt = max(self.population.prompts, key=lambda p: p.fitness)
            print("Best prompt this generation:\n")
            best_prompt.print_prompt()
            print(f"Best Prompt Fitness: {best_prompt.fitness}\n")

            #Create the next population for the next generation
            self.population.generate_next_generation(
                mutation_operator=self.mutation_operator,
                metric_evaluator=self.metric_evaluator, 
                elite_k=1
            )


    def test_best_prompts(self, k=3):
        """
        Evaluate the best k prompts on test set
        """
        best_prompts = sorted(self.population.prompts, key=lambda p: p.fitness, reverse=True)[:k]
        for prompt in best_prompts:
            prompt.evaluate(self.model, self.tokenizer, self.dataset.test_samples, self.metric_evaluator)

        return best_prompts
