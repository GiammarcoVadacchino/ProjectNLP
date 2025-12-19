from copy import deepcopy
import numpy as np
import torch


class Prompt:

    """

    Represents a text prompt used for generating summaries with a language model

    Attributes:

    - prompt: the prompt text for summary generation
    - text: the article or input text to summarize
    - metric_values: stores computed metrics for summaries
    - objective_score: aggregated score for objective metrics
    - pairwise_score: score from judge LLM comparisons
    - fitness: combined fitness value
    - generated_summaries: list of summaries generated for input samples

    """


    def __init__(self, prompt: str, text: str):
        self.prompt = prompt
        self.text = text
        self.metric_values = {"KCS": 0.0, "BERTScore": 0.0, "LengthScore": 0.0}
        self.objective_score = None
        self.pairwise_score = None
        self.fitness = None
        self.generated_summaries = []

    def compute_objective_score(self, model, tokenizer, samples, metric_evaluator, device="mps"):
 
        """
        Generate summaries for each sample using the prompt and compute objective metrics 
        such as KCS, BERTScore, and LengthScore.
        Updates metric_values and objective_score.
        """

        all_kcs, all_bertscore, all_length = [], [], []

        for sample in samples:

            self.text = sample.article_text
            
            #Summary Generation
            prompt_input = f"{self.prompt}\n{self.text}"
            inputs = tokenizer(prompt_input, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_length=30,
                    do_sample=True,
                    top_k=40,
                    top_p=0.95,
                    temperature=0.8
                )

            summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            self.generated_summaries.append(summary)


            #Compute the metrics 
            #TODO: sobstitute BLEU metric with a custom metric that identify how much keywords of the article are also in the generated summary, this beacuse BLEU has always the lowest value among the other metrics 
            # and so the objective of the metrics aware mutation is always the same.
            kcs = metric_evaluator.compute_kcs_score(summary, sample.summary)
            bscore = metric_evaluator.compute_bertscore(summary, sample.summary)
            length = metric_evaluator.compute_length_score(summary, sample.summary)

            all_kcs.append(kcs)
            all_bertscore.append(bscore)
            all_length.append(length)

            #Print stats for the current sample
            print(f"\nPrompt: {self.prompt}")
            print(f"Text: {sample.article_text}")
            print(f"Exp summary: {sample.summary}")
            print(f"LLM summary: {summary}")
            print(f"KCS: {kcs:.4f}, BERTScore: {bscore:.4f}, LengthScore: {length:.4f}\n")


        #Calculate the prompt metrics with a weighted sum, NOTE: maybe the weights has to be fine tuned, but idk if i'll do it.
        self.metric_values["KCS"] = 0.33 * np.mean(all_kcs)
        self.metric_values["BERTScore"] = 0.33 * np.mean(all_bertscore)
        self.metric_values["LenghtScore"] = 0.33 * np.mean(all_length)

        #Calculate the objective score. This metrics is the sum of the three metrics, the goal of this metric is to try to capture and represents the objective part in a summary.
        #(e.g tell if the lenght of the generated summary is kinda similar to the gt summary in the dataset is kinda an objective thing or surely a less subjective.)
        self.objective_score = self.metric_values["KCS"] + self.metric_values["BERTScore"] + self.metric_values["LenghtScore"]

        #The goal of the judje model is to compare two summary of the same article and give a reward to the prompt that corresponds to the winner generated summary.
        #This process simulate the RLHF but the feedback is not given by a human but by anther LLM, the goal is to simulate a human that reads and evaluate both summary and in a 
        #subjective way declare a winner, the goal is to reward the model that is better in a subjective way (NOTE: this because summarize a text is naturally a subjective thing).
        #These process is usefull because penalize the summaries that are inventing things or contanins some allucinations, this is done by a judje LLM that controls coherence and other aspects of the summary that
        #the metrics don't take care.
        #NOTE: in this process the current prompt is vs all or (a subset) of other prompts, so che computational time is higher, maybe is better to do this challenges not for all the training samples, but 
        #only a random subset


    def mutate(self, mutation_operator, metric_evaluator):

        """
        Return a mutated copy of the current prompt by modifying the weakest metric.
        """

        base = deepcopy(self)

        weakest = metric_evaluator.identify_weakest(self.metric_values)
        mutated = mutation_operator.apply(base, weakest)

        mutated.metric_values = {}
        mutated.fitness = None
        mutated.objective_score = None
        mutated.pairwise_score = None

        return mutated
    

    def compute_pairwise_score(self, opponent_prompts, samples, judge_model, judge_tokenizer):
        """
        Compute the average pairwise score by comparing this prompt's summaries with 
        summaries from other prompts using a judge LLM.
        Updates pairwise_score.
        """
        wins = 0
        total = 0
        print(f"Computing pairwise score for prompt: {self.prompt}")
        for opp in opponent_prompts:
            print(f"  Against opponent prompt: {opp.prompt}")
            for idx, sample in enumerate(samples):
                summary_a = self.generated_summaries[idx]
                summary_b = opp.generated_summaries[idx]
                result = self.llm_pairwise_judge(sample.article_text, summary_a, summary_b,
                                                judge_model, judge_tokenizer)
                wins += result
                total += 1
        self.pairwise_score = wins / max(total, 1)



    @staticmethod #NOTE: we don't need to access self
    def llm_pairwise_judge(article, summary_a, summary_b, judge_model, judge_tokenizer):

        """
        Compute the average pairwise score by comparing this prompt's summaries with 
        summaries from other prompts using a judge LLM.
        Updates pairwise_score.
        """
        prompt = f"""
            Article:
            {article}

            A:
            {summary_a}

            B:
            {summary_b}

            Which summary is better in terms of factuality, coverage, and clarity?
            Answer with: A, B, or Tie.
        """
        print(f"      Judge prompt:\n{prompt}\n")

        inputs = judge_tokenizer(prompt,return_tensors="pt")
        inputs = {k: v.to(judge_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = judge_model.generate(**inputs, max_length=5)
        decision = judge_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"      Judge decision: '{decision}'")
        if "A" in decision:
            return 1.0
        elif "B" in decision:
            return 0.0
        else:
            return 0.5



    def crossover(self, other_prompt):
        """
        Combine this prompt with another prompt to create a child prompt.
        Currently unimplemented.
        """
        pass

    def compute_fitness(self, delta=0.5):
        """
        Combine objective_score and pairwise_score into a single fitness value.
        delta controls the weight between objective and pairwise scores.
        """
        if self.objective_score is None:
            self.objective_score = 0.0
        if self.pairwise_score is None:
            self.pairwise_score = 0.0

        self.fitness = delta * self.objective_score + (1 - delta) * self.pairwise_score
        print(f"Objective score: {self.objective_score:.4f}")
        print(f"Pairwise score computed: {self.pairwise_score:.4f}")
        print(f"Fitness score computed: {self.fitness:.4f}")

    def print_prompt(self):
        """
        Print the text of the prompt.
        """
        print(self.prompt)
