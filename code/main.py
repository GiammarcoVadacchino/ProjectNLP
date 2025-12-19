# main.py

"""
Main entry point for prompt evolution pipeline for summarization.

Loads dataset, initializes model, tokenizer, prompts, and runs the
evolution pipeline for a number of generations. Designed to support
T5-based summarization models.
"""

from dataset import Dataset
from prompt import Prompt
from pipeline import PromptEvolutionPipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration

DATA_PATH = "../data/summarization.json"
NUM_GENERATIONS = 2
MODEL_NAME = "google/flan-t5-small"  # 80M parameters
DEVICE = "mps"


def main():
    """
    Main execution function.

    1. Initializes the T5 model and tokenizer.
    2. Loads and preprocesses the dataset.
    3. Initializes a population of prompts.
    4. Sets up the prompt evolution pipeline.
    5. Runs the evolution for a predefined number of generations.
    6. Optional: tests the best prompts on a held-out set.
    7. Optional: performs result analysis.
    """

    # 1. Setup model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

    # 2. Load and preprocess dataset
    dataset = Dataset()
    dataset.load_data(DATA_PATH)
    dataset.tokenize_samples(tokenizer)
    dataset.assign_buckets(thresholds={'short': 150, 'medium': 250, 'long': 350})
    dataset.split_train_test(test_size=0.5)

    # 3. Initialize initial prompt population
    initial_prompts = [
        Prompt(prompt="Summarize this following text:", text=""),
        Prompt(prompt="Summarize this:", text="")
    ]

    # 4. Setup prompt evolution pipeline
    pipeline = PromptEvolutionPipeline(
        model, tokenizer, dataset, initial_prompts, NUM_GENERATIONS
    )

    # 5. Run the evolution / local search
    pipeline.run()

    # 6. Optional: test the best prompts on the test set
    # best_prompts = pipeline.test_best_prompts(k=1)

    # 7. Optional: perform result analysis
    # pipeline.analyze_results()


if __name__ == "__main__":
    main()
