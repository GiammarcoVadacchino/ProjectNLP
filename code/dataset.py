from sklearn.model_selection import train_test_split
import json
import random

seed = random.seed(20)

DATA_PATH = "../data/summarization.json"

class Sample:
    """
    Represent a single summarization sample

    Attributes:

    - article_text: str → complete text of the article (FIXED)
    - summary: str → ground truth summary
    - prompt: str → prompt used for summarization (MUTABLE)
    - article_tokens: int → number of input tokens of the article text
    - prompt_tokens: int → number of input tokens of the prompt
    - total_tokens: int → number of tokens of prompt + article
    - bucket: str → bucket of the number of input tokens of the article ('short', 'medium', 'long')
    """

    def __init__(self, article_text: str, summary: str):
        self.article_text = article_text
        self.summary = summary

        self.prompt = ""
        self.article_tokens = None
        self.prompt_tokens = None
        self.total_tokens = None
        self.bucket = None

    def set_prompt(self, prompt: str):
        """
        Set the prompt for the sample.
        The prompt is expected to change across generations.
        """
        self.prompt = prompt

    def build_input(self):
        """
        Build the full input for the model.
        NOTE: prompt and article text are kept separated for better control.
        """
        return f"{self.prompt}\n\n{self.article_text}"

    def tokenize(self, tokenizer):
        """
        Calculate the number of input tokens.

        NOTE:
        - Tokens are computed separately for prompt and article
        - Buckets are assigned ONLY based on article length
        This is needed to analyze how metrics change with respect
        to the input article length, independently of the prompt.
        """
        article_ids = tokenizer(self.article_text)["input_ids"]
        prompt_ids = tokenizer(self.prompt)["input_ids"]
        total_ids = tokenizer(self.build_input())["input_ids"]

        self.article_tokens = len(article_ids)
        self.prompt_tokens = len(prompt_ids)
        self.total_tokens = len(total_ids)

    def assign_bucket(self, thresholds):
        """
        Assign a bucket for each sample based on article length.
        """
        if self.article_tokens <= thresholds["short"]:
            self.bucket = "short"
        elif self.article_tokens <= thresholds["medium"]:
            self.bucket = "medium"
        else:
            self.bucket = "long"


class Dataset:
    """
    Contains and manages all the samples.

    Attributes:

    - self.samples: list → contains couples of input and output (article, summary (GT))
    - self.train_samples: samples used to find the best prompts
    - self.test_samples: samples used to test the generalization of the best prompts found in training
    - self.buckets: dictionary of samples grouped by bucket
    - self.number_of_samples: number of total samples

    Include:
    - load the data
    - tokenization and assignment of a bucket
    - split the data into train and test set
    """

    def __init__(self, samples=None):
        self.samples = samples or []
        self.train_samples = []
        self.test_samples = []
        self.buckets = {"short": [], "medium": [], "long": []}
        self.number_of_samples = 10

    def load_data(self, data_path):
        """
        Load the data from a JSON file and build the samples.

        NOTE:
        - Only article text and summary are loaded
        - Prompt is assigned later and can change across experiments
        """
        with open(data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        sampled_data = random.sample(data["Instances"], self.number_of_samples)

        for item in sampled_data:

            print(f"Input text: {item['input']}")
            print(f"Output text: {item['output'][0]}")

            self.samples.append(
                Sample(
                    article_text=item["input"],
                    summary=item["output"][0]
                )
            )

        

    def set_prompt_for_all(self, prompt: str):
        """
        Assign the same prompt to all samples.
        Useful for prompt comparison and evolutionary optimization.
        """
        for sample in self.samples:
            sample.set_prompt(prompt)

    def tokenize_samples(self, tokenizer):
        """
        Tokenize all the samples.
        """
        for sample in self.samples:
            sample.tokenize(tokenizer)

    def assign_buckets(self, thresholds):
        """
        Assign the corresponding bucket for each sample.
        """
        for sample in self.samples:
            sample.assign_bucket(thresholds)
            self.buckets[sample.bucket].append(sample)

    def split_train_test(self, test_size=0.2, seed=42):
        """
        Split all the data into a training and test set.

        Test set is needed to evaluate the generalization
        of the best prompts found during training.
        """
        train, test = train_test_split(
            self.samples,
            test_size=test_size,
            random_state=seed
        )
        self.train_samples = train
        self.test_samples = test

    def get_samples_by_bucket(self, bucket_name, train=True):
        """
        Given a bucket, returns the list of samples that
        correspond to the given bucket.
        """
        target = self.train_samples if train else self.test_samples
        return [s for s in target if s.bucket == bucket_name]

    def print_samples(self):
        """
        Print samples with prompt, article and summary.
        Useful for debugging.
        """
        for s in self.samples:
            print(
                f"PROMPT:\n{s.prompt}\n\n"
                f"ARTICLE:\n{s.article_text}\n\n"
                f"SUMMARY:\n{s.summary}\n"
                f"{'-'*50}"
            )


"""
dataset = Dataset()
dataset.load_data(DATA_PATH)
dataset.print_samples()
"""