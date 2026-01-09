import csv
import random
from typing import List, Tuple
from transformers import T5Tokenizer


class Dataset:
    """
    Dataset for summarization experiments.
    Supports token-length filtering and train/test splitting.
    """

    def __init__(
        self,
        csv_path: str,
        input_col: str,
        output_col: str,
        model_name: str,
    ):
        self.csv_path = csv_path #csv file that contains the dataset
        self.input_col = input_col #col of the input document
        self.output_col = output_col #col of the GT summary
        #tokenizer used for filtering the documents, in that way i can't exceed the max lenght tokens of the model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name) 

        
        self.pairs: List[Tuple[str, str]] = [] #list of couples (input,output)

        self.train_pairs: List[Tuple[str, str]] = [] #training dataset
        self.test_pairs: List[Tuple[str, str]] = [] #test dataset

    def load_with_token_limit(
        self,
        number_of_samples: int,
        target_tokens: int,
        tolerance: int = 20
    ) -> None:
        """
        Load examples whose input length is within target_tokens ± tolerance.
        """

        #random.seed(42)

        min_tokens = target_tokens - tolerance #min number of tokens that a document has to have for being considered
        max_tokens = target_tokens + tolerance #max number of tokens that a document has to have for being considered
        pairs_sample = []

        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            #iterate the rows of the csv file
            for row in reader:
                
                input_text = row[self.input_col] #document
                output_text = row[self.output_col] #summary GT

                n_tokens = len(self.tokenizer.encode(input_text, truncation=False)) #calulate the number of tokens of the document

                #if the threshold is respcted add the document to list, so it can be sampled
                if min_tokens <= n_tokens <= max_tokens:
                    pairs_sample.append((input_text, output_text))

        self.pairs = random.sample(pairs_sample,number_of_samples) #randomly sampling a list of pairs

    def split_train_test(
        self,
        train_fraction: float,
        seed: int,
        shuffle: bool = True,
        
    ) -> None:

        assert 0.0 < train_fraction <= 1.0

        #takes indices of the pairs
        indices = list(range(len(self.pairs)))

        #if shuffle is enabled, shuffle all the pairs
        if shuffle:
            random.seed(seed)
            random.shuffle(indices)

        #split into train and test dataset
        split_idx = int(len(indices) * train_fraction)

        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        #construct test and training dataset
        self.train_pairs = [self.pairs[i] for i in train_idx]
        self.test_pairs = [self.pairs[i] for i in test_idx]


    def __len__(self):
        return len(self.pairs)
