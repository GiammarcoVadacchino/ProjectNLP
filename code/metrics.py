import numpy as np
import re
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score

NLP = spacy.load("en_core_web_lg")
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")



class MetricEvaluator:
    """
    Evaluates automatic metrics for generated summaries.


    Attributes:

    - bert_model: the model name for computing BERTScore
    - bleu_smooth: smoothing function for BLEU computation (NOTE: this should be deleted since i don't use BLEU anymore)
        
    """

    def __init__(self, bert_model="distilbert-base-uncased"):
        self.bert_model = bert_model
        self.bleu_smooth = SmoothingFunction().method1

    def compute_bleu(self, prediction: str, reference: str):

        """
        Compute the BLEU score between a predicted and reference summary.
        Returns a value between 0 and 1.
        """
        
        bleu = sentence_bleu(
            [reference.split()],
            prediction.split(),
            smoothing_function=self.bleu_smooth
        )

        return bleu  

    def compute_bertscore(self, prediction: str, reference: str):

        """
        Compute the BERTScore (F1) between a predicted and reference summary.
        Returns a value between 0 and 1.
        """

        _, _, F1 = bert_score(
            [prediction],
            [reference],
            model_type=self.bert_model,
            verbose=False
            
        )

        return F1.mean().item()

    
    def compute_length_score(self, prediction: str, reference: str):

        """
        Compute a normalized length similarity score between prediction and reference.
        Returns 1.0 if lengths match, decreasing as the difference increases.
        """
        
        score = 1.0 - abs(len(prediction.split()) - len(reference.split())) / max(len(reference.split()), 1)
        score = max(score, 0.0)
        return score

 
    def extract_keywords(self, text: str, top_k: int = 15):
        """
        Extract the most relevant keywords from a text using spaCy POS tagging.

        Considers NOUNs, VERBs, and ADJs, removes stopwords and non-alphabetic tokens.
        Returns a list of top_k keywords based on frequency.
        """
        
        text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        doc = NLP(text)

        candidates = []

        for token in doc:
            if token.pos_ in ("NOUN", "VERB", "ADJ") and token.is_alpha and not token.is_stop:
                candidates.append(token.lemma_)

        freq = Counter(candidates)
        keywords = [w for w, _ in freq.most_common(top_k)]
        return keywords

    def kcs_hard(self, article: str, summary: str, top_k: int = 15) -> float:

        """
        Compute the hard Keyword Coverage Score.
        Measures the proportion of top article keywords present in the summary.
        """
        keywords = self.extract_keywords(article, top_k=top_k)
        summary = re.sub(r"[^a-z\s]", "", summary.lower())
        summary_tokens = set(summary.split())
        covered = sum(1 for k in keywords if k in summary_tokens)
        return covered / max(len(keywords), 1)

    def kcs_soft(self, article: str, summary: str, top_k: int = 15, similarity_threshold: float = 0.6) -> float:
        """
        Compute the soft Keyword Coverage Score using embeddings and cosine similarity.

        Returns the average similarity of each keyword to the most similar token in the summary,
        with scores below the threshold considered 0.
        """
        keywords = self.extract_keywords(article, top_k=top_k)
        if len(keywords) == 0 or len(summary.strip()) == 0:
            return 0.0

        summary_tokens = list(set(re.sub(r"[^a-z\s]", "", summary.lower()).split()))
        # embeddings
        keyword_emb = EMBEDDING_MODEL.encode(keywords)
        summary_emb = EMBEDDING_MODEL.encode(summary_tokens)
        sim_matrix = cosine_similarity(keyword_emb, summary_emb)
        best_scores = sim_matrix.max(axis=1)
        covered = [score if score >= similarity_threshold else 0.0 for score in best_scores]
        return float(np.mean(covered))

    def compute_kcs_score(self, article: str, summary: str, top_k: int = 15, alpha: float = 0.4):
        """
        Compute a combined KCS score as a weighted sum of hard and soft variants.
        alpha determines the weight of the hard score; (1-alpha) is weight for soft score.
        """
        hard = self.kcs_hard(article, summary, top_k)
        soft = self.kcs_soft(article, summary, top_k)
        return alpha * hard + (1 - alpha) * soft




    def identify_weakest(self, metric_values: dict):
        """
        Identify the weakest metric among all except 'Pairwise'.

        Useful for guiding mutations to improve the prompt on its weakest aspect.
        """
        comparable = {k: v for k, v in metric_values.items() if k != "Pairwise"}
        return min(comparable, key=comparable.get)
