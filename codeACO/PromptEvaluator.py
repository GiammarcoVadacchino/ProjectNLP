import torch
import nltk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel
)
from tqdm import tqdm
from rouge_score import rouge_scorer
import numpy as np
import csv

#used for splitting the text in word or phrase in an automatic way, used in the computation of faithfulness
nltk.download("punkt")


class PromptEvaluator:

    def __init__(self, model, tokenizer, dataset, device="mps"):
       
        self.model = model.to(device) # model used to do the summarization task
        self.tokenizer = tokenizer # tokenizer for the model
        self.dataset = dataset # train dataset used in the iterations (is always the same for each iteration and ant)
        self.device = device # mps

        
        #the faithfulness measure how the summary generated respect the input text
        #measure if the model invents or not new informations that are not in the input text
        #It capture hallucinations of the model
        self.nli_tokenizer = AutoTokenizer.from_pretrained("valhalla/distilbart-mnli-12-1") 
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("valhalla/distilbart-mnli-12-1").to(device) #model used to calculate the faithfulness beetween the summary generated and the input text

        
        #the relevence measure how much the generated summary is relevent respect of the input text
        #it capture if the summary captures the principle information of the input text and drops meanless information
        #the aim is to understand if the model generates a summary usefull and informative, without distraction or irrilevant informations
        self.emb_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
        self.emb_model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2").to(device) #embedding model used to calculate the relevance beetween the summary generated and the input text




    def evaluatePrompt(
            self, 
            prompt, #prompt ready for evaluation 
            humanPrompt = False, 
            humanPromptcsvPath = ""
        ):

        fitness_scores = [] #collect the avg fitness of the prompt (of an ant) in each iteration
        faithfulness_scores = [] #collect the avg faithfulness of the prompt (of an ant) in each iteration
        relevance_scores = [] #collect the avg relevence of the prompt (of an ant) in each iteration
        rogue_scores = [] #collect the avg rogue of the prompt (of an ant) in each iteration

        if not humanPrompt:
            print("=== Starting ACO prompt evaluation ===")
        else:
            print("=== Starting HUMAN prompt evaluation ===")
            print(f"HUMAN PROMPT: \n {prompt}")
        #iterate over the training dataset
        for i, doc in enumerate(tqdm(self.dataset, desc="Evaluating documents")):

            #trasform a path (a series of nodes where each nodes is a part of the prompt) in a textual prompt ready to tokenize
            textual_prompt = self.serialize(prompt, doc[0]) if not humanPrompt else prompt.replace('{INPUT}', doc[0])

            print(f"\nDocument {i+1}/{len(self.dataset)}")
            print(f"Input text:\n{doc[0]}")

            #tokenize the textual_prompts
            inputs = self.tokenizer(
                textual_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)

            #generate the summary
            with torch.no_grad():
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample = True,
                    temperature = 0.3,
                    top_p=0.9
                )

            #given the summary generated, decodes to have words and not tokens
            summary = self.tokenizer.decode(
                gen_ids[0],
                skip_special_tokens=True
            )

            print(f"Ground Truth Summary:\n {doc[1]}")
            print(f"Generated Summary:\n {summary}")

            
            faithfulness = self.faithfulness_score(doc[0], summary) #given input text and summary generated calculates the faithfulness
            relevance = self.relevance_score(doc[1], summary) #given input text and summary generated calculates the relevance
            rouge = self.rogueL_score(doc[1],summary) #given the GT summary and summary generated calculates the rouge
            print(f"Metrics - Faithfulness: {faithfulness:.3f}, Relevance: {relevance:.3f}, Rougue: {rouge:.3f}")

            faithfulness_scores.append(faithfulness)
            relevance_scores.append(relevance)
            rogue_scores.append(rouge)

            # given the metrics calculates the fitness value for a single document
            fitness_scores.append(self.fitness_function(faithfulness, relevance, rouge)) 
            print(f"Calculated fitness: {fitness_scores[-1]:.3f}")

        #save result for human prompt, need this for comparison with ACO approach
        if humanPrompt:
            with open(humanPromptcsvPath, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    prompt.replace("\n", " | "),
                    f"{np.mean(faithfulness_scores):.3f} ± {np.std(faithfulness_scores):.3f}",
                    f"{np.mean(relevance_scores):.3f} ± {np.std(relevance_scores):.3f}",
                    f"{np.mean(rogue_scores):.3f} ± {np.std(rogue_scores):.3f}",
                ])
            return
        
        return fitness_scores,faithfulness_scores,relevance_scores,rogue_scores





    def serialize(
            self, 
            path, #path that correspond to a valid prompt
            input_text #source document
        ):
        parts = []

        #given a path of the DAG iterate on it
        for node in path:
            #if the node text contains the {INPUT} placeholder means that the level of the node is input_level and so the input text is injected into and replace the placeholder
            if "{INPUT}" in node.text:
                parts.append(node.text.replace("{INPUT}", input_text))
            #otherwise append the normal text
            else:
                parts.append(node.text)

        return "\n".join(parts) # return the textual prompt as a string



    def faithfulness_score(
            self, 
            document: str, #source document
            summary: str #summary generated by the LLM
        ):

        #split the generated summary using the "punkt" model into prhases
        #ignores the phrases that have a lenght less than 4, beacuse problably they contains less informations and are more meaningless
        sentences = [
            s for s in nltk.sent_tokenize(summary)
            if len(s.strip().split()) > 4
        ]

        #if no sentence return
        if len(sentences) == 0:
            return 0.0

        scores = []

        #iterate over the sentences of the generated summary
        for sent in sentences:
            #prepare the inputs for the model (document, sent), in this way the model can "tell" if the sent is implicated in the document
            inputs = self.nli_tokenizer(
                document,
                sent,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                #using a NLI model that returns a value for each possible label [contradiction, neutral, entailment])
                logits = self.nli_model(**inputs).logits
                #softmax transforms the logits in probabilty value beetween 0 and 1, the sums is 1, contains three values, one for each label
                probs = torch.softmax(logits, dim=-1)[0]


            #NOTE: neutral label is not taken in consideration

            #take the value for the entailment label 
            entail = probs[2].item()
            #take the value for the contradiction label 
            contra = probs[0].item()

            #penalize if the sentence is contraddictory respect to the given document
            score = max(entail - contra, 0.0)
            scores.append(score)

        return sum(scores) / len(scores) #return the mean


    def embed(
            self,
            text # text to be embedded
        ):

        #tokenize the given text
        inputs = self.emb_tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            #generate the embedding
            outputs = self.emb_model(**inputs)
            #avg of all tokens
            emb = outputs.last_hidden_state.mean(dim=1)

        return emb


    def relevance_score(
            self, 
            summary_gt: str, #gt summary
            summary: str #summary generated by the LLM
        ):

        #get embeddings
        doc_emb = self.embed(summary_gt)
        sum_emb = self.embed(summary)

        #compute cosine similarity beetween the two embeddings, higher value means that the summary is more relevant to the document
        cosine = torch.nn.functional.cosine_similarity(doc_emb, sum_emb)
        return max(0.0, float(cosine))
    


    def rogueL_score(
            self, 
            summary_gt: str, #gt summary
            summary: str #summary generated by the LLM
        ):


        #prepare the computation of rouge1,rouge2,rougeL
        #rouge-1: overlaps of unigram (singles characters)
        #rouge-2: overlaps of bigram
        #rouge-L: longest commong subsequences
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True # reduce the words into their radix (running --> run) for better comparation
        )

        #compute the scores, returns a dictionary with the metrics, for each metrics we have precision recall and F1
        scores = scorer.score(summary_gt, summary)

        #take the F1 value for each metrics
        rouge1 = scores["rouge1"].fmeasure
        rouge2 = scores["rouge2"].fmeasure
        rougeL = scores["rougeL"].fmeasure

        #compute a weighted sum with the metrics values
        score = (
            0.3 * rouge1 +
            0.4 * rougeL +
            0.3 * rouge2
        )

        return score


    def fitness_function(
            self, 
            faithfulness, #faith score
            relevance, #rel score
            rogue #rouge score
        ):

        #given the three metrics calculate a fitness value, in this case a weighted sum of the metrics
        
        return (
            0.5 * faithfulness +
            0.2 * relevance +
            0.3 * rogue
        )
