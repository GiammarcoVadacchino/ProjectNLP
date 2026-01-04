import pandas as pd
import random
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from PromptNode import PromptNode
import numpy as np

DEVICE = "mps"



class PromptDAG:

    def __init__(self, level_names: List[str]):

        self.level_names = level_names # label of the components of a summarization prompt
        self.n_levels = len(level_names) # number of levels of the DAG
        self.start_node = PromptNode("START", optional=False) # start node, each ant starts in this node
        self.level_nodes: List[List[PromptNode]] = [] # adjaceny matrix used for graph rapresentation 

    def initialize_levels(self, base_texts: List[str]):
        
        #base_texts: one base phrase per level (same order as level_names)
        assert len(base_texts) == self.n_levels 

        self.level_nodes = []

        # Initialize the first node for each level of the DAG
        for level_name, text in zip(self.level_names, base_texts):
            mandatory = level_name in ["TaskInstruction", "Input"]  # mandatory levels

            # Add placeholder only for Input level
            if level_name == "Input":
                text = text + "\n{INPUT}"

            #create Node
            node = PromptNode(
                name=f"{level_name}_0",
                text=text,
                optional=not mandatory
            )

            self.level_nodes.append([node])



    def build_graph(self):

        #Connecting the start node with the first layer of the DAG
        for node in self.level_nodes[0]:
            self.start_node.add_child(node)

        #Connecting the internal layers of the DAG
        for i in range(len(self.level_nodes) - 1):
            for parent in self.level_nodes[i]:
                for child in self.level_nodes[i + 1]:
                    parent.add_child(child)





    def generate_variants_for_initialized_levels(
        self,
        path_csv: str,
        nodes_per_level: List[int],
        model,
        tokenizer,
        max_length: int = 64
    ) -> Dict[str, List[str]]:

        #the length of the list that contains the number of nodes for each level has to be equal the number of the nodes
        assert len(nodes_per_level) == self.n_levels

        all_variants: Dict[str, List[str]] = {} #contains variants of inizializeds base texts

        for level_name, nodes, n_nodes in zip(
            self.level_names, self.level_nodes, nodes_per_level
        ):
            base_node = nodes[0] #first node of the first level
            variants = [base_node.text] #take always the base text

            n_variants_to_generate = 3 * n_nodes  # number of variants that has to be generated

            for _ in range(n_variants_to_generate - 1):
                #prompt that use the model for generating variants
                prompt = (
                    "Rewrite the following prompt using different wording but "
                    "keeping exactly the same task, constraints, and meaning:\n"
                    f"{base_node.text}"
                )

                #tokenize the input
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
                #generate the pharaphrase
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.8
                )

                paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)
                variants.append(paraphrase)

            all_variants[level_name] = variants

        #save variants for each inizialized levels in a csv
        df = pd.DataFrame(all_variants)
        df.to_csv(path_csv, index=False)

        return all_variants





    def load_variants_from_csv_for_initialized_levels(
        self,
        path_csv: str
    ) -> Dict[str, List[str]]:


        df = pd.read_csv(path_csv)

        all_variants: Dict[str, List[str]] = {}

        # Iterate over level names and load available columns
        for level_name in self.level_names:
            if level_name not in df.columns:
                continue

            texts = df[level_name].dropna().tolist()
            all_variants[level_name] = texts

        return all_variants




    def prepare_variants(
        self,
        load_variants: bool,
        path_csv: str,
        nodes_per_level: List[int],
        model=None,
        tokenizer=None,
        diversity_model_name: str = "all-MiniLM-L6-v2",
        diversity_threshold: float = 0.85
    ):


        #Load or generate tha varints for each level
        if load_variants:
            all_variants = self.load_variants_from_csv_for_initialized_levels(path_csv)
        else:
            all_variants = self.generate_variants_for_initialized_levels(
                path_csv, nodes_per_level, model, tokenizer
            )

        #embedding model used to calculate the cosine similarity for all the variants of a certain level
        #this is done because we want to maintain a certain degree of diversity (more exploration) into the DAG
        embedder = SentenceTransformer(diversity_model_name)

        #contains filtered variants, so the variants that have a certain degree of diversity from the others
        filtered_variants: Dict[str, List[str]] = {}

        for level_idx, level_name in enumerate(self.level_names):

            #Only initialized levels
            if level_idx >= len(self.level_nodes):
                continue
            if len(self.level_nodes[level_idx]) == 0:
                continue
            if level_name not in all_variants:
                continue

            texts = all_variants[level_name]
            if len(texts) == 0:
                continue

            #calculate embeddings of each variant of a certain level
            embeddings = embedder.encode(texts, convert_to_numpy=True)

            #Get the top-K different variants
            kept_texts = self.check_diversity(texts, embeddings, diversity_threshold)

            #get the number of required nodes for a certain level
            required_nodes = nodes_per_level[level_idx]

            print(f"[{level_name}] Selected {len(kept_texts)}/{len(texts)} top diverse variants (required: {required_nodes})")

            #Fallback: if the number of top-K variants are lower than the number of required_nodes, then pick randomly other variants
            # in order to have a number of node in that level equal to the required number of nodes.
            if len(kept_texts) < required_nodes:
                #list of variants that aren't in top K
                remaining_texts = [t for t in texts if t not in kept_texts]
                #iterate over the remaining texts
                while len(kept_texts) < required_nodes and remaining_texts:
                    #pick randomly
                    chosen = random.choice(remaining_texts)
                    kept_texts.append(chosen)
                    remaining_texts.remove(chosen)

            #Fallback: if the top-K variants are more than the required nodes, keep the bests of the top
            if len(kept_texts) > required_nodes:
                kept_texts = kept_texts[:required_nodes]

            filtered_variants[level_name] = kept_texts
            print(f"[{level_name}] Selected {len(kept_texts)}/{len(texts)} top diverse variants AFTER FALLBACK (required: {required_nodes})")

        #update the DAG with the variants
        self.update_dag(filtered_variants)



    
    def check_diversity(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        threshold: float
    ) -> List[str]:


        base_emb = embeddings[0] #embedding of the base text
        kept = []

        #iterate over the embeddings varinats
        for i, emb in enumerate(embeddings):
            #cosine similarity respect to the base text and the variant #NOTE: maybe use the torch function
            numerator = np.dot(emb, base_emb)
            denominator = np.linalg.norm(emb) * np.linalg.norm(base_emb)
            sim_to_base = 0.0 if denominator == 0 else numerator / denominator
            #print(sim_to_base)

            #if is the first embedding or respect the threshold
            if i == 0 or sim_to_base < threshold:
                kept.append((texts[i], sim_to_base))

        #Sort in respect to the similarity
        kept.sort(key=lambda x: x[1])

        #returns only the text 
        return [t for t, _ in kept]




    #Print the strucutre of the DAG, used for debugging
    def print_graph(self):
        print("\nPrompt DAG (Sequential):\n")
        print("START")
        for i, nodes in enumerate(self.level_nodes):
            print(f"\nLevel {i} — {self.level_names[i]}")
            for node in nodes:
                children = [c.text for c in node.children]
                print(f"  {node.text} -> {children}")
        print("\nEND\n")


    def update_dag(self, variants: Dict[str, List[str]]):

        #iterate over the DAG
        for level_idx, level_name in enumerate(self.level_names):
            #skip the levels that are not initialized
            if level_name not in variants:
                continue

            #base node for each level of the DAG
            base_node = self.level_nodes[level_idx][0]
            #Add the base node to the corresponding level
            self.level_nodes[level_idx] = [base_node]

            #iterate over the variants of a certain level (skipping the first variant, the first variant is the base text of the level)
            for i, text in enumerate(variants[level_name][1:], start=1):

                #Add the placeholder for the nodes that are in the input level
                if level_name == "Input" and "{INPUT}" not in text:
                    text = text + "\n{INPUT}"

                #add levels to the graph
                self.level_nodes[level_idx].append(
                    PromptNode(
                        name=f"{level_name}_{i}",
                        text=text,
                        optional=base_node.optional
                    )
                )
