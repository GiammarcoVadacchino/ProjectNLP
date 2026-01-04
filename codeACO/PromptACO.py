import random
import math
from collections import defaultdict
import csv
import os
import numpy as np


class PromptACO:
    def __init__(
        self,
        dag,
        evaluator,
        strategyACO,
        strategy_name,
        n_ants,
        iterations,
        alpha=1.0,
        evaporation_rate=0.2,
        initial_pheromone=0.01
    ):
        

        self.alpha = alpha #parameter of ACO, that is used for calculating the pheromones on the edges
        self.evaporation_rate = evaporation_rate #parameter for the evaporation of pheromone across iterations
        self.pheromones = defaultdict(lambda: initial_pheromone) #dict for the pheromones a key of the dict is (level_starting_node,starting_node,target_node), it contains the pheromones of the DAG
        self.dag = dag #input DAG
        self.n_ants = n_ants #number of ants that cross the DAG
        self.iterations = iterations #number of iterations of ACO
        self.history = [] #list that collects stats during iterations, used for save the stats in a csv file
        self.evaluator = evaluator #evaluator, used for evalutaing the summary generated for a given prompt 
        self.strategyACO = strategyACO #stategy used to upgrade the pheromones of the edges
        self.strategy_name = strategy_name #name of the strategy used

        #mapping the strategies names to the correspondent function
        self.STRATEGY_MAP = {
                "Basic ACO": strategyACO.update_pheromone_AS,
                "EAS": strategyACO. update_pheromone_EAS,
                "Rank":strategyACO. update_pheromone_Rank,
                "MMAS": strategyACO.update_pheromone_MMAS,
                "BWAS": strategyACO.update_pheromone_BWAS
            }
        
        #check if the given strategy name is in the map
        if strategy_name not in self.STRATEGY_MAP:
            raise ValueError(f"Strategy '{strategy_name}' not recognized. Available strategies: {list(self.STRATEGY_MAP.keys())}")
        
        print(f"I'M USING {self.strategy_name} STRATEGY\n")
    
        
        self.pheromone_update_func = self.STRATEGY_MAP[strategy_name] #function that apply the strategy choosed
        self.initialize_pheromones() #initalization of pheromones

    def initialize_pheromones(self):
        """
        Initialize pheromones for ALL edges in the DAG
        """
        for level_idx, level_nodes in enumerate(self.dag.level_nodes):
            if level_idx == 0:
                parents = [self.dag.start_node]
            else:
                parents = self.dag.level_nodes[level_idx - 1]

            for parent in parents:
                for child in level_nodes:
                    key = (level_idx - 1 if level_idx > 0 else 0, parent, child)
                    self.pheromones[key] = self.pheromones[key]  # force creation


    def select_node(self, level_idx, prev_node):
        #given the level of a node, take all the nodes in that level
        candidates = self.dag.level_nodes[level_idx]
        scores = []


        #iterate over the all possible nodes of a certain level
        for node in candidates: 
            if prev_node is self.dag.start_node:
                #calculate pheromones on the first level (so level 0), START --> first level
                tau = self.pheromones[(0, prev_node, node)]
            else:
                #calculate pheromones, if the target nodo is on a level i, it means that the edge goes to a prev_node in the i-1 level to the target node, so (i-1,prev_node,target_node)
                tau = self.pheromones[(level_idx - 1, prev_node, node)]

            #calculate scores
            scores.append(tau ** self.alpha)

        total = sum(scores)
        if total == 0:
            return random.choice(candidates)
        
        #calculate probability ditribution using the scores
        probs = [s / total for s in scores]

        #return the sampled node
        return random.choices(candidates, probs)[0]

    

    def sample_prompt(self):
        path = []

        #Cross the DAG levels 
        for level_idx, _ in enumerate(self.dag.level_nodes):
            if level_idx == 0:
                #select the next node of the path starting in the initial node of the DAG
                choice = self.select_node(level_idx, prev_node=self.dag.start_node)
            else:
                #otherwise pick the last crossed node, and selecting the node
                prev = path[-1]
                choice = self.select_node(level_idx, prev)

            #build the path
            path.append(choice)

        return path


    def evaporate(self):
        #Apply pheromons evaporation on the path crossed
        for k in self.pheromones:
            self.pheromones[k] *= (1 - self.evaporation_rate)

    def run(self):
        
        global_best_path = None #Prompt (so a path) with the highest fitness in the iteration
        global_best_fitness = -math.inf #best fitness value

        #Iterate over the iterations
        for iteration in range(self.iterations):

            paths = [] #Paths of the ants
            fitness_values = [] #fitness over iterations
            faithfulness_values = [] #faithfulness over iterations
            relevance_values = [] #relevance over iterations
            rouge_values = [] #rouge over iterations

            #Iterate over the ants
            for _ in range(self.n_ants):
                path_sampled = self.sample_prompt() #cross the DAG and build a path (so a list of the part of a prompt)
                print(f"Initial Prompt:\n {self.serialize(path_sampled)}")
                
                avg_fitness,avg_faith,avg_rel,avg_rouge = self.evaluator.evaluatePrompt(path_sampled) #evaluate the prompt

                paths.append((path_sampled,avg_fitness))
                fitness_values.append(avg_fitness)
                faithfulness_values.append(avg_faith)
                relevance_values.append(avg_rel)
                rouge_values.append(avg_rouge)

                #If we found a path with a better global avg fitness,then save the prompt and the its fitness value
                if avg_fitness > global_best_fitness:
                    global_best_fitness = avg_fitness
                    global_best_path = path_sampled

            #Logging stats over iterations
            best_path, best_fit = max(paths, key=lambda x: x[1])
            worst_path, worst_fit = min(paths, key=lambda x: x[1])
            mean_fit = sum(fitness_values) / len(fitness_values)


            #Add logging history
            self.history.append({
                "iteration": iteration,
                "levels_names": " ".join([level_name for level_name in self.dag.level_names]),
                "strategy_name": self.strategy_name,
                "n_ants": self.n_ants,
                "best_prompt": " ".join(" ".join(node.text.split()) for node in best_path),
                "best_fitness": f"{best_fit:.3f}",
                "worst_prompt": " ".join(" ".join(node.text.split()) for node in worst_path),
                "worst_fitness": f"{worst_fit:.3f}",
                "mean_fitness": f"{mean_fit:.3f}",
                "faith": f"{np.mean(faithfulness_values):.3f} ± {np.std(faithfulness_values):.3f}",
                "relevance": f"{np.mean(relevance_values):.3f} ± {np.std(relevance_values):.3f}",
                "rogue": f"{np.mean(rouge_values):.3f} ± {np.std(rouge_values):.3f}"
            })

            #Evaporation of the pheromones after each iteration
            self.evaporate()

            #Update pheromones after each iteration
            self.pheromone_update_func(self.dag,self.pheromones,paths)
            
            #if the value is lower it means that a specific node of that level has more pheromones on it and so when it is crossed the prompt produce better outputs
            #if the value is higher, it means that the pheromones are equally distributed among the nodes of a certain level, so it means that all have the same importance in the prompt
            self.compute_level_entropies() #compute the entropy after each iteration, this tell how much a level is important in the DAG

        return global_best_path, global_best_fitness


        
    def serialize(self, path):
        parts = []

        #Iterate over a path
        for node in path:
            #collect texts of the node crossed
            parts.append(node.text)

        return "\n".join(parts) #return a textual prompt as a string
    


    def compute_level_entropies(self):
 
    
        level_entropies = {} # map that contains the entropy for each level of the DAG
        prev_nodes = [self.dag.start_node]  # initial node START

        #Iterate over the DAG
        for level_idx, level_nodes in enumerate(self.dag.level_nodes):
            # accumalate the pheromones of a certain level
            all_pheromones = []

            #iterate the nodes where the edge start
            for parent in prev_nodes:
                #iterate the node where the edge end
                for child in level_nodes:
                    #build a key for the edge,follows the pattern of the dict for pheromones
                    key = (level_idx - 1, parent, child) if parent is not self.dag.start_node else (0, self.dag.start_node, child)
                    #get the pheromones value from the map
                    all_pheromones.append(self.pheromones.get(key, 0.01))

            if not all_pheromones:
                all_pheromones = [0.01] * len(level_nodes)

            pheromones = np.array(all_pheromones)
            prob = pheromones / pheromones.sum() # normalize the pheromones values into probabilities

            
            H = -np.sum(prob * np.log(prob + 1e-12)) #calculate the entropy
            H_max = np.log(len(prob)) #calculate the max entropy for a level 
            H_norm = H / H_max #normalize the entropy

            #save the entropy for a certain level
            level_name = self.dag.level_names[level_idx] if hasattr(self, "dag_levels_names") else f"Level_{level_idx}"
            level_entropies[level_name] = H_norm

            #update the previous nodes, moves on the DAG into the right
            prev_nodes = level_nodes

        print("\n--- Normalized Entropy for Levels ---")
        for lvl, H in level_entropies.items():
            print(f"{lvl}: {H:.3f}")
        print("----------------------------------------\n")


        return level_entropies
    




    def save_history_to_csv(self, file_path):

        #if no history logged
        if not self.history:
            raise ValueError("La history è vuota, niente da salvare.")

        #Create the dir if no exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        #Get the coloumns of the csv
        fieldnames = self.history[0].keys()

        with open(file_path, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerows(self.history)

