

class StrategyACO:

    def __init__(self):
        pass

    @staticmethod
    def update_pheromone_AS(dag,pheromones, paths):
        """
        Classical Ant System: Increase pheromones of all the paths crossed
        """
        for path, fitness in paths:
            #Update of the first edge: START --> first level
            pheromones[(0, dag.start_node, path[0])] += 0.1 * fitness

            #Update the pheromones on the edges (first edge START --> first level is not in path)
            for i in range(len(path)-1):
                key = (i, path[i], path[i+1])
                pheromones[key] += fitness * 0.3

    def update_pheromone_EAS(self, dag, pheromones, paths, elitist_factor=0.3):
        """
        Elitist Ant System: AS classic,but the best path in the iteration have an extra increments of pheromones.
        """
        #Classic update
        self.update_pheromone_AS(dag,pheromones, paths)

        #best path of the iteration
        best_path_global, fitness = max(paths, key=lambda x: x[1])

        #Update of the first edge: START --> first level
        pheromones[(0, dag.start_node, best_path_global[0])] += elitist_factor * 0.1 * fitness

        #Update the pheromones on the edges (first edge START --> first level is not in path)
        for i in range(len(best_path_global)-1):
            key = (i, best_path_global[i], best_path_global[i+1])
            pheromones[key] += elitist_factor * fitness * 0.3

    @staticmethod
    def update_pheromone_Rank(dag,pheromones, paths, Q=0.3, rank_k=3):
        """
        Rank-Based Ant System: update only the top K paths, higher is the position in the top higher is the increment of pheromones
        """

        #sort the paths by the fitness value, higher fitness = better path
        sorted_paths = sorted(paths, key=lambda x: x[1], reverse=True) 

        #Iterate over the top K paths
        for rank, (path, fitness) in enumerate(sorted_paths[:rank_k]):
            weight = rank_k - rank  # higher is the rank higher is the weight factor

            #Update of the first edge: START --> first level
            pheromones[(0, dag.start_node, path[0])] += Q * 0.1 * fitness * weight

            #Update the pheromones on the edges (first edge START --> first level is not in path)
            for i in range(len(path)-1):
                key = (i, path[i], path[i+1])
                pheromones[key] += Q * fitness * weight * 0.3

    @staticmethod
    def update_pheromone_MMAS(dag,pheromones, paths, Q=0.3, tau_min=0.01, tau_max=0.5):
        """
        Max-Min Ant System: only the best path increase its pheromones, with upper and lower bound
        """
        #best path
        best_path, best_fitness = max(paths, key=lambda x: x[1])

        #Update of the first edge: START --> first level
        pheromones[(0, dag.start_node, best_path[0])] += Q * best_fitness * 0.1
        pheromones[(0, dag.start_node, best_path[0])] = max(tau_min, min(pheromones[(0, dag.start_node, best_path[0])], tau_max))

        #Update the pheromones on the edges (first edge START --> first level is not in path)
        for i in range(len(best_path)-1):
            key = (i, best_path[i], best_path[i+1])
            pheromones[key] += Q * best_fitness * 0.3
            pheromones[key] = max(tau_min, min(pheromones[key], tau_max))

    @staticmethod
    def update_pheromone_BWAS(dag,pheromones, paths, Q=0.3):
        """
        Best-Worst Ant System: increase pheromones on the best path and deacrease them in the worst paths.
        """

        best_path, best_fitness = max(paths, key=lambda x: x[1]) #best paths
        worst_path, worst_fitness = min(paths, key=lambda x: x[1]) #worst paths

        #Update of the first edge: START --> first level
        pheromones[(0, dag.start_node, best_path[0])] += Q * best_fitness * 0.1

        #Update the pheromones on the edges of the best path (first edge START --> first level is not in path)
        for i in range(len(best_path)-1):
            key = (i, best_path[i], best_path[i+1])
            pheromones[key] += Q * best_fitness *0.3

        #Update of the first edge: START --> first level
        pheromones[(0, dag.start_node, worst_path[0])] -= 0.1 * Q * worst_fitness
        pheromones[(0, dag.start_node, worst_path[0])] = max(0.01, pheromones[(0, dag.start_node, worst_path[0])])

        #Update the pheromones on the edges of the worst path (first edge START --> first level is not in path)
        for i in range(len(worst_path)-1):
            key = (i, worst_path[i], worst_path[i+1])
            pheromones[key] -= 0.1 * Q * worst_fitness *0.3
            pheromones[key] = max(0.01, pheromones[key])
