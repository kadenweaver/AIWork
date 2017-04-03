# Author: Kaden Weaver
# Purpose: evolve an N-queens solution.
# Citations:


from queens import QueenState
import random
import copy

# A state in an N-queens puzzle environment.
# Subclass of the one we wrote in class.
class QueenStateIndividual(QueenState):

    #***************************************************
    # Return the number of non-conflicting queen pairs.
    def fitness(self):
        conflicts = self.conflicts()
        maxcon = (self.n * (self.n-1))/2
        return (maxcon - conflicts)

    #****************************************************
    # Return a new individual that is a combination of self and other.
    def crossover(self, other):
        babystate = copy.deepcopy(self)
        randomdiv = random.randint(0,self.n-1)
        locations = babystate.locations
        while randomdiv < self.n:
            locations[randomdiv] = other.locations[randomdiv]
            randomdiv+=1
        return babystate


    #******************************************************
    # Give each location a small probability of getting randomly changed.
    def mutate(self):
        i = 0
        while i < self.n:
            chance = random.randint(0,99)
            if chance == 50 or chance == 51:
                self.locations[i] = (random.choice(range(n)), random.choice(range(n)))
            i += 1




# A collection of states in a genetic algorithm.
class Population(object):

    # Begin with an empty list of states.
    def __init__(self):
        self.states = list()
        self.total_fitness = 0

    # Add a state to this population, remembering its fitness.
    # Keep a running sum of fitness values for the entire population.
    def add(self, state):
        fitness = state.fitness()
        self.total_fitness += fitness
        self.states.append((fitness, state))

    # Return the state with the highest fitness.
    def best(self):
        (fitness, state) = max(self.states)
        return state

    # ******************************************************
    # Return a randomly chosen state.
    # Probability of being selected is proportional to fitness.
    def select(self):
        pick = random.random() * self.total_fitness
        count = 0
        for (fitness,state) in self.states:
            if count < pick < (count + fitness):
                return state
            else:
                count += fitness

    # ******************************************************
    # Return another population of the same size as this one.
    # Use elitism: include the best state from this population.
    # Create the rest by selecting parents and doing crossover and mutation.
    def nextgen(self):
        capacity = len(self.states)-1
        newgen = Population()
        beststate = self.best()
        newgen.add(beststate)
        i = 0
        while i < capacity:
            p1 = self.select()
            p2 = self.select()
            baby = p1.crossover(p2)
            baby.mutate()
            newgen.add(baby)
            i += 1
        return newgen



# Try to evolve a solution to the N-queens puzzle.
if __name__ == '__main__':

    n = 8 # The puzzle size
    pop = 100 # The population size

    # Initial population of random states
    population = Population()
    for count in range(pop):
        population.add(QueenStateIndividual(n))

    best = population.best()
    print(best.fitness(), "after 0 generations")

    # Evolve until we get a goal state
    generation = 0
    while best.fitness() < n*(n-1)/2:
        generation += 1

        population = population.nextgen()

        # Watch when improvements occur
        if population.best().fitness() > best.fitness():
            best = population.best()

            print(best.fitness(), "after", generation, "generations")

    # See the final solution
    best.display()
    print(best.conflicts())
