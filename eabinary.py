import random as rand;
import math as math;
import numpy as np;
#hyperparameters of EA

POPULATION_SIZE = 50;
SELECTION_COUNT = 5;
MUTATION_PROBABILITY = 0.6;
FITNESS_EVALUATOR_GENERATION_COUNT = 5;
WEIGHT_VECTOR_DIMENSION = 24


class Gene:

    def __init__(self, id, weightVector = np.array([]), fitness = math.inf): #Intialize the object and also give it a rating -

        self.id = id;

        #These if else block is just to make sure that when creating multiple instances of the object
        #they won't end up having the same random values

        if(weightVector.size==0):
            weightVector = np.random.randint(2,size = WEIGHT_VECTOR_DIMENSION,dtype=np.uint8); #initializing with 0's and 1's; type explicit specifying for binary conversion
            weightVector = np.reshape(weightVector,(WEIGHT_VECTOR_DIMENSION,1));#making the weightVector a column vector
            self.weightVector = np.unpackbits(weightVector,axis = 1) #the weight vectors are binary

        else:
           self.weightVector = weightVector;

        self.fitness = fitness;

    def getFitness(self):
        #TODO call neural network and simulation thingy here


    def crossOver(self, p2):  # additive cross over

        """
        This function perform binary cross over.
        :param self: represents the object calling this function or simply the first parent
        p2: represents the second parent.
        :return: The function returns two child after crossing the binary weight vectors based on a crossoverpoint
        """
        p1 = self;

        c1WeightVector = np.array([], dtype=np.uint8);
        c2WeightVector = np.array([], dtype=np.uint8);


        for w1,w2 in zip(p1.weightVector,p2.weightVector):
            crossOverPoint = rand.randrange(-1,7) #since the binary is 8 bits when -1 there will be no cross over
            cw1 = np.concatenate((w1[0:crossOverPoint+1],w2[crossOverPoint+1,8]));
            cw2 = np.concatenate((w2[0:crossOverPoint+1],w1[crossOverPoint+1,8]));
            c1WeightVector = np.append(c1WeightVector, cw1);
            c2WeightVector = np.append(c2WeightVector, cw2);

        c1WeightVector = np.reshape(c1WeightVector, (WEIGHT_VECTOR_DIMENSION, 8));#8bits bcoz representation after unpacking
        c2WeightVector = np.reshape(c2WeightVector, (WEIGHT_VECTOR_DIMENSION, 8));

        return Gene(0, c1WeightVector), Gene(0,c2WeightVector); #return two children or offsprings

    def mutate(self):
        """
        This method select a random bit within the binary weight vector and complement it
        :return: mutated gene
        """
        mutatedWeightVector = np.array([], dtype=np.uint8);
        for w in self.weightVector:
            index = rand.randint(0,7)
            w[index] = 0 if w[index]==1 else 0;#to perform complement
            mutatedWeightVector = np.append(w);
        return Gene(0, mutatedWeightVector);  # perforing random mutation

class Generation:

    def __init__(self, population_size, selectionCount):
        self.population_size = population_size;
        self.selectionCount = selectionCount;
        self.genes = [Gene(i) for i in range(population_size)]; #population

    def evaluation(self):
        for gene in self.genes:
            gene.fitness = gene.getFitness();

    def selection(self):
        self.genes.sort(key=lambda node: node.fitness);#TODO we have to experiment
        self.genes = self.genes[0:self.selectionCount-1];

    def reproduction(self): #cross over is doing here
            for i in range(self.population_size):
                i = rand.randint(0, self.genes.__len__() - 1);
                j = rand.randint(0, self.genes.__len__() - 1)
                self.genes.append(self.genes[i].crossOver(self.genes[j]));


    def mutation(self, prob):
        p = rand.random();
        if(p<=prob): #mutation
            indeces = rand.sample(range(0,self.population_size),rand.randint(0,self.population_size-1));
            for index in indeces:
                self.genes[index]=self.genes[index].mutate();


def main():

    flag = 0;
    iter = 0;
    bestGene = Gene(0);

    g = Generation(POPULATION_SIZE,SELECTION_COUNT);
    """
    while True:
        temp = bestGene;

        g.evaluation();
        for gene in g.genes:
            if (gene.fitness < bestGene.fitness):  # keeping track of optimum
                flag = 0;
                bestGene = gene;
        g.selection();
        g.reproduction();
        g.mutation(MUTATION_PROBABILITY);

        if(temp.fitness==bestGene.fitness):
            flag +=1;

        if(flag==FITNESS_EVALUATOR_GENERATION_COUNT):
            break;

        iter += 1;

    #for gene in g.genes:
      #  print("X=%f,Y=%f"%(gene.x,gene.y));

    #print("Total no: generations = %d Optimum: X=%f, Y=%f"%(iter, bestGene.x,bestGene.y));
    """
    g.evaluation();
    g.selection();
    g.reproduction();
    g.mutation(MUTATION_PROBABILITY);

if __name__=="__main__":
    main();