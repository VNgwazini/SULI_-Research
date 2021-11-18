#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:35:01 2019

@author: 1vn
"""
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
##--------------------------------------------------------CITY CLASS-----------------------------------------------------##
#create class for distance calculations
class City:
    #initialize future class objects with a constructor
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    #create a printable string like representation of the object
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"
##-----------------------------------------------------FITNESS CLASS-----------------------------------------------------------##
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                #make toCity hold the route to a city
                if i + 1 < len(self.route):
                    #hold route t next city if there is one next
                    toCity = self.route[i + 1]
                else:
                    #hold route to first city if there is no next city
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness
##------------------------------------------------------GLOBAL FUNCTIONS----------------------------------------------------##

#create a route
def createRoute(cityList):
    #select a random city from a list of cities and return a route
    route = random.sample(cityList, len(cityList))
    return route


#create an ordered list of routes
def initialPopulationOfRoutes(populationSize, cityList):
    populationOfRoutes = []
    for i in range(0,populationSize):
        #add each created routes to the list one by one
        populationOfRoutes.append(createRoute(cityList))
    return populationOfRoutes
    
##------------------------------------------------------FITNESS----------------------------------------------------##
# rank the routes
def rankRoutes(populationOfRoutes):
    #create dictionary to hold fitness results
    fitnessResults = {}
    for i in range(0,len(populationOfRoutes)):
        #store the fitness calculations "objects" in dictionary
        fitnessResults[i] = Fitness(populationOfRoutes[i]).routeFitness()
        
        #sort dictionary where highest fitness score is first
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse =True)

##------------------------------------------------------SELECTION----------------------------------------------------##
#select highest ranked populations
def selection(rankedPopulation, eliteSize):
    selectionResults = []
    
    #create a data frame of the ranked population with index and fitness columns
    df = pd.DataFrame(np.array(rankedPopulation), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    
    #store each rank in the dataframe that holds selectionResults's data        
    for i in range(0, eliteSize):
        selectionResults.append(rankedPopulation[i][0])
    for i in range(0, len(rankedPopulation) -eliteSize):
        pick = 100*random.random()
        for i in range(0, len(rankedPopulation)):
            if pick <= df.iat[i,3]:
                selectionResults.append(rankedPopulation[i][0])
                break
    #print('selection results: ', selectionResults)
    #print('selection legnth: ', len(selectionResults))
    return selectionResults
        
##------------------------------------------------------MATING----------------------------------------------------##        
#create a mating pool of routes to mate
def matingPool(populationOfMates, selectionResults):
    poolOfParentRoutes = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        poolOfParentRoutes.append(populationOfMates[index])
    #print('pool of parent routes: ', poolOfParentRoutes)
    return poolOfParentRoutes


#breed two routes and return a new one 
def breed(parent1,parent2):
    #create odered lists to hold child route and parent genes(cities)
    ChildRoute = []
    genesFrom_P1 = []
    genesFrom_P2 = []
    
    #each gene is really just a city --> its corresponding point on the grid
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    #staring city
    startGene = min(geneA, geneB)
    #ending city
    endGene = max(geneA, geneB)
    
    #store parent1's genes in genesFrom_P1 list
    for i in range(startGene, endGene):
        genesFrom_P1.append(parent1[i])
        
     #store parent2's genes in genesFrom_P2 list that parent1 does not also have   
    genesFrom_P2 = [item for item in parent2 if item not in genesFrom_P1]
        
    ChildRoute = genesFrom_P1 + genesFrom_P2
    return ChildRoute


#breedPopulation
def breedPopulation(poolOfMates, eliteSize):
    
    #create ordered list of childrens genes(cities) 
    ChildrenRoutes = []
    length = len(poolOfMates) - eliteSize
    parentPool = random.sample(poolOfMates, len(poolOfMates))
    
    #append each pair of mates from the elite sample to the Children Routes Pop
    for i in range(0,eliteSize):
        ChildrenRoutes.append(poolOfMates[i])
        
    #mate two parents playoff style and append their child to the Children Population
    for i in range(0,length):
        ChildRoute = breed(parentPool[i], parentPool[len(poolOfMates) -i -1])
        ChildrenRoutes.append(ChildRoute)
        
    return ChildrenRoutes 
##------------------------------------------------------MUTATION OR CROSSOVER----------------------------------------------------##    
#mutate a single route
def mutate(route, mutationRate):
    for swapGene in range(len(route)):
        #when the random percentage is less than the muation percentage
        if(random.random() < mutationRate):
            #randomly generate a float bigger than 0 and less than 1
            #then multiply to make the potential numbers range from 1 to length of route
            #use this value to pick a gene to swap in  the given route
            swapWith = int(random.random() * len(route))
            
            #city1 = route[a city/gene that will be swapped]
            city1 = route[swapGene]
            #city2 = route[the city/gene that it will be swapped with]
            city2 = route[swapWith]
            
            #swap city 1 with city 2
            route[swapGene] = city2
            route[swapWith] = city1
            
    return route
        
def mutatePopulation(routes, mutationRate):
    #create an ordered list of mutated routes
    mutatedRoutes = []
    
    #mutate each route and append it to the list of routes
    for route in range(0, len(routes)):
        mutatedRoute = mutate(routes[route], mutationRate)
        mutatedRoutes.append(mutatedRoute)
        
    return mutatedRoutes

##------------------------------------------------------NEXT GENERATION----------------------------------------------------##            
#generate next generation of routes
def nextGeneration(currentGen, eliteSize,mutationRate):
    rankedPopulation = rankRoutes(currentGen)
    selectionResults = selection(rankedPopulation, eliteSize)
    potentialMates =  matingPool(currentGen, selectionResults)
    childrenRoutes = breedPopulation(potentialMates, eliteSize)
    nextGeneration = mutatePopulation(childrenRoutes, mutationRate)
    #print('selection results: ', selectionResults)
    return nextGeneration

##------------------------------------------------------GENETIC ALGORITHM----------------------------------------------------##            

def geneticAlgorithm(routes,popSize,eliteSize,mutationRate,numOfGeneration):
    #create an initial population of ranked routes and print the best
    pop = initialPopulationOfRoutes(popSize, routes)
    print("Initial Distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    #run the evolution process 
    for i in range(0, numOfGeneration):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    #store the index of the best route in the pop and return it    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute




cityList = []

for i in range(0,25):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))
    plt.scatter(cityList[i].x, cityList[i].y)
    plt.ylabel('Distance')
    plt.xlabel('Generation')

plt.show()
geneticAlgorithm(routes=cityList, popSize=100, eliteSize=20, mutationRate=0.01, numOfGeneration=500)


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, numOfGeneration):
    pop = initialPopulationOfRoutes(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, numOfGeneration):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, numOfGeneration=500)
