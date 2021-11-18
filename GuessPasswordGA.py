#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:25:40 2019

@author: 1vn
"""
#-------------------------------LIBRARIES AND MODULES------------------------------------------------------------------------ 
    
import random
import operator
import time
import matplotlib.pyplot as plt

#child = "banano"

temps1 = time.time()

#-------------------------------INITAL CONDITONS----------------------------------------------------------------------------- 
#print("Hello World")    
#define initial test conditons
password = "banana"
populationSize = 100
bestSample = 20
luckyFew = 20
childrenPerCouple = 5
generationCountLimit = 50
mutationProbability = 5

#------------------------------------------------------------------------------------------------------------------------------ 
    

#geneate a word long enough to test------------------
def generateATestChild(length):
    #grow word length by 1 letter until length of new word is same as password
    geneIndex = 0
    testChild = ""
    #while the new word is shorter than password, add a random letter
    while ( geneIndex < length):
        #typecast ascII values to thier coresponding characters (i.e 97 = 'a')
        randomLetter = chr(97+ int(26 * random.random()))
        testChild += randomLetter
        geneIndex+=1
    #print("The Test Child Is: \n",testChild, "\n")
    return testChild


#--------------------------------INITIAL POPULATION----------------------------------------------------------------------------- 


#generate a population of potential solutions
def generatePopulation(populationSize,password):  
    #create population ordered list that is empty
    population = []
    i = 0  
    #fill populatiobn record with indivuduals the length of password until we 
    #reach desired population size
    while(i < populationSize):
        population.append(generateATestChild(len(password)))
        i+=1
    return population #this will be a whole population of desired size


#--------------------------------FITNESS FUNCTIONS------------------------------------------------------------------------------ 


#compute fitness one potential anwers
def computeChildFitness(password,testChild):
    score = 0
    child = 0
    #words get a point for every matching letter
    while(child < len(password)):
        if(testChild[child] == password[child]):
            score +=1
        child+=1
    return score * 100 / len(password)        


#----------------------------------------------------------------------------------------------------------------------------------- 


#find the best fit and put them in front
def findBestFit(population,password):
    #create empty dictionary for individuals so they can be called by group key
    mostFitInPopulation = {} #makes a dictionary
    #for each individual in the previous population, sort them and store them
    for individual in population:
        mostFitInPopulation[individual] = computeChildFitness(password, individual)
    #return a sorted dictionary, sorteed from fit to not fit
    #get the first item ---> which is the most fit since we worted it that way
    #print("mostFitInPopulation----------->\n",mostFitInPopulation)
    #print("\nSorted Most fit--------------_>\n",sorted(mostFitInPopulation.items(), key = operator.itemgetter(1),reverse=True))
    return sorted(mostFitInPopulation.items(), key = operator.itemgetter(1),reverse=True)


#---------------------------SELECTION----------------------------------------------------------------------

#define functon to pick the best solutions so they can mate later
def matingSelection(sortedPopulation,bestSample,luckyFew):
#make an ordered list to store new generation
    nextGeneration = []
    #pick the best samples an a few lucky (random) samples, add them to newGen lis
    for eachSolution in range(bestSample):
        #nextGeneration.append(the sorted population)
        nextGeneration.append(sortedPopulation[eachSolution][0])
    for eachSolution in range(luckyFew):
        nextGeneration.append(random.choice(sortedPopulation)[0])
    #print('Pre Sorted Next Generation: ',nextGeneration, "\n") 
    random.shuffle(nextGeneration)
    #print('Post Sorted Next Generation: ',nextGeneration, "\n")
    #print('next generation----->',nextGeneration)
    return nextGeneration


#---------------------------CROSSOVER MATING---------------------------------------------------------------------------

import sys

#define child and children creation function
#remember individuals (parents, children) are --->> soluttions to the probelem
def createChild(individual1, individual2):
    #print('Individual 1', individual1)
    #print('Individual 2', individual2)
    global child
    child = ""
    #print("This Child Should be a string: "''+ child +''"----> Is it a string???\n")
    #build a child gene by gene
    for eachGene in range(len(individual1)):
        #make odds of getting each gene 50/50
        #from either parents curent gene position
        if (int(100 * random.random()) < 50):
            #print("Indi 1",type(child))
            child += individual1[eachGene]
            #print("Indi 1",type(child))
            #print("This Child Should be a string: "''+ child +''"----> Is it a string???\n")

                
        else:
           # print("Indi 2",type(child))
            child += individual2[eachGene]
            #print("Indi 2",type(child))
            #print("This Child Should be a string: "''+ child +''"----> Is it a string???\n")
    #print("CHILD = ", child)
  #  input()
    if (child == password):
        #historicalData = multipleGeneration(generationCountLimit,password,populationSize,bestSample, luckyFew, childrenPerCouple, mutationProbability)
        print('Current Child is: ',child)
        evolutionBestFitness(historicalData, password)
        evolutionAverageFitness(historicalData, password, populationSize)
        sys.exit()
    return child

def createChildPopulation(parentPopulation, childPopulation):
    #print('parentPopulation: \n', parentPopulation)
    #print('childPopulation:\n', childPopulation)
    
    nextPopulation = []
    #print("Parent Population is: \n",parentPopulation, "\n")
    for eachNewCouple in range(int((len(parentPopulation)/2))):
        for eachNewChild in range(childPopulation):
            #print("create child function",createChild(parentPopulation[eachNewCouple],parentPopulation[len(parentPopulation) -1 -eachNewCouple]))
            nextPopulation.append(createChild(parentPopulation[eachNewCouple],parentPopulation[len(parentPopulation) -1 -eachNewCouple]))
            
    #print("next population",nextPopulation)
    #input()
    return nextPopulation


#------------------------MUTATION-----------------------------------------------------------------------------------


#define mutation functiont to randomly change a gene
def mutateAChild(child):
    geneMutationIndex = int(random.random() * len(child))
    #if
    if(geneMutationIndex == 0):
        child = chr(97 + int(26 * random.random())) + child[1:]
    else:
        child = child[:geneMutationIndex] + chr(97 + int(26 * random.random())) + child[geneMutationIndex+1:]
    return child
    
#each child needs the same chance of mutation, but some might not have one
def mutatePopulation(population, mutationProbability):
    for eachIndividual in range(len(population)):
        if random.random() * 100 <  mutationProbability:
            population[eachIndividual] = mutateAChild(population[eachIndividual])
    return population


#------------------------SIMULATE GENERATIONS-----------------------------------------------------------------------------------------------


#define a function that creates a new generation using our previous functions
def nexGeneration(firstGeneration, password, bestSample, luckyFew, childrenPerCouple, mutationProbability):
    sortedPopulation = findBestFit(firstGeneration, password)
    nextCouple = matingSelection(sortedPopulation,bestSample,luckyFew)
    nextPopulation = createChildPopulation(nextCouple, childrenPerCouple)
#    print("Next Couple:-------->  ",nextCouple)
#    print("Children Couple:-------->  ",childrenPerCouple)
    nextGeneration = mutatePopulation(nextPopulation, mutationProbability)
    #print('nEXT GEN',nextGeneration)
    return nextGeneration

#define function that crate multiple generations in succesion
def multipleGeneration(generationCountLimit,password,populationSize,bestSample, lucky_few, childrenPerCouple, mutationProbability):
    #print("Inside Multiple Gen")
    #create historical database of past generations
    global historicalData
    historicalData = []
    #add first generation to historical record
    historicalData.append(generatePopulation(populationSize,password))
    for eachGeneration in range(generationCountLimit):
        historicalData.append(nexGeneration(historicalData[eachGeneration], password, bestSample, luckyFew, childrenPerCouple, mutationProbability))
    return historicalData


#-------------------------ANALYSIS---------------------------------------------------------------------------------


#print results and analyze data
def showBestResults(historicalData, password, generationCountLimit):
    #
    global result
    result= getList_ofBest_FromHistory(historicalData, password,)[generationCountLimit-1]
    print("The closest password attempt was: \"" + result[0] + "\" with a fitness score of: " + str(result[1]))

def getList_ofBest_FromPopulation(population,password):
    #print('find Most Fit---->',findMostFit(population,password)[0])
    return findBestFit(population,password)[0]

def getList_ofBest_FromHistory(historicalData, password):
    #create ordered list of best indivduals
    bestIndividuals = []
    #add best from each population to list of best overall
    for eachPopulation in historicalData:
        bestIndividuals.append(getList_ofBest_FromPopulation(eachPopulation, password))
    return bestIndividuals


#------------------------VISUALIZATION--------------------------------------------------------------------------------------


#graph
def evolutionBestFitness(historicalData, password):
	plt.axis([0,len(historicalData),0,105])
	plt.title(password)
	
	evolutionFitness = []
	for population in historicalData:
		evolutionFitness.append(getList_ofBest_FromPopulation(population, password)[1])
	plt.plot(evolutionFitness)
	plt.ylabel('fitness best individual')
	plt.xlabel('generation')
	plt.show()

def evolutionAverageFitness(historicalData, password, populationSize):
	plt.axis([0,len(historicalData),0,105])
	plt.title(password)
	
	evolutionFitness = []
	for population in historicalData:
		populationPerf = findBestFit(population, password)
		averageFitness = 0
		for individual in populationPerf:
			averageFitness += individual[1]
		evolutionFitness.append(averageFitness/populationSize)
	plt.plot(evolutionFitness)
	plt.ylabel('Average fitness')
	plt.xlabel('generation')
	plt.show()


#------------------------MAIN PRAGRAM--------------------------------------------------------------------------------------
#program

#while (child != password):
#    if ((bestSample + luckyFew) / 2 * childrenPerCouple != populationSize):
#        print ("population size not stable")
#    else:
#        print('Child in while: ', child)
#        historicalData = multipleGeneration(generationCountLimit,password,populationSize,bestSample, luckyFew, childrenPerCouple, mutationProbability)
#        print('Child in while: ', child)
#        
#print('Child in main: ', child)
#
#evolutionBestFitness(historicalData, password)
#evolutionAverageFitness(historicalData, password, populationSize)
#
#
#print(time.time() - temps1)
    
if ((bestSample + luckyFew) / 2 * childrenPerCouple != populationSize):
    print ("population size not stable")
else:
    historicalData = multipleGeneration(generationCountLimit,password,populationSize,bestSample, luckyFew, childrenPerCouple, mutationProbability)
evolutionBestFitness(historicalData, password)
evolutionAverageFitness(historicalData, password, populationSize)


print(time.time() - temps1)    




















#import random
#import operator
#import time
#import matplotlib.pyplot as plt
#
#temps1 = time.time()
#
##genetic algorithm function
#def fitness (password, test_word):
#	score = 0
#	i = 0
#	while (i < len(password)):
#		if (password[i] == test_word[i]):
#			score+=1
#		i+=1
#	return score * 100 / len(password)
#
#def generateAWord (length):
#	i = 0
#	result = ""
#	while i < length:
#		letter = chr(97 + int(26 * random.random()))
#		result += letter
#		i +=1
#	return result
#
#def generateFirstPopulation(sizePopulation, password):
#	population = []
#	i = 0
#	while i < sizePopulation:
#		population.append(generateAWord(len(password)))
#		i+=1
#	return population
#
#def computePerfPopulation(population, password):
#    populationPerf = {}
#    for individual in population:
#        populationPerf[individual] = fitness(password, individual)
#        print('\n\n',(sorted(populationPerf.items(), key = operator.itemgetter(1), reverse=True)))
#    return sorted(populationPerf.items(), key = operator.itemgetter(1), reverse=True)
#   
#
#def selectFromPopulation(populationSorted, best_sample, lucky_few):
#	nextGeneration = []
#	for i in range(best_sample):
#		nextGeneration.append(populationSorted[i][0])
#	for i in range(lucky_few):
#		nextGeneration.append(random.choice(populationSorted)[0])
#	random.shuffle(nextGeneration)
#	return nextGeneration
#
#def createChild(individual1, individual2):
#	child = ""
#	for i in range(len(individual1)):
#		if (int(100 * random.random()) < 50):
#			child += individual1[i]
#		else:
#			child += individual2[i]
#	return child
#
#def createChildren(breeders, number_of_child):
#	nextPopulation = []
#	for i in range(int(len(breeders)/2)):
#		for j in range(number_of_child):
#			nextPopulation.append(createChild(breeders[i], breeders[len(breeders) -1 -i]))
#	return nextPopulation
#
#def mutateWord(word):
#	index_modification = int(random.random() * len(word))
#	if (index_modification == 0):
#		word = chr(97 + int(26 * random.random())) + word[1:]
#	else:
#		word = word[:index_modification] + chr(97 + int(26 * random.random())) + word[index_modification+1:]
#	return word
#	
#def mutatePopulation(population, chance_of_mutation):
#	for i in range(len(population)):
#		if random.random() * 100 < chance_of_mutation:
#			population[i] = mutateWord(population[i])
#	return population
#
#def nextGeneration (firstGeneration, password, best_sample, lucky_few, number_of_child, chance_of_mutation):
#	 populationSorted = computePerfPopulation(firstGeneration, password)
#	 nextBreeders = selectFromPopulation(populationSorted, best_sample, lucky_few)
#	 nextPopulation = createChildren(nextBreeders, number_of_child)
#	 nextGeneration = mutatePopulation(nextPopulation, chance_of_mutation)
#	 return nextGeneration
#
#def multipleGeneration(number_of_generation, password, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation):
#	historic = []
#	historic.append(generateFirstPopulation(size_population, password))
#	for i in range (number_of_generation):
#		historic.append(nextGeneration(historic[i], password, best_sample, lucky_few, number_of_child, chance_of_mutation))
#	return historic
#
##print result:
#def printSimpleResult(historic, password, number_of_generation): #bestSolution in historic. Caution not the last
#	result = getListBestIndividualFromHistorique(historic, password)[number_of_generation-1]
#	print ("solution: \"" + result[0] + "\" de fitness: " + str(result[1]))
#
##analysis tools
#def getBestIndividualFromPopulation (population, password):
#	return computePerfPopulation(population, password)[0]
#
#def getListBestIndividualFromHistorique (historic, password):
#	bestIndividuals = []
#	for population in historic:
#		bestIndividuals.append(getBestIndividualFromPopulation(population, password))
#	return bestIndividuals
#
##graph
#def evolutionBestFitness(historic, password):
#	plt.axis([0,len(historic),0,105])
#	plt.title(password)
#	
#	evolutionFitness = []
#	for population in historic:
#		evolutionFitness.append(getBestIndividualFromPopulation(population, password)[1])
#	plt.plot(evolutionFitness)
#	plt.ylabel('fitness best individual')
#	plt.xlabel('generation')
#	plt.show()
#
#def evolutionAverageFitness(historic, password, size_population):
#	plt.axis([0,len(historic),0,105])
#	plt.title(password)
#	
#	evolutionFitness = []
#	for population in historic:
#		populationPerf = computePerfPopulation(population, password)
#		averageFitness = 0
#		for individual in populationPerf:
#			averageFitness += individual[1]
#		evolutionFitness.append(averageFitness/size_population)
#	plt.plot(evolutionFitness)
#	plt.ylabel('Average fitness')
#	plt.xlabel('generation')
#	plt.show()
#
##variables
#password = "banana"
#size_population = 100
#best_sample = 2
#lucky_few = 0
#number_of_child = 100
#number_of_generation = 50
#chance_of_mutation = 99
#
##program
#if ((best_sample + lucky_few) / 2 * number_of_child != size_population):
#	print ("population size not stable")
#else:
#	historic = multipleGeneration(number_of_generation, password, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation)
#	
#	printSimpleResult(historic, password, number_of_generation)
#	
#	evolutionBestFitness(historic, password)
#	evolutionAverageFitness(historic, password, size_population)
#
#print(time.time() - temps1)