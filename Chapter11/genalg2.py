# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 13:58:57 2020

@author: qtckp
"""


import random
import numpy as np
import os

import image_test


# problem related constants
POLYGON_SIZE = 3
NUM_OF_POLYGONS = 200

# calculate total number of params in chromosome:
# For each polygon we have:
# two coordinates per vertex, 3 color values, one alpha value
NUM_OF_PARAMS = NUM_OF_POLYGONS * (POLYGON_SIZE * 2 + 4)

# Genetic Algorithm constants:
POPULATION_SIZE = 200
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.5   # probability for mutating an individual
MAX_GENERATIONS = 5000
HALL_OF_FAME_SIZE = 20


# set the random seed:
RANDOM_SEED = 11
random.seed(RANDOM_SEED)

# create the image test class instance:
imageTest = image_test.ImageTest("images/me1.jpg", POLYGON_SIZE, 'white')

# calculate total number of params in chromosome:
# For each polygon we have:
# two coordinates per vertex, 3 color values, one alpha value
NUM_OF_PARAMS = NUM_OF_POLYGONS * (POLYGON_SIZE * 2 + 4)

# all parameter values are bound between 0 and 1, later to be expanded:
BOUNDS_LOW, BOUNDS_HIGH = 0.0, 1.0  # boundaries for all dimensions


# minimized function -- difference between images
def getDiff(individual):
    return imageTest.getDifference(individual, "MSE")
    #return imageTest.getDifference(individual, "SSIM")


from geneticalgorithm2 import geneticalgorithm2 as ga # GA method
from geneticalgorithm2 import Callbacks # for callbacks


model = ga(function = getDiff, 
    dimension = NUM_OF_PARAMS, 
    variable_type = 'real', 
    variable_boundaries = np.array([[BOUNDS_LOW, BOUNDS_HIGH]]*NUM_OF_PARAMS),
    function_timeout = 100, 
    algorithm_parameters = {'max_num_iteration': 15,
                            'population_size':POPULATION_SIZE,
                            'mutation_probability': P_MUTATION,
                            'elit_ratio': HALL_OF_FAME_SIZE/POPULATION_SIZE,
                            'crossover_probability': P_CROSSOVER,
                            'parents_portion': 0.3,
                            'crossover_type':'two_point',
                            'mutation_type': 'uniform_by_center',
                            'selection_type': 'tournament',
                            'max_iteration_without_improv':None
                            }
    )




# save the best current drawing (used as a callback):
def saveImage(gen, polygonData):

    # create folder if does not exist:
    folder = "images/results/run-{}-{}".format(POLYGON_SIZE, NUM_OF_POLYGONS)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # save the image in the folder:
    imageTest.saveImage(polygonData, "{}/after-{}-gen.png".format(folder, gen), "After {} Generations".format(gen))

# convert to callback for GA
def my_callback(gen, report, pop, scores):
    if gen % 10 == 0: saveImage(gen, pop[0, :])




#st = np.load('data_saved/population_160.npz')

model.run(
    #start_generation = {'variables':st['population'], 'scores': st['scores']},
    studEA = False,
    callbacks = [
        Callbacks.PlotOptimizationProcess('plots', save_gen_step = 10, show = False),
        Callbacks.SavePopulation('data_saved', save_gen_step=5),
        my_callback
        ],
    time_limit_secs= 32000
    )


saveImage(len(model.report), model.output_dict['variable']) # plot solution


