# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:06:25 2018

@author: L1817
"""

import numpy as np
import random
import multiprocessing
import time
import pickle
import scipy.sparse as sparse
from deap import creator, base, tools, algorithms
import preprocess_data, utilities, hydro


time0 = time.time()

"""
###########################
Read and preprocess data
###########################
"""
retrieve_canalarr_from_pickled = False

preprocessed_datafolder= r"C:\Users\L1817\Dropbox\PhD\Computation\Indonesia_WaterTable\ForestCarbon\preprocessed_data"
rst_dem_fn = preprocessed_datafolder + r'\50x50dem_filled_and_interpolated.tif'

if 'cm' and 'cr' and 'c_to_r_list' not in globals():
    datafolder = r"C:\Users\L1817\Dropbox\PhD\Computation\Indonesia_WaterTable\ForestCarbon\data"
    shp_fn = datafolder + r"\canal2017_clip.shp"
    rst_out_fn = preprocessed_datafolder + r'\canals_rasterized.tif'

    cm, cr, c_to_r_list =  preprocess_data.gen_can_matrix_and_raster_from_raster(shp_fn=shp_fn,
                                    rst_dem_fn=rst_dem_fn, rst_out_fn=rst_out_fn,
                                    preprocessed_datafolder=preprocessed_datafolder)

elif retrieve_canalarr_from_pickled==True:
    pickle_folder = r"C:\Users\L1817\ForestCarbon"
    pickled_canal_array_fn = r'\50x50_DEM_and_canals.pkl'
    with open(pickle_folder + pickled_canal_array_fn) as f:
        cm, cr, c_to_r_list = pickle.load(f)
    print "Canal adjacency matrix and raster loaded from pickled."
    
else:
    print "Canal adjacency matrix and raster loaded from memory."

dem = utilities.read_DEM(rst_dem_fn)
print("DEM read from file")


"""
##################################
    parameters and hydrology setup
####################################
"""

n_canals = len(c_to_r_list)
N_BLOCKS = 3
BLOCK_HEIGHT = 0.4 # water level of canal after placing dam.
CANAL_WATER_LEVEL = 1.2 # water level of canal before placing dams

srfcanlist =[dem[coords] for coords in c_to_r_list]
oWTcanlist = [x - CANAL_WATER_LEVEL for x in srfcanlist]

cnm = sparse.csr_matrix(cm)
cnm.eliminate_zeros()  # happens in place. Frees disk usage.

ny, nx = dem.shape
dx = 1.; dy = 1. # metres per pixel  
dt = 1. # timestep, in days
DIRI_BC = 1.2 # ele-phi in meters
hINI = - 0.9 # initial wt wrt surface elevation in meters.

ele = dem[:]
Hinitial = ele + hINI #initial h (gwl) in the compartment.
#depth = np.ones(shape=Hinitial.shape) # elevation of the impermeable bottom from comon ref. point
# catchment mask
catchment_mask = np.ones(shape=Hinitial.shape, dtype=bool)
catchment_mask[np.where(dem==-9999)] = False # -9999 is current value of dem for outlayer points.



creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # if weights are negative, we have minimization. They must be a tuple.
creator.create("Individual", list, fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, n_canals)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_int, n=N_BLOCKS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalDryPeatVol(individual): # this should be returning dry peat volume in a tuple
    wt_canals = utilities.place_dams(oWTcanlist, srfcanlist, BLOCK_HEIGHT, individual, cnm)
    
    wt_canal_arr = np.zeros((ny,nx)) # (nx,ny) array with wt canal height in corresponding nodes
    for canaln, coords in enumerate(c_to_r_list):
        wt_canal_arr[coords] = wt_canals[canaln]
        Hinitial[coords] = wt_canals[canaln]
    
    dry_peat_volume = hydro.hydrology(nx, ny, dx, dy, dt, ele, Hinitial, catchment_mask, wt_canal_arr, value_for_masked= 0.9, diri_bc=None, neumann_bc = 0.0)

    return dry_peat_volume,

toolbox.register("evaluate", evalDryPeatVol)
toolbox.register("mate", tools.cxOnePoint) # single point crossover
toolbox.register("mutate", tools.mutUniformInt, low=0, up=n_canals, indpb=0.1) # replaces individual's attribute with random int
toolbox.register("select", tools.selBest) # k best are selected

if __name__ == "__main__":
#    random.seed(64)
    N_GENERATIONS = 5
    
    pool = multiprocessing.Pool(processes=7)
    toolbox.register("map", pool.map)
    
    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=N_GENERATIONS, 
                        stats=stats, halloffame=hof, verbose=1)

    pool.close()

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual of current population is %s, %s" % (best_ind, best_ind.fitness.values))
    print("Best individual ever is %s, %s" % (hof[0],hof[0].fitness.values))
    with open(r'C:\Users\L1817\Dropbox\PhD\Computation\Indonesia_WaterTable\ForestCarbon\computation\after_21.11.2018\results_ga.txt', 'a') as output_file:
        output_file.write("\n dry_peat_volume = " + str(best_ind.fitness.values) + "    blocked dams:" + str(best_ind) + "   DATE: " + str(time.ctime()) + "   Number of generations = " + str(N_GENERATIONS))
    
    print("\n timespent = ", time.time() - time0)