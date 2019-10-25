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
from deap import creator, base, tools, algorithms
import preprocess_data, utilities, hydro, hydro_utils
import argparse



"""
Parse command-line arguments
"""
parser = argparse.ArgumentParser(description='Run GA.')

parser.add_argument('-d','--days', default=3, help='(int) Number of outermost iterations of the fipy solver, be it steadystate or transient. Default=10.', type=int)
parser.add_argument('-b','--nblocks', default=5, help='(int) Number of blocks to locate. Default=5.', type=int)
parser.add_argument('-n','--nopti', default=5, help='(int) Number of iterations of the optimization algorithm. Number of generations in GA. Default=100.', type=int)
parser.add_argument('-p', '--processes', default=2, help='(int) Number of parallel processes for the optimization', type=int)
args = parser.parse_args()

DAYS = args.days
N_BLOCKS = args.nblocks
N_GENERATIONS = args.nopti
N_PROCESSES = args.processes

#N_BLOCKS = 3
#N_GENERATIONS = 5

"""
###########################
Read and preprocess data
###########################
"""
retrieve_canalarr_from_pickled = False
preprocessed_datafolder = r"data"
dem_rst_fn = preprocessed_datafolder + r"/lidar_100_resampled_interp.tif"
can_rst_fn = preprocessed_datafolder + r"/canal_clipped_resampled_2.tif"
peat_type_rst_fn = preprocessed_datafolder + r"/Landcover_clipped.tif"
peat_depth_rst_fn = preprocessed_datafolder + r"/peat_depth.tif"

if 'CNM' and 'cr' and 'c_to_r_list' not in globals():
    CNM, cr, c_to_r_list = preprocess_data.gen_can_matrix_and_raster_from_raster(can_rst_fn=can_rst_fn, dem_rst_fn=dem_rst_fn)

elif retrieve_canalarr_from_pickled==True:
    pickle_folder = r"C:\Users\L1817\Winrock"
    pickled_canal_array_fn = r'\DEM_and_canals.pkl'
    with open(pickle_folder + pickled_canal_array_fn) as f:
        CNM, cr, c_to_r_list = pickle.load(f)
    print "Canal adjacency matrix and raster loaded from pickled."
    
else:
    print "Canal adjacency matrix and raster loaded from memory."
    
_ , dem, peat_type_arr, peat_depth_arr = preprocess_data.read_preprocess_rasters(can_rst_fn, dem_rst_fn, peat_type_rst_fn, peat_depth_rst_fn)

print("rasters read and preprocessed from file")


"""
##################################
    parameters and hydrology setup
####################################
"""
# catchment mask
catchment_mask = np.ones(shape=dem.shape, dtype=bool)
catchment_mask[np.where(dem<-10)] = False # -99999.0 is current value of dem for nodata points.

# peel the dem. Only when dem is not surrounded by water
boundary_mask = utilities.peel_raster(dem, catchment_mask)
 
# after peeling, catchment_mask should only be the fruit:
#catchment_mask[boundary_mask] = False

# soil types and soil physical properties and soil depth:
peat_type_mask = peat_type_arr * catchment_mask
peat_bottom_elevation = - peat_depth_arr * catchment_mask # meters with respect to dem surface. Should be negative!


h_to_tra_and_C_dict = hydro_utils.peat_map_interp_functions() # Load peatmap soil types' physical properties dictionary
#soiltypes[soiltypes==255] = 0 # 255 is nodata value. 1 is water (useful for hydrology! Maybe, same treatment as canals).

#BOTTOM_ELE = -6.0 
#peat_bottom_elevation = np.ones(shape=dem.shape) * BOTTOM_ELE
#peat_bottom_elevation = peat_bottom_elevation*catchment_mask
tra_to_cut = hydro_utils.peat_map_h_to_tra(soil_type_mask=peat_type_mask,
                                           gwt=peat_bottom_elevation, h_to_tra_and_C_dict=h_to_tra_and_C_dict)
sto_to_cut = hydro_utils.peat_map_h_to_sto(soil_type_mask=peat_type_mask,
                                           gwt=peat_bottom_elevation, h_to_tra_and_C_dict=h_to_tra_and_C_dict)
sto_to_cut = sto_to_cut * catchment_mask.ravel()

srfcanlist =[dem[coords] for coords in c_to_r_list]


n_canals = len(c_to_r_list)

BLOCK_HEIGHT = 0.4 # water level of canal after placing dam.
CANAL_WATER_LEVEL = 1.2 # water level of canal before placing dams
# HANDCRAFTED WATER LEVEL IN CANALS. CHANGE WITH MEASURED, IDEALLY.
srfcanlist =[dem[coords] for coords in c_to_r_list]
oWTcanlist = [x - CANAL_WATER_LEVEL for x in srfcanlist]

n_canals = len(c_to_r_list)


ny, nx = dem.shape
dx = 1.; dy = 1. # metres per pixel  
dt = 1. # timestep, in days
DIRI_BC = 0.0 # ele-phi in meters
hINI = - 0.0 # initial wt wrt surface elevation in meters.

boundary_arr = boundary_mask * (dem - DIRI_BC) # constant Dirichlet value in the boundaries

ele = dem[:]
phi_ini = ele + hINI #initial h (gwl) in the compartment.



creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # if weights are negative, we have minimization. They must be a tuple.
creator.create("Individual", list, fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 1, n_canals-1) # 0 and n_canals are excluded for potential problems with those limiting cases.
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_int, n=N_BLOCKS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalDryPeatVol(individual): # this should be returning dry peat volume in a tuple
    wt_canals = utilities.place_dams(oWTcanlist, srfcanlist, BLOCK_HEIGHT, individual, CNM)
    
    wt_canal_arr = np.zeros((ny,nx)) # (nx,ny) array with wt canal height in corresponding nodes
    for canaln, coords in enumerate(c_to_r_list):
        wt_canal_arr[coords] = wt_canals[canaln]
        phi_ini[coords] = wt_canals[canaln]
    
    dry_peat_volume = hydro.hydrology('transient', nx, ny, dx, dy, DAYS, ele, phi_ini, catchment_mask, wt_canal_arr, boundary_arr,
                                                  peat_type_mask=peat_type_mask, httd=h_to_tra_and_C_dict, tra_to_cut=tra_to_cut, sto_to_cut=sto_to_cut,
                                                  diri_bc=DIRI_BC, neumann_bc = None, plotOpt=False, remove_ponding_water=True)

    return dry_peat_volume,

toolbox.register("evaluate", evalDryPeatVol)
toolbox.register("mate", tools.cxOnePoint) # single point crossover
toolbox.register("mutate", tools.mutUniformInt, low=1, up=n_canals-1, indpb=0.1) # replaces individual's attribute with random int
toolbox.register("select", tools.selBest) # k best are selected

if __name__ == "__main__":
#    random.seed(64)
    N_POPULATION = N_PROCESSES
    
    pool = multiprocessing.Pool(processes=N_PROCESSES)
    toolbox.register("map", pool.map)
    
    pop = toolbox.population(n=N_POPULATION)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=N_GENERATIONS, 
                        stats=stats, halloffame=hof, verbose=0)

    pool.close()

    best_ind = tools.selBest(pop, 1)[0]
#    print("Best individual of current population is %s, %s" % (best_ind, best_ind.fitness.values))
#    print("Best individual ever is %s, %s" % (hof[0],hof[0].fitness.values))
    with open(r'output/results_ga.txt', 'a') as output_file:
        output_file.write("\n" + str(best_ind.fitness.values) + "    " + str(N_BLOCKS) + "    " + str(N_GENERATIONS) + "    " + str(DAYS) + "    " + str(time.ctime()) + "    " + str(hof[0]))

