# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:13:45 2018

@author: L1817
"""

import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
#import argparse

import preprocess_data,  utilities, hydro, hydro_utils


plt.close("all")

"""
Read general help on main.README.txt
"""

time0 = time.time()

np.random.seed(3)

"""
Parse command-line arguments
"""
#parser = argparse.ArgumentParser(description='Run the general code without any optimization algorithm.')
#
#parser.add_argument('-n','--niter', default=10, help='(int) Number of outermost iterations of the fipy solver, be it steadystate or transient. Default=10.', type=int)
#parser.add_argument('-b','--nblocks', default=5, help='(int) Number of blocks to locate. Default=5.', type=int)
#args = parser.parse_args()
#
#days = args.niter # outtermost loop in fipy. 'timesteps' in fipy manual.
#nblocks = args.nblocks


"""
Read and preprocess data
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

# catchment mask
catchment_mask = np.ones(shape=dem.shape, dtype=bool)
catchment_mask[np.where(dem<-10)] = False # -99999.0 is current value of dem for nodata points.

# peel the dem. Only when dem is not surrounded by water
boundary_mask = utilities.peel_raster(dem, catchment_mask)
 

# after peeling, catchment_mask should only be the fruit:
catchment_mask[boundary_mask] = False

# soil types and soil physical properties and soil depth:
peat_type_mask = peat_type_arr * catchment_mask
peat_bottom_elevation = - peat_depth_arr * catchment_mask # meters with respect to dem surface. Should be negative!


h_to_tra_and_C_dict = hydro_utils.peat_map_interp_functions() # Load peatmap soil types' physical properties dictionary
#soiltypes[soiltypes==255] = 0 # 255 is nodata value. 1 is water (useful for hydrology! Maybe, same treatment as canals).

tra_to_cut = hydro_utils.peat_map_h_to_tra(soil_type_mask=peat_type_mask,
                                           gwt=peat_bottom_elevation, h_to_tra_and_C_dict=h_to_tra_and_C_dict)
sto_to_cut = hydro_utils.peat_map_h_to_sto(soil_type_mask=peat_type_mask,
                                           gwt=peat_bottom_elevation, h_to_tra_and_C_dict=h_to_tra_and_C_dict)

srfcanlist =[dem[coords] for coords in c_to_r_list]

n_canals = len(c_to_r_list)
n_blocks = 5
block_height = 0.4 # water level of canal after placing dam.

# HANDCRAFTED WATER LEVEL IN CANALS. CHANGE WITH MEASURED, IDEALLY.
canal_water_level = 1.2
oWTcanlist = [x - canal_water_level for x in srfcanlist]



"""
Initial configuration of blocks in canals
"""
iDamLocation = np.random.randint(0,n_canals,n_blocks).tolist() # Generate random kvector
iWTcanlist = utilities.place_dams(oWTcanlist, srfcanlist, block_height, iDamLocation, CNM)



"""
Metropolis/Simulated Annealing.
"""
# This would be one of the movements of the SimulatedAnnealing
#wt_canals = utilities.switch_one_dam(oWTcanlist, srfcanlist, iWTcanlist, block_height, iDamLocation, n_canals, CNM)

#time0 = time.time()
#
#Tini = 1.0
#Tfin = 0.0
#Tstep = 0.09
#MCsteps = 100
#WT0canal, E0canal, bpos0, acc, MCfile = opti.SimAnneal('canals',
#                                               WT0, oWTcanArr, srfcanlist, kvector0Arr, bpos0,
#                                               prohib_list, ktndict, n_nodes, n_blocks, block_height, n_canals,
#                                               MCsteps=MCsteps, Tini=Tini, Tfin=Tfin, Tstep=Tstep,
#                                               T_cooling=True, CNM=CNM, save_to_file=True, save_to_file_freq=0.001)

wt_canals = iWTcanlist


"""
#########################################
                HYDROLOGY
#########################################
"""
ny, nx = dem.shape
dx = 1.; dy = 1. # metres per pixel  
diri_bc = 0.0


hini = - 0.0 # initial wt wrt surface elevation in meters.

boundary_arr = boundary_mask * (dem - diri_bc) # constant Dirichlet value in the boundaries

ele = dem

# Get a pickled phi solution (not ele-phi!) computed before without blocks, independently,
# and use it as initial condition to improve convergence time of the new solution
retrieve_transient_phi_sol_from_pickled = False
if retrieve_transient_phi_sol_from_pickled:
    with open(r"pickled/transient_phi_sol.pkl", 'r') as f:
        phi_ini = pickle.load(f)
    print "transient phi solution loaded as initial condition"
    
else:
    phi_ini = ele + hini #initial h (gwl) in the compartment.
    
    

wt_canal_arr = np.zeros((ny,nx)) # (nx,ny) array with wt canal height in corresponding nodes
owt_canal_arr = np.zeros((ny,nx)) # checking purposes
for canaln, coords in enumerate(c_to_r_list):
    if canaln == 0: 
        continue # because c_to_r_list begins at 1
    wt_canal_arr[coords] = wt_canals[canaln] 
    owt_canal_arr[coords] = oWTcanlist[canaln]


dry_peat_volume, wt, dneg = hydro.hydrology('transient', nx, ny, dx, dy, ele, phi_ini, catchment_mask, wt_canal_arr, boundary_arr,
                                                  peat_type_mask=peat_type_mask, httd=h_to_tra_and_C_dict, tra_to_cut=tra_to_cut, sto_to_cut=sto_to_cut,
                                                  diri_bc=diri_bc, neumann_bc = None, plotOpt=True, remove_ponding_water=True)


"""
Final printings
"""
timespent = time.time() - time0
utilities.print_time_in_mins(timespent)
