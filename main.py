# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:13:45 2018

@author: L1817
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
#import argparse
import time

import preprocess_data,  utilities, hydro, hydro_utils


plt.close("all")

"""
Read general help on main.README.txt
"""

"""
Parse command-line arguments
"""
parser = argparse.ArgumentParser(description='Run hydro without any optimization.')

parser.add_argument('-d','--days', default=3, help='(int) Number of outermost iterations of the fipy solver, be it steadystate or transient. Default=10.', type=int)
parser.add_argument('-b','--nblocks', default=0, help='(int) Number of blocks to locate. Default=5.', type=int)
parser.add_argument('-n','--niter', default=1, help='(int) Number of repetitions of the whole computation. Default=10', type=int)
args = parser.parse_args()

DAYS = args.days
N_BLOCKS = args.nblocks
N_ITER = args.niter


"""
Read and preprocess data
"""


""" Stratification 2 data version. Remove in the future"""
#preprocessed_datafolder = r"data/Strat2"
#dem_rst_fn = preprocessed_datafolder + r"/lidar_100_resampled_interp.tif"
#can_rst_fn = preprocessed_datafolder + r"/canal_clipped_resampled_2.tif"
#peat_type_rst_fn = preprocessed_datafolder + r"/Landcover_clipped.tif"
#peat_depth_rst_fn = preprocessed_datafolder + r"/peat_depth.tif"

"""Stratification 4  keep this!"""
preprocessed_datafolder = r"data/Strat4"
dem_rst_fn = preprocessed_datafolder + r"/DTM_metres_clip.tif"
can_rst_fn = preprocessed_datafolder + r"/canals_clip.tif"
#land_use_rst_fn = preprocessed_datafolder + r"/Landcover2017_clip.tif" # Not used
peat_depth_rst_fn = preprocessed_datafolder + r"/Peattypedepth_clip.tif" # peat depth, peat type in the same raster
#params_fn = r"/home/inaki/GitHub/dd_winrock/data/params.xlsx" # Luke
#params_fn = r"/home/txart/Programming/GitHub/dd_winrock/data/params.xlsx" # home
params_fn = r"/homeappl/home/urzainqu/dd_winrock/data/params.xlsx" # CSC


if 'CNM' and 'cr' and 'c_to_r_list' not in globals():
    CNM, cr, c_to_r_list = preprocess_data.gen_can_matrix_and_raster_from_raster(can_rst_fn=can_rst_fn, dem_rst_fn=dem_rst_fn)

#else:
#    print "Canal adjacency matrix and raster loaded from memory."
    
_ , dem, peat_type_arr, peat_depth_arr = preprocess_data.read_preprocess_rasters(can_rst_fn, dem_rst_fn, peat_depth_rst_fn, peat_depth_rst_fn)

PARAMS_df = preprocess_data.read_params(params_fn)
BLOCK_HEIGHT = PARAMS_df.block_height[0]; CANAL_WATER_LEVEL = PARAMS_df.canal_water_level[0]
DIRI_BC = PARAMS_df.diri_bc[0]; HINI = PARAMS_df.hini[0]; P = PARAMS_df.P[0]
ET = PARAMS_df.ET[0]; TIMESTEP = PARAMS_df.timeStep[0]; KADJUST = PARAMS_df.Kadjust[0]

print(">>>>> WARNING, OVERWRITING PEAT DEPTH")
peat_depth_arr[peat_depth_arr < 2.] = 2.

# catchment mask
catchment_mask = np.ones(shape=dem.shape, dtype=bool)
catchment_mask[np.where(dem<-10)] = False # -99999.0 is current value of dem for nodata points.

# peel the dem. Only when dem is not surrounded by water
boundary_mask = utilities.peel_raster(dem, catchment_mask)
 
# after peeling, catchment_mask should only be the fruit:
catchment_mask[boundary_mask] = False

# soil types and soil physical properties and soil depth:
peat_type_masked = peat_type_arr * catchment_mask
peat_bottom_elevation = - peat_depth_arr * catchment_mask # meters with respect to dem surface. Should be negative!
#

h_to_tra_and_C_dict, K = hydro_utils.peat_map_interp_functions(Kadjust=KADJUST) # Load peatmap soil types' physical properties dictionary

# Plot K
#import matplotlib.pyplot as plt
#plt.figure(); z = np.linspace(0.0, -20.0, 400); plt.plot(K,z); plt.title('K')
#soiltypes[soiltypes==255] = 0 # 255 is nodata value. 1 is water (useful for hydrology! Maybe, same treatment as canals).

#BOTTOM_ELE = -6.0 
#peat_bottom_elevation = np.ones(shape=dem.shape) * BOTTOM_ELE
#peat_bottom_elevation = peat_bottom_elevation*catchment_mask
tra_to_cut = hydro_utils.peat_map_h_to_tra(soil_type_mask=peat_type_masked,
                                           gwt=peat_bottom_elevation, h_to_tra_and_C_dict=h_to_tra_and_C_dict)
sto_to_cut = hydro_utils.peat_map_h_to_sto(soil_type_mask=peat_type_masked,
                                           gwt=peat_bottom_elevation, h_to_tra_and_C_dict=h_to_tra_and_C_dict)
sto_to_cut = sto_to_cut * catchment_mask.ravel()

srfcanlist =[dem[coords] for coords in c_to_r_list]

n_canals = len(c_to_r_list)


# HANDCRAFTED WATER LEVEL IN CANALS. CHANGE WITH MEASURED, IDEALLY.
oWTcanlist = [x - CANAL_WATER_LEVEL for x in srfcanlist]

hand_made_dams = False # compute performance of cherry-picked locations for dams.
quasi_random = False # Don't allow overlapping blocks
"""
MonteCarlo
"""
for i in range(0,N_ITER):
    
    if quasi_random == False or i==0: # Normal fully random block configurations
        damLocation = np.random.randint(1, n_canals, N_BLOCKS).tolist() # Generate random kvector. 0 is not a good position in c_to_r_list
    else:
        prohibited_node_list = [i for i,_ in enumerate(oWTcanlist[1:]) if oWTcanlist[1:][i] < wt_canals[1:][i]]      # [1:] is to take the 0th element out of the loop
        candidate_node_list = np.array([e for e in range(1, n_canals) if e not in prohibited_node_list]) # remove 0 from the range of possible canals
        damLocation = np.random.choice(candidate_node_list, size=N_BLOCKS)

    if hand_made_dams:
        # HAND-MADE RULE OF DAM POSITIONS TO ADD:
        hand_picked_dams = (11170, 10237, 10514, 2932, 4794, 8921, 4785, 5837, 7300, 6868) # rule-based approach
        hand_picked_dams = [11170, 10237, 10514, 2932, 4794, 8921, 4785, 5837, 7300, 6868]
        damLocation = hand_picked_dams
    
    wt_canals = utilities.place_dams(oWTcanlist, srfcanlist, BLOCK_HEIGHT, damLocation, CNM)
    """
    #########################################
                    HYDROLOGY
    #########################################
    """
    ny, nx = dem.shape
    dx = 1.; dy = 1. # metres per pixel  
    
    boundary_arr = boundary_mask * (dem - DIRI_BC) # constant Dirichlet value in the boundaries
    
    ele = dem * catchment_mask
    
    # Get a pickled phi solution (not ele-phi!) computed before without blocks, independently,
    # and use it as initial condition to improve convergence time of the new solution
    retrieve_transient_phi_sol_from_pickled = False
    if retrieve_transient_phi_sol_from_pickled:
        with open(r"pickled/transient_phi_sol.pkl", 'r') as f:
            phi_ini = pickle.load(f)
        print "transient phi solution loaded as initial condition"
        
    else:
        phi_ini = ele + HINI #initial h (gwl) in the compartment.
        phi_ini = phi_ini * catchment_mask
           
    wt_canal_arr = np.zeros((ny,nx)) # (nx,ny) array with wt canal height in corresponding nodes
    for canaln, coords in enumerate(c_to_r_list):
        if canaln == 0: 
            continue # because c_to_r_list begins at 1
        wt_canal_arr[coords] = wt_canals[canaln] 
    
    
    dry_peat_volume = hydro.hydrology('transient', nx, ny, dx, dy, DAYS, ele, phi_ini, catchment_mask, wt_canal_arr, boundary_arr,
                                                      peat_type_mask=peat_type_masked, httd=h_to_tra_and_C_dict, tra_to_cut=tra_to_cut, sto_to_cut=sto_to_cut,
                                                      diri_bc=DIRI_BC, neumann_bc = None, plotOpt=False, remove_ponding_water=True,
                                                      P=P, ET=ET, dt=TIMESTEP)
    
    water_blocked_canals = sum(np.subtract(wt_canals[1:], oWTcanlist[1:]))
    
    cum_Vdp_nodams = 21088.453521509597
    print('dry_peat_volume(%) = ', dry_peat_volume/cum_Vdp_nodams * 100, '\n',
          'water_blocked_canals = ', water_blocked_canals)

    """
    Final printings
    """
    if quasi_random == True:
        fname = r'output/results_mc_quasi_3.txt'
    else:
        fname = r'output/results_mc_3_cumulative.txt'
    if N_ITER > 20:
        
        with open(fname, 'a') as output_file:
            output_file.write(
                                "\n" + str(i) + "    " + str(dry_peat_volume) + "    "
                                + str(N_BLOCKS) + "    " + str(N_ITER) + "    " + str(DAYS) + "    "
                                + str(time.ctime()) + "    " + str(water_blocked_canals)
                              )



