# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:13:45 2018

@author: L1817
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import preprocess_data,  utilities, hydro, hydro_utils, read


plt.close("all")

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

preprocessed_datafolder = r"data/Strat4"
dem_rst_fn = preprocessed_datafolder + r"/DTM_metres_clip.tif"
can_rst_fn = preprocessed_datafolder + r"/canals_clip.tif"
peat_depth_rst_fn = preprocessed_datafolder + r"/Peattypedepth_clip.tif" # peat depth, peat type in the same raster

abs_path_data = os.path.abspath('./data') # Absolute path to data folder needed for Excel file with parameters
params_fn = abs_path_data + r"/params.xlsx"

# Read rasters, build up canal connectivity adjacency matrix
if 'CNM' and 'cr' and 'c_to_r_list' not in globals(): # call only if needed
    CNM, cr, c_to_r_list = preprocess_data.gen_can_matrix_and_raster_from_raster(can_rst_fn=can_rst_fn, dem_rst_fn=dem_rst_fn)

_ , dem, peat_type_arr, peat_depth_arr = preprocess_data.read_preprocess_rasters(can_rst_fn, dem_rst_fn, peat_depth_rst_fn, peat_depth_rst_fn)

# Read parameters
PARAMS_df = preprocess_data.read_params(params_fn)
BLOCK_HEIGHT = PARAMS_df.block_height[0]; CANAL_WATER_LEVEL = PARAMS_df.canal_water_level[0]
DIRI_BC = PARAMS_df.diri_bc[0]; HINI = PARAMS_df.hini[0];
ET = PARAMS_df.ET[0]; TIMESTEP = PARAMS_df.timeStep[0]; KADJUST = PARAMS_df.Kadjust[0]

P = read.read_precipitation() # precipitation read from separate historical data
ET = ET * np.ones(shape=P.shape)


# Even if maps say peat depth is less than 2 meters, the impermeable bottom is at most at 2m.
# This can potentially break the hydrological simulation if the WTD would go below 2m.
print(">>>>> WARNING, OVERWRITING PEAT DEPTH")
peat_depth_arr[peat_depth_arr < 2.] = 2. 

# catchment mask: delimit the study area
catchment_mask = np.ones(shape=dem.shape, dtype=bool)
catchment_mask[np.where(dem<-10)] = False # -99999.0 is current value of dem for nodata points.

# 'peel' the dem. Dirichlet BC will be applied at the peel.
boundary_mask = utilities.peel_raster(dem, catchment_mask)
 
# after peeling, catchment_mask should only be the fruit:
catchment_mask[boundary_mask] = False

# soil types, soil physical properties and soil depth:
peat_type_masked = peat_type_arr * catchment_mask
peat_bottom_elevation = - peat_depth_arr * catchment_mask # meters with respect to dem surface. Should be negative!

# Load peatmap soil types' physical properties dictionary.
# Kadjust is hydraulic conductivity multiplier for sapric peat
h_to_tra_and_C_dict, K = hydro_utils.peat_map_interp_functions(Kadjust=KADJUST) 

# Transmissivity and storage are computed as: T(h) = T(h) - T(peat depth).
#  These quantities are the latter
tra_to_cut = hydro_utils.peat_map_h_to_tra(soil_type_mask=peat_type_masked,
                                           gwt=peat_bottom_elevation, h_to_tra_and_C_dict=h_to_tra_and_C_dict)
sto_to_cut = hydro_utils.peat_map_h_to_sto(soil_type_mask=peat_type_masked,
                                           gwt=peat_bottom_elevation, h_to_tra_and_C_dict=h_to_tra_and_C_dict)
sto_to_cut = sto_to_cut * catchment_mask.ravel()

# Water level in canals and list of pixels in canal network.
srfcanlist =[dem[coords] for coords in c_to_r_list]
n_canals = len(c_to_r_list)

oWTcanlist = [x - CANAL_WATER_LEVEL for x in srfcanlist]

hand_made_dams = True # compute performance of cherry-picked locations for dams.

"""
MonteCarlo
"""
for i in range(0,N_ITER):
    
    if i==0: #  random block configurations
        damLocation = np.random.randint(1, n_canals, N_BLOCKS).tolist() # Generate random kvector. 0 is not a good position in c_to_r_list
    else:
        prohibited_node_list = [i for i,_ in enumerate(oWTcanlist[1:]) if oWTcanlist[1:][i] < wt_canals[1:][i]]      # [1:] is to take the 0th element out of the loop
        candidate_node_list = np.array([e for e in range(1, n_canals) if e not in prohibited_node_list]) # remove 0 from the range of possible canals
        damLocation = np.random.choice(candidate_node_list, size=N_BLOCKS)

    if hand_made_dams:
        # HAND-MADE RULE OF DAM POSITIONS TO ADD:
        hand_picked_dams = (11170, 10237, 10514, 2932, 4794, 8921, 4785, 5837, 7300, 6868) # rule-based approach
        hand_picked_dams = [6959, 901, 945, 9337, 10089, 7627, 1637, 7863, 7148, 7138, 3450, 1466, 420, 4608, 4303, 6908, 9405, 8289, 7343, 2534, 9349, 6272, 8770, 2430, 2654, 6225, 11152, 118, 4013, 3381, 6804, 6614, 7840, 9839, 5627, 3819, 7971, 402, 6974, 7584, 3188, 8316, 1521, 856, 770, 6504, 707, 5478, 5512, 1732, 3635, 1902, 2912, 9220, 1496, 11003, 8371, 10393, 2293, 4901, 5892, 6110, 2118, 4485, 6379, 10300, 6451, 5619, 9871, 9502, 1737, 4368, 7290, 9071, 11222, 3085, 2013, 5226, 597, 5038]
        damLocation = hand_picked_dams
    
    wt_canals = utilities.place_dams(oWTcanlist, srfcanlist, BLOCK_HEIGHT, damLocation, CNM)
    """
    #########################################
                    HYDROLOGY
    #########################################
    """
    ny, nx = dem.shape
    dx = 1.; dy = 1. # metres per pixel  (Actually, pixel size is 100m x 100m, so all units have to be converted afterwards)
    
    boundary_arr = boundary_mask * (dem - DIRI_BC) # constant Dirichlet value in the boundaries
    
    ele = dem * catchment_mask
    
    # Get a pickled phi solution (not ele-phi!) computed before without blocks, independently,
    # and use it as initial condition to improve convergence time of the new solution
    retrieve_transient_phi_sol_from_pickled = False
    if retrieve_transient_phi_sol_from_pickled:
        with open(r"pickled/transient_phi_sol.pkl", 'r') as f:
            phi_ini = pickle.load(f)
        print("transient phi solution loaded as initial condition")
        
    else:
        phi_ini = ele + HINI #initial h (gwl) in the compartment.
        phi_ini = phi_ini * catchment_mask
           
    wt_canal_arr = np.zeros((ny,nx)) # (nx,ny) array with wt canal height in corresponding nodes
    for canaln, coords in enumerate(c_to_r_list):
        if canaln == 0: 
            continue # because c_to_r_list begins at 1
        wt_canal_arr[coords] = wt_canals[canaln] 
    
    
    dry_peat_volume, wt_track_drained, wt_track_notdrained, avg_wt_over_time = hydro.hydrology('transient', nx, ny, dx, dy, DAYS, ele, phi_ini, catchment_mask, wt_canal_arr, boundary_arr,
                                                      peat_type_mask=peat_type_masked, httd=h_to_tra_and_C_dict, tra_to_cut=tra_to_cut, sto_to_cut=sto_to_cut,
                                                      diri_bc=DIRI_BC, neumann_bc = None, plotOpt=True, remove_ponding_water=True,
                                                      P=P, ET=ET, dt=TIMESTEP)
    
    water_blocked_canals = sum(np.subtract(wt_canals[1:], oWTcanlist[1:]))
    
    cum_Vdp_nodams = 21088.453521509597 # Value of dry peat volume without any blocks, without any precipitation for 3 days. Normalization.
    print('dry_peat_volume(%) = ', dry_peat_volume/cum_Vdp_nodams * 100. , '\n',
          'water_blocked_canals = ', water_blocked_canals)

    """
    Final printings
    """
    fname = r'output/results_mc_3_cumulative.txt'
    if N_ITER > 20:  # only if big enough number of simulated days
        with open(fname, 'a') as output_file:
            output_file.write(
                                "\n" + str(i) + "    " + str(dry_peat_volume) + "    "
                                + str(N_BLOCKS) + "    " + str(N_ITER) + "    " + str(DAYS) + "    "
                                + str(time.ctime()) + "    " + str(water_blocked_canals)
                              )
"""
Save WTD data if simulating a year
"""
fname = r'output/wtd_year_' + str(N_BLOCKS) + '.txt'
if DAYS > 300:
   with open(fname, 'a') as output_file:
       output_file.write("\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" +
                             str(time.ctime()) + " nblocks = " + str(N_BLOCKS) + " ET = " + str(ET[0]) +
                             '\n' + 'drained notdrained mean'
                             )
       for i in range(len(wt_track_drained)): 
           output_file.write( "\n" + str(wt_track_drained[i]) + " " + str(wt_track_notdrained[i]) + " " + str(avg_wt_over_time[i]))

plt.figure()
plt.plot(list(range(0,DAYS)), wt_track_drained, label='close to drained')
plt.plot(list(range(0,DAYS)), wt_track_notdrained, label='away from drained')
plt.plot(list(range(0,DAYS)), avg_wt_over_time, label='average')
plt.xlabel('time(days)'); plt.ylabel('WTD (m)')
plt.legend()
plt.show()