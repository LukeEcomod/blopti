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
parser.add_argument('-b','--nblocks', default=5, help='(int) Number of blocks to locate. Default=5.', type=int)
parser.add_argument('-n','--niter', default=2, help='(int) Number of repetitions of the whole computation. Default=10', type=int)
args = parser.parse_args()

DAYS = args.days
N_BLOCKS = args.nblocks
N_ITER = args.niter


"""
Read and preprocess data
"""
retrieve_canalarr_from_pickled = False

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
peat_type_rst_fn = preprocessed_datafolder + r"/Landcover2017_clip.tif"
peat_depth_rst_fn = preprocessed_datafolder + r"/Peattypedepth_clip.tif"


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


print(">>>>> WARNING, OVERWRITING PEAT DEPTH")
peat_depth_arr[peat_depth_arr < 2.] = 2.

print("rasters read and preprocessed from file")

# catchment mask
catchment_mask = np.ones(shape=dem.shape, dtype=bool)
catchment_mask[np.where(dem<-10)] = False # -99999.0 is current value of dem for nodata points.

# soil types and soil physical properties and soil depth:
peat_type_mask = peat_type_arr * catchment_mask
peat_bottom_elevation = - peat_depth_arr * catchment_mask # meters with respect to dem surface. Should be negative!
dem = dem * catchment_mask

# peel the dem. Only when dem is not surrounded by water
boundary_mask = utilities.peel_raster(dem, catchment_mask)
 
# after peeling, catchment_mask should only be the fruit:
catchment_mask[boundary_mask] = False

h_to_tra_and_C_dict, K = hydro_utils.peat_map_interp_functions() # Load peatmap soil types' physical properties dictionary

# Plot K
import matplotlib.pyplot as plt
plt.figure(); z = np.linspace(0.0, -20.0, 400); plt.plot(K,z); plt.title('K')
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
print('>>>> WARNING! BLOCK HEIGHT SHOULD BE = 0.4 TO COMPARE WITH OPTIMISATION!')
block_height = 0.1 # water level of canal after placing dam.

# HANDCRAFTED WATER LEVEL IN CANALS. CHANGE WITH MEASURED, IDEALLY.
canal_water_level = 1.2
oWTcanlist = [x - canal_water_level for x in srfcanlist]


"""
MonteCarlo
"""
for i in range(0,N_ITER):
    
    damLocation = np.random.randint(1, n_canals, N_BLOCKS).tolist() # Generate random kvector. 0 is not a good position in c_to_r_list
    wt_canals = utilities.place_dams(oWTcanlist, srfcanlist, block_height, damLocation, CNM)
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
    
    ele = dem[:]
    
    # Get a pickled phi solution (not ele-phi!) computed before without blocks, independently,
    # and use it as initial condition to improve convergence time of the new solution
    retrieve_transient_phi_sol_from_pickled = False
    if retrieve_transient_phi_sol_from_pickled:
        with open(r"pickled/transient_phi_sol.pkl", 'r') as f:
            phi_ini = pickle.load(f)
        print "transient phi solution loaded as initial condition"
        
    else:
        phi_ini = ele + hini #initial h (gwl) in the compartment.
        phi_ini = phi_ini * catchment_mask
           
    wt_canal_arr = np.zeros((ny,nx)) # (nx,ny) array with wt canal height in corresponding nodes
    for canaln, coords in enumerate(c_to_r_list):
        if canaln == 0: 
            continue # because c_to_r_list begins at 1
        wt_canal_arr[coords] = wt_canals[canaln] 
    
    
    dry_peat_volume = hydro.hydrology('transient', nx, ny, dx, dy, DAYS, ele, phi_ini, catchment_mask, wt_canal_arr, boundary_arr,
                                                      peat_type_mask=peat_type_mask, httd=h_to_tra_and_C_dict, tra_to_cut=tra_to_cut, sto_to_cut=sto_to_cut,
                                                      diri_bc=diri_bc, neumann_bc = None, plotOpt=False, remove_ponding_water=True)
    """
    Final printings
    """
    with open(r'output/results_mc_2.txt', 'a') as output_file:
        output_file.write("\n" + str(i) + "    " + str(dry_peat_volume) + "    " + str(N_BLOCKS) + "    " + str(N_ITER) + "    " + str(DAYS) + "    " + str(time.ctime()))
