# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:13:45 2018

@author: L1817
"""

import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

import preprocess_data,  utilities, hydro_steadystate, hydro_utils


plt.close("all")

"""
Read general help on main.README.txt
"""

time0 = time.time()

"""
Read and preprocess data
"""
retrieve_canalarr_from_pickled = False
preprocessed_datafolder = r"data"
dem_rst_fn = preprocessed_datafolder + r"\lidar_100_resampled_interp.tif"
can_rst_fn = preprocessed_datafolder + r"\canal_clipped_resampled_2.tif"
peat_type_rst_fn = preprocessed_datafolder + r"\Landcover_clipped.tif"

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

_ , dem, peat_type_arr = preprocess_data.read_preprocess_rasters(can_rst_fn, dem_rst_fn, peat_type_rst_fn)

print("rasters read and preprocessed from file")

# catchment mask
catchment_mask = np.ones(shape=dem.shape, dtype=bool)
catchment_mask[np.where(dem<-10)] = False # -99999.0 is current value of dem for nodata points.

# peel the dem. Only when dem is not surrounded by water
boundary_mask = utilities.peel_raster(dem, catchment_mask)
 

# after peeling, catchment_mask should only be the fruit:
catchment_mask[boundary_mask] = False

# soil types and soil physical properties:

peat_type_mask = peat_type_arr * catchment_mask

# Pad a couple of columns and rows to match dem. Only with older Winrock dataset. Eliminate in the future.
#peat_type_mask = np.ones(shape=(peat_type_mask_raw.shape[0]+2, peat_type_mask_raw.shape[1]+1)) * 255 
#peat_type_mask[2:,1:] = peat_type_mask_raw 
#peat_type_mask[peat_type_mask == 255] = 5 # Fill nodata values with something

h_to_tra_dict = hydro_utils.peat_map_interp_functions() # Load peatmap soil types' physical properties dictionary
#soiltypes[soiltypes==255] = 0 # 255 is nodata value. 1 is water (useful for hydrology! Maybe, same treatment as canals).

BOTTOM_ELE = -43.0 # meters with respect to sea level. Or at least to same common ref. point as the dem. Can be negative!
#peat_bottom_elevation = np.ones(shape=dem.shape) * BOTTOM_ELE - dem
peat_bottom_elevation = np.ones(shape=dem.shape) * BOTTOM_ELE
peat_bottom_elevation = peat_bottom_elevation*catchment_mask
tra_to_cut = hydro_utils.peat_map_h_to_tra(soil_type_mask=peat_type_mask,
                                           gwt=peat_bottom_elevation, h_to_tra_dict=h_to_tra_dict)


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
Metropolis/Simulated Annealing. Maybe use external function?
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



"""
Plots
    - Energy vs Temperature
    - WT in canals
"""
#MCplot = False
#if MCplot:
#    plottingARR.PlotFromData(MCfile, Tini, Tfin, MCsteps, n_nodes)
#
#WTcanals_plot = False
#if WTcanals_plot:
#    plottingARR.PlotWT(n_nodes, oWTcanArr, WT0canal, srfcanlist, bpos0, ktndict, titlename=1)
 

wt_canals = iWTcanlist

"""
#########################################
                HYDROLOGY
#########################################
"""
ny, nx = dem.shape
dx = 1.; dy = 1. # metres per pixel  
dt = 1. # timestep, in days
diri_bc = 1.2 # ele-phi in meters

hini = - 0.9 # initial wt wrt surface elevation in meters.

boundary_arr = boundary_mask * (dem - diri_bc) # constant Dirichlet value in the boundaries

# Clean dem... REVIEW
#dem[dem==-9999] = np.nan
#x = np.arange(0, nx)
#y = np.arange(0, ny)
#dem = np.ma.masked_invalid(dem)
#xx, yy = np.meshgrid(x, y)
#x1 = xx[~dem.mask] # get only the valid values
#y1 = yy[~dem.mask]
#clean_dem = dem[~dem.mask] 

ele = dem

Hinitial = ele + hini #initial h (gwl) in the compartment.
#depth = np.ones(shape=Hinitial.shape) # elevation of the impermeable bottom from comon ref. point

wt_canal_arr = np.zeros((ny,nx)) # (nx,ny) array with wt canal height in corresponding nodes
owt_canal_arr = np.zeros((ny,nx)) # checking purposes
for canaln, coords in enumerate(c_to_r_list):
    if canaln == 0: 
        continue # because c_to_r_list begins at 1
    wt_canal_arr[coords] = wt_canals[canaln] 
    owt_canal_arr[coords] = oWTcanlist[canaln]
    Hinitial[coords] = wt_canals[canaln]


dry_peat_volume, wt = hydro_steadystate.hydrology(nx, ny, dx, dy, dt, ele, Hinitial, catchment_mask, wt_canal_arr, boundary_arr,
                                                  peat_type_mask=peat_type_mask, httd=h_to_tra_dict, tra_to_cut=tra_to_cut,
                                                  diri_bc=0.9, neumann_bc = None, plotOpt=True, remove_ponding_water=True)


# Old and bad hydrology computation. Remove after finished with the checks
#dry_peat_volume = hydro.hydrology(nx, ny, dx, dy, dt, ele, Hinitial, catchment_mask, wt_canal_arr,
#                                  peat_type_mask=peat_type_mask, httd=h_to_tra_dict, tra_to_cut=tra_to_cut,
#                                  value_for_masked=0.9, diri_bc=None, neumann_bc = 0.0, plotOpt=True)


"""
Final printings
"""
timespent = time.time() - time0
utilities.print_time_in_mins(timespent)