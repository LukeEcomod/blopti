# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:42:36 2019

@author: L1817
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:34:44 2018

@author: L1817
"""
import numpy as np
import fipy as fp
import matplotlib.pyplot as plt
import copy

import hydro_utils, utilities

"""
   SET FIPY SOLVER
"""
fp.solvers.DefaultSolver = fp.solvers.LinearLUSolver

def plot_2D_raster(raster, title, colormap='pink'):
    """
    raster must be converted to (ny, nx) form in advance. 
    """
    plt.figure()
    plt.title(title)
    plt.imshow(raster, cmap=colormap, interpolation='nearest', animated='True')
    plt.colorbar()
    
    return 0

def plot_raster_by_value(raster, title, bottom_value=None, top_value=None):
    # No differences if unspecified bottom and top
    if bottom_value == None:
        bottom_value = raster.min()
    if top_value == None:
        top_value = raster.max()
        
    raster_discrete = np.zeros(shape=raster.shape)
    raster_discrete = raster_discrete + np.array(raster <= bottom_value, dtype=int) * (-1)
    raster_discrete = raster_discrete + np.array(raster >= top_value, dtype=int) * 1
    
    plt.figure()
    plt.title(title)
    plt.imshow(raster_discrete, cmap='Reds', interpolation='nearest')
    plt.colorbar()

    
    return 0

def plot_line_of_peat(raster, y_value, title, nx, ny, label):
    plt.figure(10)
    plt.title(title)
    plt.plot(raster[y_value,:], label=label)
    plt.legend()
    
    return 0
        
    

def hydrology(solve_mode, nx, ny, dx, dy, dt, ele, phi_initial, catchment_mask, wt_canal_arr, boundary_arr,
              peat_type_mask, httd, tra_to_cut, sto_to_cut, 
              diri_bc=0.9, neumann_bc = None, plotOpt=False, remove_ponding_water=True):
    """
    INPUT:
        - ele: (nx,ny) sized NumPy array. Elevation in m above c.r.p.
        - Hinitial: (nx,ny) sized NumPy array. Initial water table in m above c.r.p.
        - catchment mask: (nx,ny) sized NumPy array. Boolean array = True where the node is inside the computation area. False if outside.
        - wt_can_arr: (nx,ny) sized NumPy array. Zero everywhere except in nodes that contain canals, where = wl of the canal.
        - value_for_masked: DEPRECATED. IT IS NOW THE SAME AS diri_bc.
        - diri_bc: None or float. If None, Dirichlet BC will not be implemented. If float, this number will be the BC.
        - neumann_bc: None or float. If None, Neumann BC will not be implemented. If float, this is the value of grad phi.
    """
    dneg = []
   
    ele[~catchment_mask] = 0.
    ele = ele.flatten()
    phi_initial = phi_initial.flatten()

    if len(ele)!= nx*ny or len(phi_initial) != nx*ny:
        raise ValueError("ele or Hinitial are not of dim nx*ny")
    
    mesh = fp.Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
    phi = fp.CellVariable(name='computed H', mesh=mesh,value=phi_initial, hasOld=True) #response variable H in meters above reference level               
    
    if diri_bc != None and neumann_bc == None:
        phi.constrain(diri_bc, mesh.exteriorFaces)
    
    elif diri_bc == None and neumann_bc != None:
        phi.faceGrad.constrain(neumann_bc * mesh.faceNormals, where=mesh.exteriorFaces)
       
    else:
        raise ValueError("Cannot apply Dirichlet and Neumann boundary values at the same time. Contradictory values.")
       
    
    #*******omit areas outside the catchment. c is input
    cmask = fp.CellVariable(mesh=mesh, value=np.ravel(catchment_mask))
    cmask_not = fp.CellVariable(mesh=mesh, value=np.array(~ cmask.value, dtype = int))
    
    # *** drain mask or canal mask
    dr = np.array(wt_canal_arr, dtype=bool)
    drmask=fp.CellVariable(mesh=mesh, value=np.ravel(dr))
    drmask_not = fp.CellVariable(mesh=mesh, value= np.array(~ drmask.value, dtype = int))      # Complementary of the drains mask, but with ints {0,1}
    
    
    # mask away unnecesary stuff
#    phi.setValue(np.ravel(H)*cmask.value)
#    ele = ele * cmask.value
    
    source = fp.CellVariable(mesh=mesh, value = 0.)                         # cell variable for source/sink
#    CC=fp.CellVariable(mesh=mesh, value=C(phi.value-ele))                   # differential water capacity


    def D_value(phi, ele, tra_to_cut, cmask, drmask_not):
        # Some inputs are in fipy CellVariable type
        gwt = phi.value*cmask.value - ele
        
        d = hydro_utils.peat_map_h_to_tra(soil_type_mask=peat_type_mask, gwt=gwt, h_to_tra_and_C_dict=httd) - tra_to_cut

        # d <0 means tra_to_cut is greater than the other transmissivity, which in turn means that
        # phi is below the impermeable bottom. We allow phi to have those values, but
        # the transmissivity is in those points is equal to zero (as if phi was exactly at the impermeable bottom).
#        d[d<0] = 1e-5 
        
        dcell = fp.CellVariable(mesh=mesh, value=d) # diffusion coefficient, transmissivity. As a cell variable.
        dface = fp.FaceVariable(mesh=mesh, value= dcell.arithmeticFaceValue.value) # THe correct Face variable.
        
        return dface.value
    
    def C_value(phi, ele, sto_to_cut, cmask, drmask_not):
        # Some inputs are in fipy CellVariable type
        gwt = phi.value*cmask.value - ele
        
        c = hydro_utils.peat_map_h_to_sto(soil_type_mask=peat_type_mask, gwt=gwt, h_to_tra_and_C_dict=httd) - sto_to_cut
        
        ccell = fp.CellVariable(mesh=mesh, value=c) # diffusion coefficient, transmissivity. As a cell variable.        
        return ccell.value

    D = fp.FaceVariable(mesh=mesh, value=D_value(phi, ele, tra_to_cut, cmask, drmask_not)) # THe correct Face variable.
    C = fp.CellVariable(mesh=mesh, value=C_value(phi, ele, sto_to_cut, cmask, drmask_not)) # differential water capacity
        
    largeValue=1e20                                                     # value needed in implicit source term to apply internal boundaries
    
    
    if plotOpt:
        plot_2D_raster(((hydro_utils.peat_map_h_to_tra(soil_type_mask=peat_type_mask, gwt=(phi.value - ele), h_to_tra_and_C_dict=httd) - tra_to_cut)*cmask.value *drmask_not.value ).reshape(ny,nx), title="D in the beginning")
        plot_2D_raster(phi.value.reshape(ny,nx), title="phi initial state")
        plot_2D_raster((ele.reshape(ny,nx) - wt_canal_arr) * dr * catchment_mask, title="canal water level", colormap='viridis')
        plot_2D_raster(ele.reshape(ny,nx), title="DEM")
        plot_2D_raster((ele-phi.value).reshape(ny,nx), title="elevation - phi Initial state")
    plotOptCrossSection = True
    if plotOptCrossSection:
        y_value=170
        print "first cross-section plot"
        ele_with_can = copy.copy(ele).reshape(ny,nx)
        ele_with_can[wt_canal_arr > 0] = wt_canal_arr[wt_canal_arr > 0]
        plot_line_of_peat(ele_with_can, y_value=y_value, title="cross-section", nx=nx, ny=ny, label="ele")
        


    # ********************************** PDE, STEADY STATE **********************************
    if solve_mode == 'steadystate':
        if diri_bc != None:
    #        diri_boundary = fp.CellVariable(mesh=mesh, value= np.ravel(diri_boundary_value(boundary_mask, ele2d, diri_bc)))
            
            eq = 0. == (fp.DiffusionTerm(coeff=D) 
                    + source*cmask*drmask_not 
                    - fp.ImplicitSourceTerm(cmask_not*largeValue) + cmask_not*largeValue*np.ravel(boundary_arr)
                    - fp.ImplicitSourceTerm(drmask*largeValue)    + drmask*largeValue*(np.ravel(wt_canal_arr))
    #                - fp.ImplicitSourceTerm(bmask_not*largeValue) + bmask_not*largeValue*(boundary_arr)
                    )
            
        elif neumann_bc != None: 
            raise NotImplementedError("Neumann BC not implemented yet!") # DOESN'T WORK RIGHT NOW!
            cmask_face = fp.FaceVariable(mesh=mesh, value=np.array(cmask.arithmeticFaceValue.value, dtype=bool))
            D[cmask_face.value] = 0.
            eq = 0. == (fp.DiffusionTerm(coeff=D) + source*cmask*drmask_not
                    - fp.ImplicitSourceTerm(cmask_not*largeValue) + cmask_not*largeValue*(diri_bc)
                    - fp.ImplicitSourceTerm(drmask*largeValue) + drmask*largeValue*(np.ravel(wt_canal_arr))
    #                + fp.DiffusionTerm(coeff=largeValue * bmask_face)
    #                - fp.ImplicitSourceTerm((bmask_face * largeValue *neumann_bc * mesh.faceNormals).divergence)
                    )
    
    elif solve_mode == 'transient':
        if diri_bc != None:
    #        diri_boundary = fp.CellVariable(mesh=mesh, value= np.ravel(diri_boundary_value(boundary_mask, ele2d, diri_bc)))
            
            eq = fp.TransientTerm(coeff=C) == (fp.DiffusionTerm(coeff=D) 
                        + source*cmask*drmask_not 
                        - fp.ImplicitSourceTerm(cmask_not*largeValue) + cmask_not*largeValue*np.ravel(boundary_arr)
                        - fp.ImplicitSourceTerm(drmask*largeValue)    + drmask*largeValue*(np.ravel(wt_canal_arr))
        #                - fp.ImplicitSourceTerm(bmask_not*largeValue) + bmask_not*largeValue*(boundary_arr)
                        )
        elif neumann_bc != None:
            raise NotImplementedError("Neumann BC not implemented yet!") # DOESN'T WORK RIGHT NOW!
        
    
                  
    
    #********************************************************
                                                                  
    d=0   # day counter
    timeStep = 1.                                                            
    days=10 # outmost loop. "timesteps" in fipy manual. Needed due to non-linearity.
    max_sweeps = 1 # inner loop.
    ET = 0. # constant evapotranspoiration mm/day
    P = 6.0 # constant precipitation

#    source.setValue((P-ET)/1000.*np.ones(ny*nx))                         # source/sink, in m. For steadystate!
    source.setValue((P-ET)/1000.*np.ones(ny*nx))                         # source/sink, in m. For steadystate!

    avg_wt_over_time = []
    avg_D_over_time = []
    
    
                                                             
    #********Finite volume computation******************
    for d in range(days):
#        if d>2:
#            timeStep = 5.
#        if d > 20:
#            timeStep = 3.
#        if d > 50:
#            timeStep = 2.
#        if d>100:
#            timeStep = 1.
        print "timeStep = ", timeStep
        print d
        
        plotOptCrossSection = True
        if plotOptCrossSection:
            print "one more cross-section plot"
            plot_line_of_peat(phi.value.reshape(ny,nx), y_value=y_value, title="cross-section",  nx=nx, ny=ny, label=d)

        
        res = 0.0
        
        phi.updateOld() 
     
        D.setValue(D_value(phi, ele, tra_to_cut, cmask, drmask_not))
        C.setValue(C_value(phi, ele, tra_to_cut, cmask, drmask_not))
#        D.setValue(100.)
#            CC.setValue(C(phi.value-ele))    
            
        for r in range(max_sweeps):
            resOld=res
                
            res = eq.sweep(var=phi, dt=timeStep) # solve linearization of PDE


            print "sum of Ds: ", np.sum(D.value)/1e8 
            print "average wt: ", np.average(phi.value-ele)
            
            print 'residue diference:    ', res - resOld                
            
            
            if abs(res - resOld) < 1e-7: break # it has reached to the solution of the linear system
        
        if (D.value<0.).any():
                print "Some value in D is negative!"

        # For some plots
        avg_wt_over_time.append(np.average(phi.value-ele))
        avg_D_over_time.append(np.average(D.value))
        

        if remove_ponding_water:                 
            s=np.where(phi.value>ele,ele,phi.value)                        # remove the surface water. This also removes phi in those masked values (for the plot only)
            phi.setValue(s)                                                # set new values for water table
    
    
    """ Volume of dry peat calc."""
    peat_vol_weights = utilities.PeatV_weight_calc(np.array(~dr*catchment_mask,dtype=int))
    dry_peat_volume = utilities.PeatVolume(peat_vol_weights, (ele-phi.value).reshape(ny,nx))
#    print "Dry peat volume = ", dry_peat_volume
    
    if plotOpt:
        plot_2D_raster(((hydro_utils.peat_map_h_to_tra(soil_type_mask=peat_type_mask, gwt=(phi.value - ele), h_to_tra_and_C_dict=httd) - tra_to_cut)*cmask.value *drmask_not.value ).reshape(ny,nx), title="D in the end")
        plot_2D_raster((ele-phi.value).reshape(ny,nx), title="elevation - phi Final state")
        plot_2D_raster((phi.value).reshape(ny,nx), title="phi Final state")
        plot_2D_raster((ele).reshape(ny,nx), title="elevation Final state")
        
       
        # Areas with WT <-1.0; areas with WT >0
#        plot_raster_by_value((ele-phi.value).reshape(ny,nx), title="ele-phi in the end, colour keys", bottom_value=0.5, top_value=0.01)
        
        
        plt.show()
    
    plt.figure()
    plt.plot(avg_D_over_time)
    plt.title("avg D over time")
    plt.figure()
    plt.plot(avg_wt_over_time)
    plt.title("avg_wt_over_time")
    plt.show()
        
#    change_in_canals = (ele-phi.value).reshape(ny,nx)*(drmask.value.reshape(ny,nx)) - ((ele-H)*drmask.value).reshape(ny,nx)
    resulting_phi = phi.value.reshape(ny,nx)


    return dry_peat_volume, resulting_phi, dneg
