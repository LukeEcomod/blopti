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
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable # for plots 
from mpl_toolkits.axes_grid1.colorbar import colorbar

import hydro_utils, utilities

"""
   SET FIPY SOLVER
"""
fp.solvers.DefaultSolver = fp.solvers.LinearLUSolver

def big_4_raster_plot(title, raster1, raster2, raster3, raster4):
        fig1, axes1 = plt.subplots(nrows=2, ncols=2, figsize=(16,12), dpi=80)
        fig1.suptitle(title)
        
        a = axes1[1,0].imshow(raster1, cmap='pink', interpolation='nearest')
        ax1_divider = make_axes_locatable(axes1[1,0])
        cax1 = ax1_divider.append_axes('right', size='7%', pad='2%')
        plt.colorbar(a, cax=cax1)
        axes1[1,0].set(title="D")
        
        b = axes1[0,1].imshow(raster2, cmap='viridis')
        ax2_divider = make_axes_locatable(axes1[0,1])
        cax2 = ax2_divider.append_axes('right', size='7%', pad='2%')
        plt.colorbar(b, cax=cax2)
        axes1[0,1].set(title="canal water level")
        
        c = axes1[0,0].imshow(raster3, cmap='viridis')
        ax3_divider = make_axes_locatable(axes1[0,0])
        cax3 = ax3_divider.append_axes('right', size='7%', pad='2%')
        plt.colorbar(c, cax=cax3)
        axes1[0,0].set(title="DEM")
        
        d = axes1[1,1].imshow(raster4, cmap='pink')
        ax4_divider = make_axes_locatable(axes1[1,1])
        cax4 = ax4_divider.append_axes('right', size='7%', pad='2%')
        plt.colorbar(d, cax=cax4)
        axes1[1,1].set(title="elevation - phi")


def plot_line_of_peat(raster, y_value, title, nx, ny, label, color, linewidth=1.):
    plt.figure(10)
    plt.title(title)
    plt.plot(raster[y_value,:], label=label, color=color, linewidth=linewidth)
    plt.legend(fontsize='x-large')
    
    return 0
        
    

def hydrology(solve_mode, nx, ny, dx, dy, days, ele, phi_initial, catchment_mask, wt_canal_arr, boundary_arr,
              peat_type_mask, httd, tra_to_cut, sto_to_cut, 
              diri_bc=0.0, neumann_bc = None, plotOpt=False, remove_ponding_water=True, P=0.0, ET=0.0, dt=1.0):
    """
    INPUT:
        - ele: (nx,ny) sized NumPy array. Elevation in m above c.r.p.
        - Hinitial: (nx,ny) sized NumPy array. Initial water table in m above c.r.p.
        - catchment mask: (nx,ny) sized NumPy array. Boolean array = True where the node is inside the computation area. False if outside.
        - wt_can_arr: (nx,ny) sized NumPy array. Zero everywhere except in nodes that contain canals, where = wl of the canal.
        - value_for_masked: DEPRECATED. IT IS NOW THE SAME AS diri_bc.
        - diri_bc: None or float. If None, Dirichlet BC will not be implemented. If float, this number will be the BC.
        - neumann_bc: None or float. If None, Neumann BC will not be implemented. If float, this is the value of grad phi.
        - P: Float. Constant precipitation. mm/day.
        - ET: Float. Constant evapotranspiration. mm/day.
    """
#    dneg = []
   
    track_WT_drained_area = (239,166)
    track_WT_notdrained_area = (522,190)
    
    ele[~catchment_mask] = 0.
    ele = ele.flatten()
    phi_initial = (phi_initial + 0.0 * np.zeros((ny,nx))) * catchment_mask
#    phi_initial = phi_initial * catchment_mask
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
    dr[np.array(wt_canal_arr, dtype=bool) * np.array(boundary_arr, dtype=bool)] = False # Pixels cannot be canals and boundaries at the same time. Everytime a conflict appears, boundaries win. This overwrites any canal water level info if the canal is in the boundary.
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
        d[d<0] = 1e-3 # Small non-zero value not to wreck the computation
        
        dcell = fp.CellVariable(mesh=mesh, value=d) # diffusion coefficient, transmissivity. As a cell variable.
        dface = fp.FaceVariable(mesh=mesh, value= dcell.arithmeticFaceValue.value) # THe correct Face variable.
        
        return dface.value
    
    def C_value(phi, ele, sto_to_cut, cmask, drmask_not):
        # Some inputs are in fipy CellVariable type
        gwt = phi.value*cmask.value - ele
        
        c = hydro_utils.peat_map_h_to_sto(soil_type_mask=peat_type_mask, gwt=gwt, h_to_tra_and_C_dict=httd) - sto_to_cut
        c[c<0] = 1e-3 # Same reasons as for D
        
        ccell = fp.CellVariable(mesh=mesh, value=c) # diffusion coefficient, transmissivity. As a cell variable.        
        return ccell.value

    D = fp.FaceVariable(mesh=mesh, value=D_value(phi, ele, tra_to_cut, cmask, drmask_not)) # THe correct Face variable.
    C = fp.CellVariable(mesh=mesh, value=C_value(phi, ele, sto_to_cut, cmask, drmask_not)) # differential water capacity
    
   
    largeValue=1e20                                                     # value needed in implicit source term to apply internal boundaries
    
    
    if plotOpt:
        big_4_raster_plot(title='Before the computation',
                raster1=((hydro_utils.peat_map_h_to_tra(soil_type_mask=peat_type_mask, gwt=(phi.value - ele), h_to_tra_and_C_dict=httd) - tra_to_cut)*cmask.value *drmask_not.value ).reshape(ny,nx),
                raster2=(ele.reshape(ny,nx) - wt_canal_arr) * dr * catchment_mask,
                raster3=ele.reshape(ny,nx),
                raster4=(ele-phi.value).reshape(ny,nx)
                )
        # for later cross-section plots
        y_value=270

        
#        print "first cross-section plot"
#        ele_with_can = copy.copy(ele).reshape(ny,nx)
#        ele_with_can = ele_with_can * catchment_mask
#        ele_with_can[wt_canal_arr > 0] = wt_canal_arr[wt_canal_arr > 0]
#        plot_line_of_peat(ele_with_can, y_value=y_value, title="cross-section", color='green', nx=nx, ny=ny, label="ele")
        


    # ********************************** PDE, STEADY STATE **********************************
    if solve_mode == 'steadystate':
        if diri_bc != None:
    #        diri_boundary = fp.CellVariable(mesh=mesh, value= np.ravel(diri_boundary_value(boundary_mask, ele2d, diri_bc)))
            
            eq = 0. == (fp.DiffusionTerm(coeff=D) 
                    + source*cmask*drmask_not 
                    - fp.ImplicitSourceTerm(cmask_not*largeValue) + cmask_not*largeValue*np.ravel(boundary_arr)
                    - fp.ImplicitSourceTerm(drmask*largeValue)    + drmask*largeValue*(np.ravel(wt_canal_arr))
#                    - fp.ImplicitSourceTerm(bmask_not*largeValue) + bmask_not*largeValue*(boundary_arr)
                    )
            
        elif neumann_bc != None: 
            raise NotImplementedError("Neumann BC not implemented yet!")
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
#                        - fp.ImplicitSourceTerm(bmask_not*largeValue) + bmask_not*largeValue*(boundary_arr)
                        )
        elif neumann_bc != None:
            raise NotImplementedError("Neumann BC not implemented yet!")
             
    
    #********************************************************
                                                       
    max_sweeps = 10 # inner loop.
    
    avg_wt = []
    wt_track_drained = []
    wt_track_notdrained = []
    
    cumulative_Vdp = 0.
                                                             
    #********Finite volume computation******************
    for d in range(days):
        
        if type(P) == type(ele): # assume it is a numpy array
            source.setValue((P[d]-ET[d])* .001 *np.ones(ny*nx))                         # source/sink, in mm/day. The factor of 10^-3 takes into account that there are 100 x 100 m^2 in one pixel
            print "(d,P) = ", (d, (P[d]-ET[d])* 10.)
        else:
            source.setValue((P-ET)* 10. *np.ones(ny*nx))
            print "(d,P) = ", (d, (P-ET)* 10.)
        
        if plotOpt and d!= 0:
           # print "one more cross-section plot"
           plot_line_of_peat(phi.value.reshape(ny,nx), y_value=y_value, title="cross-section", color='cornflowerblue', nx=nx, ny=ny, label=d)

        
        res = 0.0
        
        phi.updateOld() 
     
        D.setValue(D_value(phi, ele, tra_to_cut, cmask, drmask_not))
        C.setValue(C_value(phi, ele, tra_to_cut, cmask, drmask_not))
            
        for r in range(max_sweeps):
            resOld=res
                
            res = eq.sweep(var=phi, dt=dt) # solve linearization of PDE


            #print "sum of Ds: ", np.sum(D.value)/1e8 
            #print "average wt: ", np.average(phi.value-ele)
            
            #print 'residue diference:    ', res - resOld                
            
            
            if abs(res - resOld) < 1e-7: break # it has reached to the solution of the linear system
            
        if solve_mode=='transient': #solving in steadystate will remove water only at the very end
            if remove_ponding_water:                 
                s=np.where(phi.value>ele,ele,phi.value)                        # remove the surface water. This also removes phi in those masked values (for the plot only)
                phi.setValue(s)                                                # set new values for water table


        
        if (D.value<0.).any():
                print "Some value in D is negative!"

        # For some plots
        avg_wt.append(np.average(phi.value-ele))
        wt_track_drained.append((phi.value - ele).reshape(ny,nx)[track_WT_drained_area])
        wt_track_notdrained.append((phi.value - ele).reshape(ny,nx)[track_WT_notdrained_area])
    
            
        """ Volume of dry peat calc."""
        not_peat = np.ones(shape=peat_type_mask.shape) # Not all soil is peat!
        not_peat[peat_type_mask == 4] = 0 # NotPeat
        not_peat[peat_type_mask == 5] = 0 # OpenWater
        peat_vol_weights = utilities.PeatV_weight_calc(np.array(~dr * catchment_mask * not_peat, dtype=int))
        dry_peat_volume = utilities.PeatVolume(peat_vol_weights, (ele-phi.value).reshape(ny,nx))
        cumulative_Vdp = cumulative_Vdp + dry_peat_volume
        print "avg_wt  = ", np.average(phi.value-ele)
#        print "wt drained = ", (phi.value - ele).reshape(ny,nx)[track_WT_drained_area]
#        print "wt not drained = ", (phi.value - ele).reshape(ny,nx)[track_WT_notdrained_area]
        print "Cumulative vdp = ", cumulative_Vdp
        
    if solve_mode=='steadystate': #solving in steadystate we remove water only at the very end
        if remove_ponding_water:                 
            s=np.where(phi.value>ele,ele,phi.value)                        # remove the surface water. This also removes phi in those masked values (for the plot only)
            phi.setValue(s)                                                # set new values for water table
    
       
       
        # Areas with WT <-1.0; areas with WT >0
#        plot_raster_by_value((ele-phi.value).reshape(ny,nx), title="ele-phi in the end, colour keys", bottom_value=0.5, top_value=0.01)
        
        
        
    
    if plotOpt:
        big_4_raster_plot(title='After the computation',
            raster1=((hydro_utils.peat_map_h_to_tra(soil_type_mask=peat_type_mask, gwt=(phi.value - ele), h_to_tra_and_C_dict=httd) - tra_to_cut)*cmask.value *drmask_not.value ).reshape(ny,nx),
            raster2=(ele.reshape(ny,nx) - wt_canal_arr) * dr * catchment_mask,
            raster3=ele.reshape(ny,nx),
            raster4=(ele-phi.value).reshape(ny,nx)
            )
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,9), dpi=80)
        x = np.arange(-21,1,0.1)
        axes[0,0].plot(httd[1]['hToTra'](x),x)
        axes[0,0].set(title='hToTra', ylabel='depth')
        axes[0,1].plot(httd[1]['C'](x),x)
        axes[0,1].set(title='C')
        axes[1,0].plot()
        axes[1,0].set(title="Nothing")
        axes[1,1].plot(avg_wt)
        axes[1,1].set(title="avg_wt_over_time")
        
        # plot surface in cross-section
        ele_with_can = copy.copy(ele).reshape(ny,nx)
        ele_with_can = ele_with_can * catchment_mask
        ele_with_can[wt_canal_arr > 0] = wt_canal_arr[wt_canal_arr > 0]
        plot_line_of_peat(ele_with_can, y_value=y_value, title="cross-section", nx=nx, ny=ny, label="surface", color='peru', linewidth=2.0)
    
        plt.show()
        
#    change_in_canals = (ele-phi.value).reshape(ny,nx)*(drmask.value.reshape(ny,nx)) - ((ele-H)*drmask.value).reshape(ny,nx)
#    resulting_phi = phi.value.reshape(ny,nx)


    return cumulative_Vdp, wt_track_drained, wt_track_notdrained, avg_wt