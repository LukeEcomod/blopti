# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:34:44 2018

@author: L1817
"""
import numpy as np
import fipy as fp
import matplotlib.pyplot as plt
import datetime
import hydro_utils, utilities

fp.solvers.DefaultSolver = fp.solvers.LinearLUSolver

def hydrology(nx, ny, dx, dy, dt, ele, Hinitial, catchment_mask, wt_canal_arr,
              value_for_masked=0.0, diri_bc=0.2, neumann_bc = None, plotOpt=False):
    """
    INPUT:
        - ele: (nx,ny) sized NumPy array. Elevation in m above c.r.p.
        - Hinitial: (nx,ny) sized NumPy array. Initial water table in m above c.r.p.
        - catchment mask: (nx,ny) sized NumPy array. Boolean array = True where the node is inside the computation area. False if outside.
        - wt_can_arr: (nx,ny) sized NumPy array. Zero everywhere except in nodes that contain canals, where = wl of the canal.
        - value_for_masked: float. Value of ele-phi for points outside the catchment mask.
        - diri_bc: None or float. If None, Dirichlet BC will not be implemented. If float, this number will be the BC.
        - neumann_bc: None or float. If None, Neumann BC will not be implemented. If float, this is the value of grad phi.
    """
    
    ele = ele.flatten()
    H = Hinitial.flatten()

    if len(ele)!= nx*ny or len(H) != nx*ny:
        raise ValueError("ele, depth or Hinitial are not of dim nx*ny")
   
    rainFile = r'C:\Users\L1817\Dropbox\PhD\Computation\hydro to Inaki\\rainfall.csv'
    rain = hydro_utils.getRainfall(rainFile)   #rainfall data
    
    # Added 19.10.2018
    spara ={ 
    'mesic':{
    'nLyrs':400, 'dzLyr': 0.05,  'Kadjust':50.0,
    'peat type':['L','L','L','L','L','L','L'], 'peat type bottom':['S'],
    'vonP top': [5,5,6,6,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8], 'vonP bottom': 10 },
    } 
    
    spara = spara['mesic']
    nLyrs = spara['nLyrs']                                                      # number of soil layers
    dz = np.ones(nLyrs)*spara['dzLyr']                                          # thickness of layers, m
    z = np.cumsum(dz)-dz/2.                                                     # depth of the layer center point, m 
    lenvp=len(spara['vonP top'])    
    vonP = np.ones(nLyrs)*spara['vonP bottom']; vonP[0:lenvp] = spara['vonP top']  # degree of  decomposition, von Post scale
    ptype = spara['peat type bottom']*spara['nLyrs']
    lenpt = len(spara['peat type']); ptype[0:lenpt] = spara['peat type']    
    pF, Ksat = hydro_utils.peat_hydrol_properties(vonP, var='H', ptype=ptype)  # peat hydraulic properties after Päivänen 1973    
    hToSto, stoToGwl, hToTra, C = hydro_utils.CWTr(nLyrs, z, dz, pF, Ksat*spara['Kadjust'], direction='negative') # interpolated storage, transmissivity and diff water capacity functions
    
    #        print hToTra([-5.,0.1,1.,5.,10.])
    #        import sys; sys.exit()
    #        #********* some checks
    #        if not np.all((ele-depth)*np.ravel(c))>=0):
    #            raise ValueError('depth cannot be greater than elevation!')
    #        if not np.all((H-depth)*np.ravel(c))>=0):
    #            raise ValueError('depth cannot be greater than H!')
    
    
    mesh = fp.Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
    phi=fp.CellVariable(name='computed H', mesh=mesh,value=1., hasOld=True) #response variable H in meters above reference level               
    
    #*******omit areas outside the catchment. catchment_mask is input
    cmask = fp.CellVariable(mesh=mesh, value=np.ravel(catchment_mask))
    cmask_not = fp.CellVariable(mesh=mesh, value=np.array(~ cmask.value, dtype = bool))
        #        cmask = fp.CellVariable(mesh=mesh, value=np.array(cmask.value, dtype = int))
    # *** drain mask or canal mask
    dr = np.array(wt_canal_arr, dtype=bool)
    drmask=fp.CellVariable(mesh=mesh, value=np.ravel(dr))
    drmask_not = fp.CellVariable(mesh=mesh, value= np.array(~ drmask.value, dtype = int))      # Complementary of the drains mask, but with ints {0,1}
    
    # mask away unnecesary stuff
    phi.setValue(np.ravel(H)*cmask.value)
    ele = ele * cmask.value
    
    #Added 19.10.2018
    bottom_ele = np.ones(np.shape(ele))*14.                                # elevation of the impearmeable bottom layer, m above sea level
    peat_depth=ele-bottom_ele
    to_cut_Tr = z[-1]-peat_depth                                           # depth of peat to be cut from the bottom of each column because of the varying peat depth
    Tr_cut = to_cut_Tr * Ksat[-1]*86400.*spara['Kadjust']*to_cut_Tr        #subtract this transmissivity from the computed to account for decreasing Tr with decreasing peat depth
    
    if diri_bc != None and neumann_bc == None:
        phi.constrain(diri_bc, mesh.exteriorFaces)
    
    elif diri_bc == None and neumann_bc != None:
       phi.faceGrad.constrain([-neumann_bc], mesh.facesRight)
       phi.faceGrad.constrain([-neumann_bc], mesh.facesTop) 
       phi.faceGrad.constrain([+neumann_bc], mesh.facesBottom)
       phi.faceGrad.constrain([+neumann_bc], mesh.facesLeft)

       
    else:
        raise ValueError("Cannot apply Dirichlet and Neumann boundary values at the same time. Contradictory values.")
       
    
    source = fp.CellVariable(mesh=mesh, value = 0.)                         # cell variable for source/sink
    dd = fp.CellVariable(mesh=mesh, value=(hToTra(phi.value-ele)-Tr_cut)*cmask*drmask_not)      # diffusion coefficient, transmissivity
    D = fp.FaceVariable(mesh=mesh, value = dd.arithmeticFaceValue.value)
#    CC=fp.CellVariable(mesh=mesh, value=C(phi.value-ele))                   # differential water capacity
    
    
    
    largeValue=1e20                                                     # variable needed in implicit source term to apply internal boundaries
    

    
    

    if plotOpt:
        plt.figure()
        plt.title("Transmissivity in the beginning")
        plt.imshow((dd.value).reshape(ny,nx), cmap='pink'); plt.colorbar()
        
#        plt.figure()
#        plt.title("C in the beginning")
#        plt.imshow((CC.value).reshape(ny,nx), cmap='pink', interpolation='nearest')
        
        plt.figure()
        plt.title("(DEM) elevation Initial state")
#        plt.imshow(ele.reshape(ny,nx), cmap='pink', interpolation='nearest', extent=[0,nx*dx,0,ny*dy]); plt.colorbar()
        plt.imshow(ele.reshape(ny,nx), cmap='pink', interpolation='nearest'); plt.colorbar()
        
        plt.figure()
        plt.title("elevation - phi Initial state")
#        plt.imshow((ele-phi.value).reshape(ny,nx), cmap='pink', interpolation='nearest', extent=[0,nx*dx,0,ny*dy]); plt.colorbar()
        plt.imshow((ele-phi.value).reshape(ny,nx), cmap='pink', interpolation='nearest'); plt.colorbar()
        # For some later plot
    
    
    steady_state = True
    if not steady_state:        
#        eq = fp.TransientTerm(coeff=CC) == (fp.DiffusionTerm(coeff=hToTra(phi.value-ele)*cmask*drmask_not) + source*cmask*drmask_not
#                          - fp.ImplicitSourceTerm(drmask*largeValue) + drmask*largeValue*(np.ravel(wt_canal_arr))
#                          - fp.ImplicitSourceTerm(cmask_not*largeValue) + cmask_not*largeValue*(value_for_masked)
#                          )
        eq = fp.TransientTerm(coeff=CC) == (fp.DiffusionTerm(coeff=hToTra(phi.value-ele)*cmask) + source*cmask
                              - fp.ImplicitSourceTerm(drmask*largeValue) + drmask*largeValue*(np.ravel(wt_canal_arr))
                              - fp.ImplicitSourceTerm(cmask_not*largeValue) + cmask_not*largeValue*(value_for_masked)
                              )
    elif steady_state:
        eq = 0. == (fp.DiffusionTerm(coeff=D) + source*cmask*drmask_not
                - fp.ImplicitSourceTerm(drmask*largeValue) + drmask*largeValue*(np.ravel(wt_canal_arr))
                - fp.ImplicitSourceTerm(cmask_not*largeValue) + cmask_not*largeValue*(value_for_masked)
                )
    #********************************************************
    yr0 = 2013; yr1 = 2013
    P= list(np.ravel(rain[str(yr0):str(yr1)].values))            
    ET = 3.5                                                                # constant evapotranspoiration mm/day
    d=0                                                                     # day counter
    days=3
    hts = np.empty((int(len(P)/dt),ny,nx), dtype=float)
    Nflag=False                                                             # Neumann boundary condition flag, false=Dirichlet, true=Neumann 
    #********Finite volume computation******************
    for p in P[0:days]:
    #for p in P:
        if p<3.: 
            dt= 1./4.
        elif p<10.: 
            dt=1./12.
        else:
            dt=1./24.
        
        dt=1.; p=7.5   #ATTN! This is manufactured, 7 is average daily rainfall
    
#        print d, p, dt
        
        for subdt in range(int(1./dt)):
#            print subdt, np.average(phi.value-ele)
            source.setValue((p-ET)/1000.*np.ones(ny*nx)*dt)                         # source/sink, in m
#            print 'source', min(source.value), max(source.value)
            res = 1e+10; resOld=1e+10
            phi.updateOld()                
            dd.setValue((hToTra(phi.value-ele)-Tr_cut)*cmask*drmask_not)
            D.setValue(dd.arithmeticFaceValue.value)
#            CC.setValue(C(phi.value-ele))    
            
            for r in range(100):
                resOld=res                
#                res = eq.sweep(var=phi,dt=dt)
                res = eq.sweep(var=phi)
                dd.setValue((hToTra(phi.value-ele)-Tr_cut)*cmask*drmask_not)
                D.setValue(dd.arithmeticFaceValue.value)
#                print '    ', r, res                
                if res < 1e-7: break
                if res>=resOld: break    
            #**********update setup**************            
    #        if midFieldDrains==True:            
    #            surr= adjacent_mean(phi.value, ny, nx, idxdrains)          # water table in the surrounding cells of a drain cell
    #            bmask = np.zeros(np.shape(np.ravel(m)),dtype=bool)         # new drain mask
    #            bmask[idxdrains]= surr-ele[idxdrains]>ddepth               # set false if wt in surrounding < water level in drain
    #            drmask.setValue(bmask)                                     # update the drain mask           
            s=np.where(phi.value>ele,ele,phi.value)                        # remove the surface water
            phi.setValue(s)                                                # set new values for water table
            
    #        fN=Nflag            
    #        Nflag = False if min(phi.value) > H else True
    #        if fN != Nflag:
    #            if min(phi.value) > H:
    #                print 'Changing boundary conditions -> Dirichlet'
    #                phi.constrain(H, mesh.facesLeft)
    #                phi.constrain(H, mesh.facesRight)
    #                phi.constrain(H, mesh.facesTop)
    #                phi.constrain(H, mesh.facesBottom)
    #            else:
    #                print 'Changing boundary conditions -> Neumann'
    #                phi.faceGrad.constrain([0], mesh.facesLeft)
    #                phi.faceGrad.constrain([0], mesh.facesRight)
    #                phi.faceGrad.constrain([0], mesh.facesTop)
    #                phi.faceGrad.constrain([0], mesh.facesBottom)
        
        hts[d,:,:]=phi.value.reshape(ny,nx)         
        #Video outputfigs
        date0=datetime.datetime(yr0,1,1)
        datenow=date0+datetime.timedelta(days=d)
        x=range(nx) 
        y=range(ny)
        X,Y =np.meshgrid(x,y)
    
        """
        plt.close('all')
        fig=plt.figure()
        Z= ele.reshape(ny,nx)-hts[d,:,:]               
        if d < 10: 
            prenum= '00'
        elif d<100:
            prenum='0'
        else:
            prenum=''
        nro='striphy '+ prenum + str(d)
        ax = fig.add_subplot(111);
        levels=[0.0, 0.2, 0.4, 1.0]
        plt.title(datenow.date())
        plt.imshow(ele.reshape(ny,nx)-hts[d,:,:], vmin = 0.0, vmax=1., cmap='Accent', interpolation=None, \
            extent=[0,nx*dx,0,ny*dy]); plt.colorbar()
        CS=plt.contour(X*dx,Y*dy,Z, levels, colors='g'); plt.clabel(CS, fontsize=10)
        sfolder='C:\Apps\WinPython-64bit-2.7.10.3\IPEWG\\hydro\\figs\\vid\\'
        plt.savefig(sfolder+nro+'.png')
        """            
        d +=1
        
    PrOpt=False
    if PrOpt==True:   
        x=range(nx) 
        y=range(ny)
        X,Y =np.meshgrid(x,y)
        fig=plt.figure()            
     
        for r in range(1,days):
            Z= ele.reshape(ny,nx)-hts[r-1,:,:]               
            nro='33'+str(r)
            ax = fig.add_subplot(int(nro));
            levels=[-0.2,0.0, 0.2, 0.4, 1.0]
            plt.imshow(ele.reshape(ny,nx)-hts[r-1,:,:], vmin = -0.5, vmax=1., cmap='Accent', interpolation=None, \
                extent=[0,nx*dx,0,ny*dy]); plt.colorbar()
            CS=plt.contour(X*dx,Y*dy,Z, levels, colors='g'); plt.clabel(CS, fontsize=10)
    #else:
    #    fig=plt.figure()            
    #    levels=[-0.2, 0.0, 0.2, 0.4, 1.0]
    #
    #    Z= ele.reshape(ny,nx)-hts[d-1,:,:]               
    #    plt.imshow(ele.reshape(ny,nx)-hts[d-1,:,:], vmin = -0.5, vmax=1., cmap='Accent', interpolation=None, \
    #        extent=[0,nx*dx,0,ny*dy]); plt.colorbar()
    #    CS=plt.contour(X*dx,Y*dy,Z, levels, colors='g'); plt.clabel(CS, fontsize=10)
    
    """ Volume of dry peat calc. Review!"""
    peat_vol_weights = utilities.PeatV_weight_calc(np.array(~dr*catchment_mask,dtype=int))
    dry_peat_volume = utilities.PeatVolume(peat_vol_weights, (ele-phi.value).reshape(ny,nx))
#    print "Dry peat volume = ", dry_peat_volume
    
    if plotOpt:
        plt.figure() 
        plt.title("elevation-phi in the end")
        plt.imshow((ele-phi.value).reshape(ny,nx), cmap='pink', interpolation='nearest'); plt.colorbar()
        #        CS=plt.contour(X*dx,Y*dy,Z, levels, colors='g'); plt.clabel(CS, fontsize=10)
        
        #plt.figure()
        #plt.title("Transmissivity at the end")
        #plt.imshow(((hToTra(ele-phi.value)-T_bottom)*cmask*drmask_not).value.reshape(ny,nx), cmap='plasma'); plt.colorbar()
            
                
        plt.figure()
        plt.title("C in the end")
        plt.imshow(C(ele-phi.value).reshape(ny,nx), cmap='pink', interpolation='nearest'); plt.colorbar()
        
        plt.show()
        
    change_in_canals = ((ele-phi.value)*cmask.value*drmask.value).reshape(ny,nx) - ((ele-H)*cmask.value* drmask.value).reshape(ny,nx)
    final_water_table = (phi.value).reshape(ny,nx)

        

    return dry_peat_volume, change_in_canals, final_water_table