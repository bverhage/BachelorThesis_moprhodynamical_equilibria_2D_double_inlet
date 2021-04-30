# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:59:38 2020

@author: billy
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:05:34 2020

@author: billy
"""

#4 
if True:
    print('----------------------------------------------- ')
    print('\n \t Welcome to the code of Assignment-6.py \n ')
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as la
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import time
    
    from tqdm import tqdm 
    
    #import scipy.fftpack
    ## just some code to mange the open plt windows
    plt.close("all")
    
#prelimanries 
if True:
    
    Ttide=44.9*10**3 #s
    
    T=10*Ttide
    Lx=59*10**3 #m
    Ly=1*10**3 #m
    H=10#m
    g=9.81 #m/s^2
    
    Nx=50+1#25#100
    Ny=int((Nx-1)*1/1)+1
    
    dx=Lx/Nx
    dy=Ly/Ny
    
    
    
  #  epsilon=1.7*10**-2
   # a=1#6.2*10**-2
    #mu=1#8.9*10**-5
    #lam=np.sqrt(0.078)
    r=-0.025
   # beta=0.08
    
    fhat=0#7.1*10**-1
    #NU=2.5*10**-2
    
    print('\n | N_x=N_y=%i \n ' %(Nx))
    
    def func0(x,y):
        return 0
    def func1(x,y):
        return 0.1
    
    def funcIC(x,y):
        return 0.05*np.sin(2*x+y)-0.05*np.sin(x+3*y)
    
    def funcIC2(x,y):
        if (x-0.5)**2<0.3**2 and (y-0.5)**2<0.3**2: return 0.01*np.sin(x)
        else: return 0
        
    #def funcIC3(x,y):
    #    if (x-Lx*0.5)**2+(y-Ly*0.5)**2<(Ly/10)**2: return 1
    #    else: return 0
    
    def funcICzeta(x,y):
        if (x-Lx*0.5)**2+(y-Ly*0.5)**2<0.2**2: return 0.02*np.exp(-((x-Lx*0.5)**2+(y-Ly*0.5)**2)/(2*0.001))
        else: return 0
        
    def funcICC(x,y):
        if (x-0.5)**2+(y-0.5)**2<0.2**2: return 1*np.exp(-((x-0.5)**2+(y-0.5)**2)/(2*0.01))
        else: return 0
        
    def funcIC3(x,y):
         #if (x-0.5)**2<0.2**2 and (y-0.5)**2<0.2**2 : return 0.7
         #else: return 0.3
         return 0.9*x/Lx+0.01*(y/Ly-1)*y/Ly*x/Lx #1*np.sin(2*np.pi*((x/Lx-0.5/Lx)**2+0*(y/Ly-0.5/Ly)**2)*x/Lx)
    
    
    def westboundary(x,y):
        if x==1*dx:#(y!=dy or y!=Ly-dy or y!=[2*dy,Ly-2*dy]): 
            return 1
        #if (x-0.5)**2+(y-0.5)**2<0.01**2: return 1
        else: return 0
        
    
    def northboundary(x,y):
        if y==Ly-dy: return 1
        #if (x-0.5)**2+(y-0.5)**2<0.01**2: return 1
        else: return 0
        
    
    def eastboundary(x,y):
        if x==Lx-dx: return 1
        #if (x-0.5)**2+(y-0.5)**2<0.01**2: return 1
        #else: return 0
        return 0
    
    def southboundary(x,y):
        if y==dy: return 1
        else: return 0
    
    Boolean_forcing=False
    forcigfrequency=(2*np.pi)/Ttide
    
    def f(tStart,i):
        return 1*np.sin(forcigfrequency*(tStart+dt*i))
            
    def fprime(tStart,i):
        return 1*forcigfrequency*np.cos(forcigfrequency*(tStart+dt*i))
            

    
    def create(Nx:int,Ny:int,func):
        Fvec=np.zeros((Nx-1)*(Ny-1))
        for j in range(Ny-1):
            for i in range(Nx-1):
                x=dx*(i+1)
                y=dy*(j+1)
                Fvec[i+j*(Nx-1)]=func(x,y)
        return Fvec
    
    
    BoundaryForcing = create(Nx,Ny,westboundary)
    NBoundary       = create(Nx,Ny,northboundary)
    EBoundary       = create(Nx,Ny,eastboundary)
    SBoundary       = create(Nx,Ny,southboundary)
    
    colsLX, rowsLX, valsLX = sp.find(BoundaryForcing>0)
    

    
    
    ICzeta0 = create(Nx,Ny,func0)
    ICu0    = create(Nx,Ny,func0)
    ICv0    = create(Nx,Ny,func0)
    ICC0    = create(Nx,Ny,func0)
    ICh0    = create(Nx,Ny,funcIC3)
        
    # if True:
    #     uendarr = np.reshape(ICu0,[Nx-1,Ny-1])
    #     vendarr = np.reshape(ICv0,[Nx-1,Ny-1])
        
    #     plt.ion()
    #     fig, (ax1, ax2) = plt.subplots(1, 2,sharex=True, sharey=True)
    #     imgu = ax1.imshow(uendarr,extent=[dx/2,L-dx/2,L-dy/2,dy/2],interpolation='none')
    #     ax1.title.set_text('activator')
        
        
    #     plt.colorbar(imgu,orientation='horizontal',ax=ax1)
     
    #     imgv=ax2.imshow(vendarr,extent=[dx/2,L-dx/2,L-dy/2,dy/2],interpolation='none')
        
    #     plt.gca().invert_yaxis()
    #     ax2.title.set_text('inhibitor')
    #     plt.colorbar(imgv,orientation='horizontal',ax=ax2)
        
    #     tlt = plt.suptitle('Initial condtion \n t = 0')
        
        
        

    
    
   
      
    
    # conrole panel Forward Euler Time stepping
    
    Boolean_Forward_Euler_calculations=False
    Boolean_FE_animation=False
    
    NtStab = 4*T*Lx*dx**-2
    
    Nt = int(1.1*np.ceil(NtStab))
    dt = T/Nt
    
    
    # controle panel BENR Time steppping
    #Boolean_Backward_Euler_calculations=True  
    #Boolean_BENR_animation=True
    
    Boolean_print=True
    epsilon=10**(-3)
    ICU=np.concatenate((ICzeta0,ICu0))
    
    I_U=sp.eye(len(ICU))
    
    NtNR=int(np.ceil(1.1*np.cbrt(NtStab))) #guess idk change later
    
    dtNR=T/NtNR
    print('Numerical Euler Forward integraion is theoretically stable for Nt >= %3.2f \n' %(NtStab))
    print('| Euler Forward intergration using Nt = %i \n  ' %(Nt))
    print('| Euler Backward intergration using Nt = %i \n \n'  %(NtNR))
    

    # initial condition
#4a
if True:
    def uvsolveFE(zetaStart,uStart,vStart,CStart,hStart,tStart,tEnd,Nt):
        
        ht=(tEnd-tStart)/Nt
        zetaEnd=zetaStart
        uEnd=uStart
        vEnd=vStart
        CEnd=CStart
        hEnd=hStart

        for i in range(Nt): 
        #for i in tqdm(range(Nt)):
            ''' computational variables that are used more than ones '''
            
            zetaStart=zetaEnd
            #*(ONES-BoundaryForcing)
            #zetaStart+=f(tStart,i)*BoundaryForcing
            uStart=uEnd
            #*(ONES-EBoundary-BoundaryForcing)
            #uStart+=BoundaryForcing*fprime(tStart,i)/(H-0+f(tStart,i))
            vStart=vEnd
            CStart=CEnd
            hStart=hEnd
            
            DEPHT = H-hStart+zetaStart 
            
            FRICITON = r/DEPHT
            
            if Boolean_forcing==True:
                '''ontinutity equaiton'''
                '''ontinutity equaiton'''
                
                f_zeta  = -LxD.dot(DEPHT*uStart)#-BoundaryForcing*fprime(tStart,i) # xdireciton
                f_zeta += -LyD.dot(DEPHT*vStart)                                  # y direction
                
                
                zetaEnd = zetaStart-ht*f_zeta
                
                '''navier stockets u (x direction)'''
                f_u  =  fhat*vStart                                                          # coriolis
                f_u += -g*LxN.dot(zetaStart)# + NU/(2*dx)*BoundaryForcing*f(tStart,i)   # surface 
                f_u += -uStart*FRICITON                                                     # bed friciton
                f_u+=-uStart*LxD.dot(uStart)+BoundaryForcing*fprime(tStart,i)**2/(1-0+f(tStart,i))**2
                f_u+=-vStart*LyD.dot(uStart)+BoundaryForcing*fprime(tStart,i)/(1-0+f(tStart,i))*vStart
                
                uEnd = uStart-ht*f_u
                
                '''navier stockets v (y direction)'''
                f_v=-fhat*uStart  
                f_v+=-g*LyN.dot(zetaStart)+ g*(NBoundary-SBoundary)*zetaStart
                f_v+=-vStart*FRICITON
                f_v+=-uStart*LxD.dot(vStart)-vStart*LyD.dot(vStart)
                
                vEnd=vStart-ht*f_v
                
                '''concentration equation'''
                CEnd=CStart#-ht*(-LxN.dot(CStart*uStart)-LyN.dot(CStart*vStart)+mu*A.dot(CStart)+(-uStart**2-vStart**2+CStart))
                '''depth equaiton '''
                hEnd=hStart#-ht*(+uStart**2+vStart**2-CStart)
            else:
                
                forcingzeta=f(tStart,i)
                forcingu=-fprime(tStart,i)/(H-hStart+forcingzeta)
                '''ontinutity equaiton'''
                
                f_zeta  = -LxND.dot(DEPHT*uStart)+BoundaryForcing*(H-hStart+forcingzeta)*forcingu # xdireciton
                f_zeta += -LxD.dot(DEPHT*vStart)               # y direction
                
                
                zetaEnd = zetaStart-ht*f_zeta
                
                '''navier stockets u (x direction)'''
                f_u  =  fhat*vStart                                                          # coriolis
                f_u += -g*LxDN.dot(zetaStart) + g/(2*dx)*BoundaryForcing*forcingzeta  # surface 
                f_u += -uStart*FRICITON                                                     # bed friciton
                #f_u+=-uStart*LxND.dot(uStart)+BoundaryForcing*uStart*forcingu*1/(2*dx)
                #f_u+=-vStart*LyD.dot(uStart)+BoundaryForcing*fprime(tStart,i)/(1-0+f(tStart,i))*vStart
                
                uEnd = uStart-ht*f_u
                
                '''navier stockets v (y direction)'''
                f_v=-fhat*uStart  
                f_v+=-g*LyN.dot(zetaStart)# + NU*(NBoundary+SBoundary)/(2*dy)*zetaStart
                f_v+=-vStart*FRICITON
                #f_v+=-uStart*LxD.dot(vStart)-vStart*LyD.dot(vStart)
                
                vEnd=vStart-ht*f_v
                
                '''concentration equation'''
                CEnd=CStart#-ht*(-LxN.dot(CStart*uStart)-LyN.dot(CStart*vStart)+mu*A.dot(CStart)+(-uStart**2-vStart**2+CStart))
                '''depth equaiton '''
                hEnd=hStart#-ht*(+uStart**2+vStart**2-CStart)

                
                # Differential_equaiton=zetaStart-ht*(-LxD.dot(DEPHT*uStart)-LyD.dot(DEPHT*vStart))
                
                # Differential_equaiton[rowsLX]=0

                # Forcing=BoundaryForcing*forcingwave#-ht*(-LxD.dot(( 1-hStart+forcingwave)*derivative_forcingwave/(1-hStart+forcingwave))-LyD.dot(( 1-hStart+forcingwave )*0))
                
                # zetaEnd=Differential_equaiton+Forcing
                
                # Differential_equaiton=uEnd-ht*(fhat*vStart  -NU*LxN.dot(zetaStart)-uStart*FRICITON)#-uStart*LxD.dot(uStart)-vStart*LyD.dot(uStart))
                # Differential_equaiton[rowsLX]=0
                # Forcing=BoundaryForcing*derivative_forcingwave/(1-hStart+forcingwave)
                
                # uEnd=Differential_equaiton+Forcing
                

                    
        return zetaEnd,uEnd,vEnd,CEnd,hEnd
    
    
if True:
   def FDLaplacian2D(Nx,Ny,dx,dy):
        
        diagonalsx = [-1*np.ones((1,Nx)),np.ones((1,Nx-1))]
        diagonalsy = [-1*np.ones((1,Ny)),np.ones((1,Ny-1))]
        diagonalsx[1][0][0]=0;
        diagonalsx[0][0][-2]=0;
        diagonalsy[1][0][0]=0;
        diagonalsy[0][0][-2]=0;
        Dx =1/(dx)*sp.diags(diagonalsx, [-1, 0], shape=(Nx,Nx-1))
        Dy =1/(dy)*sp.diags(diagonalsy, [-1, 0], shape=(Ny,Ny-1))
        
        LXX=Dx.transpose().dot(Dx)
        LYY=Dy.transpose().dot(Dy)

        Ans = sp.kron(LYY,sp.eye(Nx-1))+sp.kron(sp.eye(Ny-1),LXX)
        return Ans
    
   def LX(Nx,Ny,dx,dy,LeftNeuman=False,RightNeuman=False):
        
        diagonalsx = [-1*np.ones((1,Nx-1-1)),np.ones((1,Nx-1-1))]
        
        if LeftNeuman==True:
            diagonalsx[1][0][0]=0;
        if RightNeuman==True:
            diagonalsx[0][0][-1]=0;
    
        Dx =1/(2*dx)*sp.diags(diagonalsx, [-1, 1], shape=(Nx-1,Nx-1))
        Ans =sp.kron(sp.eye(Ny-1),Dx)#sp.kron(Dx,sp.eye(Ny-1))
        return Ans
   
   def LFDX(Nx,Ny,dx,dy,LeftNeuman=False,RightNeuman=False):  
       
        diagonalsx = [-1*np.ones((1,Nx-1-1)),np.ones((1,Nx-1))]
        
        if RightNeuman==True:
            diagonalsx[1][0][-1]=0
            diagonalsx[0][0][-1]=0;
    
        Dx =1/(2*dx)*sp.diags(diagonalsx, [-1, 0], shape=(Nx-1,Nx-1))
        Ans =sp.kron(sp.eye(Ny-1),Dx)#sp.kron(Dx,sp.eye(Ny-1))
        return Ans
       
    
   def LY(Nx,Ny,dx,dy,LeftNeuman=False,RightNeuman=False):
        
        diagonalsy = [-1*np.ones((1,Ny-1-1)),np.ones((1,Ny-1-1))]
        
        if LeftNeuman==True:
            diagonalsy[1][0][0]=0;
        if RightNeuman==True:
            diagonalsy[0][0][-1]=0;
    


        Dy =1/(2*dy)*sp.diags(diagonalsy, [-1, 1], shape=(Ny-1,Ny-1))
       
        
        

        Ans =sp.kron(Dy,sp.eye(Nx-1))#sp.kron(sp.eye(Nx-1),Dy)
        return Ans
    
    
        
        
FLD = LFDX(Nx,Ny,dx,dy,LeftNeuman=False,RightNeuman=False)
 
# 2D FD Laplacian and identity matrices
A = FDLaplacian2D(Nx,Ny,dx,dy)
LxD = LX(Nx,Ny,dx,dy)
LxND = LX(Nx,Ny,dx,dy,True)
LxDN = LX(Nx,Ny,dx,dy,False,True)
LxN = LX(Nx,Ny,dx,dy,True,True)

LyD = LY(Nx,Ny,dx,dy)
LyND = LY(Nx,Ny,dx,dy,True)
LyDN = LY(Nx,Ny,dx,dy,False,True)
LyN = LY(Nx,Ny,dx,dy,True,True)

I = sp.eye((Nx-1)*(Ny-1))

ONES= np.ones((Nx-1)*(Ny-1))

    


if Boolean_Forward_Euler_calculations:
    
    
    
    if Boolean_FE_animation:
    
        t = 0
    
        plt.ion()
        
        fig, ((ax1, ax2, ax3, ax4, ax5),(ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5)
        
        # Inital conditoin 
        zeta0=np.copy(ICzeta0)
        u0=np.copy(ICu0)
        v0=np.copy(ICv0)
        C0=np.copy(ICC0)
        h0=np.copy(ICh0)
        # initialization of the movie figure
        zeta0arr = np.reshape(zeta0,[Ny-1,Nx-1])
        u0arr = np.reshape(u0,[Ny-1,Nx-1])
        v0arr = np.reshape(v0,[Ny-1,Nx-1])
        C0arr = np.reshape(C0,[Ny-1,Nx-1])
        h0arr = np.reshape(h0,[Ny-1,Nx-1])
        
        
        FFT=np.fft.fftshift(np.fft.fft2(zeta0arr,(Ny-1,Nx-1)))
        
        imgzetacross = ax1.plot(zeta0arr.mean(0),'k.',markersize=1)
        ax1.title.set_text('zeta cross avg')
        ax1.set_ylim([-1.2,1.2])
        
        imgucross = ax2.plot(u0arr.mean(0),'k.',markersize=1)
        ax2.title.set_text('u cross avg')
        ax2.set_ylim([-0.2,0.2])
        
        imgvcross = ax3.plot(v0arr.mean(0),'k.',markersize=1)
        ax2.title.set_text('v cross avg')
        
        
      
        imgCcross = ax4.plot(C0arr.mean(0),'k.',markersize=1)
        ax4.title.set_text('C cross avg')
        
        
        
        imghcross = ax5.plot(h0arr.mean(0),'k.',markersize=5)
        ax5.title.set_text('h cross avg')
        
        imgzeta = ax6.imshow(zeta0arr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        ax6.title.set_text('zeta')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgzeta,orientation='horizontal',ax=ax6)
     
        imgu=ax7.imshow(u0arr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        plt.gca().invert_yaxis()
        ax7.title.set_text('u')
        plt.colorbar(imgu,orientation='horizontal',ax=ax7)
        
        imgv=ax8.imshow(v0arr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        plt.gca().invert_yaxis()
        ax8.title.set_text('v')
        plt.colorbar(imgu,orientation='horizontal',ax=ax8)
        
        
        imgC = ax9.imshow(abs(FFT),interpolation='none')
        
        ax9.title.set_text('abs(FFT)')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgC,orientation='horizontal',ax=ax9)
     
        imgh=ax10.imshow(np.angle(np.fft.fft2(zeta0arr)),interpolation='none')
        
        plt.gca().invert_yaxis()
        ax10.title.set_text('angle')
        plt.colorbar(imgh,orientation='horizontal',ax=ax10)
        plt.gca().invert_yaxis()
        
        
        tlt = plt.suptitle('t = %3.3f' %(t))
        
        
        
        def animate(frame):
            '''
            This function updates the solution array
            '''
            global t, dt, Nx, Ny, zeta0, zeta1, u0, u1, v0, v1, C0,C1, h0,h1,Data
            NF=1000
            t = NF*(frame+1)*dt
            CPUtime=time.time()
           
            
            #if frame == 0:
            #    #inital frame
            #    u1 = np.copy(u0)
            #else:
            
            
            zeta1,u1,v1,C1,h1 = np.copy(uvsolveFE(zeta0,u0,v0,C0,h0,t,t+dt*NF,NF))
            
            zeta0 = np.copy(zeta1)
            u0 = np.copy(u1)
            v0 = np.copy(v1)
            
            FFT=np.fft.fftshift(np.fft.fft2(np.reshape(zeta1,[Ny-1,Nx-1])))
            
            C0 = np.copy(C1)
            h0 = np.copy(h1)
            #if np.max([np.abs(u0-u1),np.abs(v0-v1)])>100:
            #     raise ValueError('Divergence \n BREAK \n ')
            #     pass
                
            
            imgzetacross[0].set_ydata(np.reshape(zeta1,[Ny-1,Nx-1]).mean(0))
            imgucross[0].set_ydata(np.reshape(u1,[Ny-1,Nx-1]).mean(0))
            imgvcross[0].set_ydata(np.reshape(v1,[Ny-1,Nx-1]).mean(0))
            imgCcross[0].set_ydata(np.reshape(C0,[Ny-1,Nx-1]).mean(0))
            

            
            imghcross[0].set_ydata(np.reshape(h0,[Ny-1,Nx-1]).mean(0))
            
            imgzeta.set_array(np.reshape(zeta1,[Ny-1,Nx-1]))
            imgu.set_array(np.reshape(u1,[Ny-1,Nx-1]))
            
            imgv.set_array(np.reshape(v1,[Ny-1,Nx-1]))
            imgC.set_array(abs(FFT))
            imgh.set_array(np.angle(FFT))

            
            
            tlt.set_text('t = %3.3f,  avg computation time = %3.3f ms' %(t/3600,1000*(time.time()-CPUtime)/NF))
            imgzeta.set_clim(zeta1.min(),zeta1.max())
            imgu.set_clim(u1.min(),u1.max())
            imgv.set_clim(v1.min(),v1.max())
            imgC.set_clim(np.abs(FFT).min(),np.abs(FFT).max())
            imgh.set_clim(np.angle(FFT).min(),np.angle(FFT).max())
            
            
            
            
            
            
            return imgzetacross,imgucross,imgvcross,imgCcross,imghcross,imgzeta,imgu,imgv,imgC,imgh
        
        # figure animation
        
        anim = animation.FuncAnimation(fig , animate , Nt , interval=50 , repeat=True)
        #plt.pause(50*1e-3*Nt+1)
    
        print('movie finished')
    
    # # # code to directly make the last image
    else:
        
        t = 0
        NtEnd=int((Nt))
    
       
        
        # Inital conditoin 
        zeta0=np.copy(ICzeta0)
        u0=np.copy(ICu0)
        v0=np.copy(ICv0)
        C0=np.copy(ICC0)
        h0=np.copy(ICh0)
        
    #     # The run
    #     print('\t --- Forward-Euler integration ---- \n')
        
    #     timeFE=time.time()
        zetaend,uend,vend,Cend,hend=uvsolveFE(zeta0,u0,v0,C0,h0,0,0+dt*NtEnd,NtEnd)
    #     CPUtimeFE = time.time()-timeFE
        zetaarr = np.reshape(zetaend,[Ny-1,Nx-1])
        uendarr = np.reshape(uend,[Ny-1,Nx-1])
        vendarr = np.reshape(vend,[Ny-1,Nx-1])
        Cendarr = np.reshape(Cend,[Ny-1,Nx-1])
        hendarr = np.reshape(hend,[Ny-1,Nx-1])
         
        plt.ion()
        
        fig, ((ax1, ax2, ax3, ax4,ax5),(ax6, ax7, ax8, ax9,ax10)) = plt.subplots(2, 5)
                
        imgzetacross = ax1.plot(zetaarr.mean(0),linewidth=0.2,color='k')
        ax1.title.set_text('zeta cross avg')
        
        imgucross = ax2.plot(uendarr.mean(0),linewidth=0.2,color='k')
        ax2.title.set_text('u cross avg')
        
        imgvcross = ax3.plot(vendarr.mean(0),linewidth=0.2,color='k')
        ax2.title.set_text('v cross avg')
        
        imgCcross = ax4.plot(Cendarr.mean(0),linewidth=0.2,color='k')
        ax4.title.set_text('C cross avg')
        
        imghcross = ax5.plot(hendarr.mean(0),linewidth=0.2,color='k')
        ax5.plot(1+zetaarr.mean(0),linewidth=0.2,color='b')
        ax5.title.set_text('h cross avg')
        
        imgzeta = ax6.imshow(zetaarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none')
        
        ax6.title.set_text('zeta')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgzeta,orientation='horizontal',ax=ax6)
     
        imgu=ax7.imshow(uendarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none')
        
        plt.gca().invert_yaxis()
        ax7.title.set_text('u')
        plt.colorbar(imgu,orientation='horizontal',ax=ax7)
        
        imgv=ax8.imshow(vendarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none')
        
        plt.gca().invert_yaxis()
        ax8.title.set_text('v')
        plt.colorbar(imgu,orientation='horizontal',ax=ax8)
        
        imgC = ax9.imshow(Cendarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none')
        
        ax9.title.set_text('Concentration')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgC,orientation='horizontal',ax=ax9)
     
        imgh=ax10.imshow(hendarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none')
        
        plt.gca().invert_yaxis()
        ax10.title.set_text('h')
        plt.colorbar(imgh,orientation='horizontal',ax=ax10)
        plt.gca().invert_yaxis()
        
    #     plt.ion()
    #     fig, (ax1, ax2) = plt.subplots(1, 2,sharex=True, sharey=True)
    #     imgu = ax1.imshow(uendarr,extent=[dx/2,L-dx/2,L-dy/2,dy/2],interpolation='none')
    #     ax1.title.set_text('activator')
        
        
    #     plt.colorbar(imgu,orientation='horizontal',ax=ax1)
     
    #     imgv=ax2.imshow(vendarr,extent=[dx/2,L-dx/2,L-dy/2,dy/2],interpolation='none')
        
    #     plt.gca().invert_yaxis()
    #     ax2.title.set_text('inhibitor')
    #     plt.colorbar(imgv,orientation='horizontal',ax=ax2)
        
    #     tlt = plt.suptitle('Forward Euler Nt = %i \n t = %3.3f' %(NtEnd,NtEnd*dt))
        
       
    #     print('Computation time for Forward Euler. \n | Nt = %i \t CPU time: %f s \n' %(NtEnd,CPUtimeFE))
        

''' Eigen function analysis'''
if False:
    Data=np.array([0,0,0,0,0,0,0,0,0])
    period=10
    i=0
    for i in tqdm(range(period*int(Ttide/dt),(period+1)*int(Ttide/dt))):
        t=Ttide+i*dt
        if i==int(Nt/2):
            zeta1,u1,v1,C1,h1=uvsolveFE(zeta0,u0,v0,C0,h0,0,0+dt*NtEnd,NtEnd)
        zeta1,u1,v1,C1,h1 = np.copy(uvsolveFE(zeta0,u0,v0,C0,h0,t,t+dt,1))
                
        zeta0 = np.copy(zeta1)
        u0 = np.copy(u1)
        v0 = np.copy(v1)
        C0 = np.copy(C1)
        h0 = np.copy(h1)
        
        phi1=np.sin(1*forcigfrequency*t)
        phi2=np.sin(2*forcigfrequency*t)
        phi3=np.sin(3*forcigfrequency*t)
        phi4=np.sin(4*forcigfrequency*t)
        phi5=np.sin(5*forcigfrequency*t)
        phi6=np.sin(6*forcigfrequency*t)
        phi7=np.sin(7*forcigfrequency*t)
        phi8=np.sin(8*forcigfrequency*t)
        nx=10
        ny=int(Ny/2)
        point=nx+ny*(Ny-1)
        Data=Data+np.array([0,
                            dt*phi1*zeta1[point],     #1
                            dt*phi2*zeta1[point],     #2
                            dt*phi3*zeta1[point],     #3
                            dt*phi4*zeta1[point],     #4
                            dt*phi5*zeta1[point],     #5
                            dt*phi6*zeta1[point],     #6
                            dt*phi7*zeta1[point],     #7
                            dt*phi8*zeta1[point]])
    
    fig, ((ax1, ax2, ax3, ax4,ax5),(ax6, ax7, ax8, ax9,ax10)) = plt.subplots(2, 5)
            
    imgzetacross = ax1.plot(np.reshape(zeta1,[Ny-1,Nx-1]).mean(0),linewidth=0.2,color='k')
    ax1.title.set_text('zeta cross avg')
    
    imgucross = ax2.plot(uendarr.mean(0),linewidth=0.2,color='k')
    ax2.title.set_text('u cross avg')
    
    imgvcross = ax3.plot(vendarr.mean(0),linewidth=0.2,color='k')
    ax2.title.set_text('v cross avg')
    
    imgCcross = ax4.plot(Cendarr.mean(0),linewidth=0.2,color='k')
    ax4.title.set_text('C cross avg')
    
    imghcross = ax5.plot(hendarr.mean(0),linewidth=0.2,color='k')
    ax5.plot(1+zetaarr.mean(0),linewidth=0.2,color='b')
    ax5.title.set_text('h cross avg')
    
    imgzeta = ax6.imshow(zetaarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none')
    
    ax6.title.set_text('zeta')
    plt.gca().invert_yaxis()
    
    plt.colorbar(imgzeta,orientation='horizontal',ax=ax6)
     
    imgu=ax7.imshow(uendarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none')
    
    plt.gca().invert_yaxis()
    ax7.title.set_text('u')
    plt.colorbar(imgu,orientation='horizontal',ax=ax7)
    
    imgv=ax8.imshow(vendarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none')
    
    plt.gca().invert_yaxis()
    ax8.title.set_text('v')
    plt.colorbar(imgu,orientation='horizontal',ax=ax8)
    
    imgC = ax9.imshow(Cendarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none')
    
    ax9.title.set_text('Concentration')
    plt.gca().invert_yaxis()
    
    plt.colorbar(imgC,orientation='horizontal',ax=ax9)
     
    imgh=ax10.imshow(hendarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none')
    
    plt.gca().invert_yaxis()
    ax10.title.set_text('h')
    plt.colorbar(imgh,orientation='horizontal',ax=ax10)
    plt.gca().invert_yaxis()
    print(Data)
#------------------------ Newton Rapson --------------------------
  
h=ICh0  

Uinnitalguess=np.concatenate((ICzeta0,ICzeta0,ICu0,ICu0,ICv0,ICv0))
    
# #part 2 the Newton rapson method

# # first we need to define some functions:
    
def MaxNormOfU(U):
     ''' The max 2 norm of U=(u,v)^T w'''
     zetas,zetac,us,uc,vs,vc=np.array_split(U,6)
     return np.max([np.linalg.norm(zetas),np.linalg.norm(zetac),np.linalg.norm(us),np.linalg.norm(uc),np.linalg.norm(vs),np.linalg.norm(vc)])


def F(U):
    zetas,zetac,us,uc,vs,vc=np.array_split(U,6)
    
    fzetas =-zetas-(1-h)*(LxDN*uc+LyN*vc)+uc*(LxDN*h)+vc*(LyN*h)
    fzetac =-zetac+(1-h)*(LxDN*us+LyN*vs)-us*(LxDN*h)-vs*(LyN*h)
    fus    =-us+fhat*vc+np.divide(r, 1-h)*uc-g*LxDN*zetas
    fuc    =-uc-fhat*vs+np.divide(r, 1-h)*us-g*(LxDN*zetac+BoundaryForcing)
    fvs    =-vs-fhat*uc+np.divide(r, 1-h)*vc-g*LyN*zetas
    fvc    =-vc+fhat*us+np.divide(r, 1-h)*vs-g*LyN*zetac
    
    return np.concatenate((fzetas,fzetac,fus,fuc,fvs,fvc))

def Jacobian(U):
     zetas,zetac,us,uc,vs,vc=np.array_split(U,6)
     zeros=sp.csr_matrix(LxD.shape)
     
     J11= -I                               # zetas,zetas
     J12= zeros                            # zetas,zetac
     J13= zeros                            # zetas, us
     J14= -LxDN.multiply(1-h)+sp.diags(LxDN*h)
     J15= zeros
     J16= -LyN.multiply(1-h)+sp.diags(LyN*h)
     
     J21= zeros
     J22= -I
     J23=  LxDN.multiply(1-h)-sp.diags(LxDN*h)
     J24= zeros
     J25=  LyN.multiply(1-h)-sp.diags(LyN*h)
     J26= zeros
     
     J31= -g*LxDN
     J32= zeros
     J33= -I
     J34= sp.diags(np.divide(r,1-h))
     J35= zeros
     J36= I*fhat
     
     J41= zeros
     J42= -g*LxDN
     J43= sp.diags(np.divide(r,1-h))
     J44= -I
     J45= -I*fhat
     J46= zeros
          
     J51= -g*LyN
     J52= zeros
     J53= zeros
     J54= I*fhat
     J55= -I
     J56= sp.diags(np.divide(r,1-h))
     
     J61= zeros
     J62= -g*LyN
     J63= -I*fhat
     J64= zeros
     J65= sp.diags(np.divide(r,1-h))
     J66= -I
     
     J=sp.bmat([
                [J11, J12, J13, J14, J15, J16],
                [J21, J22, J23, J24, J25, J26],
                [J31, J32, J33, J34, J35, J36],
                [J41, J42, J43, J44, J45, J46],
                [J51, J52, J53, J54, J55, J56],
                [J61, J62, J63, J64, J65, J66]
                ],format='csr')
     return J

# def residual(U,U1,ht):
#     return U-U1+ht*F(U1)


def NewtonRapsonInnerloop(Uinnitalguess:'np.ndarray'):
    Uiend=np.copy(Uinnitalguess)
    
    i=0
    
#     if Boolean_print: print('\t Newton Rapson Inner loop \n i=0 \t ||U-U_i+h_t*F(U_i)|| = %f < %f ' %(MaxNormOfU(residual(Uinnitalguess,Uiend,ht)),epsilon))
    
#     if MaxNormOfU(residual(Uinnitalguess,Uiend,ht))>epsilon:
#         Stopcondition=1
#     else:
#         Stopcondition=0
#         #Uiend=Uiend+la.spsolve(I_U-ht*Jacobian(Uiend),residual(Uinnitalguess,Uiend,ht))
    
#     while Stopcondition==1:
    for i in range(1,20):
        
         Uiend=Uiend-la.spsolve(Jacobian(Uiend),F(Uiend))
         #i+=1
        
#         if MaxNormOfU(residual(Uinnitalguess,Uiend,ht))>epsilon:
#             Stopcondition=1
#             if Boolean_print:print('\t Newton Rapson Inner loop \n i=%i \t ||U-U_i+h_t*F(U_i)|| = %f < %f' %(i,MaxNormOfU(residual(Uinnitalguess,Uiend,ht)),epsilon))
#         if MaxNormOfU(residual(Uinnitalguess,Uiend,ht))>1000000000*epsilon: # this is the fail save if the explodes.
#             if Boolean_print:print('\n \t -----Divergence----- ')
#             break 
#         if MaxNormOfU(residual(Uinnitalguess,Uiend,ht))<=epsilon:
#             Stopcondition=0
#             if Boolean_print:print('\t Newton Rapson Inner loop \n i=%i \t ||U-U_i+h_t*F(U_i)|| = %f < %f' %(i,MaxNormOfU(residual(Uinnitalguess,Uiend,ht)),epsilon))
    
    return Uiend
    
     
zetas,zetac,us,uc,vs,vc=np.array_split(NewtonRapsonInnerloop(Uinnitalguess),6)

if True:
        plt.ion()
        
        fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3)
        

        # initialization of the movie figure
        zetasarr = np.reshape(zetas,[Ny-1,Nx-1])
        zetacarr = np.reshape(zetac,[Ny-1,Nx-1])
        usarr = np.reshape(us,[Ny-1,Nx-1])
        ucarr = np.reshape(uc,[Ny-1,Nx-1])
        vsarr = np.reshape(vs,[Ny-1,Nx-1])
        vcarr = np.reshape(vc,[Ny-1,Nx-1])
        
        
    
        
        imgzetas = ax1.imshow(zetasarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        ax1.title.set_text('zeta-sin')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgzetas,orientation='horizontal',ax=ax1)
        
        imgzetac = ax2.imshow(zetacarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        ax2.title.set_text('zeta-cos')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgzetac,orientation='horizontal',ax=ax2)
     
        imgus = ax3.imshow(usarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        ax3.title.set_text('u-sin')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgus,orientation='horizontal',ax=ax3)
        
        imguc = ax4.imshow(ucarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        ax4.title.set_text('u-cos')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imguc,orientation='horizontal',ax=ax4)
        
        imgvs = ax5.imshow(vsarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        ax5.title.set_text('v-sin')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgvs,orientation='horizontal',ax=ax5)
        
        imgvc = ax6.imshow(vcarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        ax6.title.set_text('v-cos')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgvc,orientation='horizontal',ax=ax6)


if True:
        t = 0
    
        plt.ion()
        
        fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2,3)
        
        # Inital conditoin 
        zeta0=zetas*np.sin(t)+zetac*np.cos(t)
        u0=us*np.sin(t)+uc*np.cos(t)
        v0=vs*np.sin(t)+vc*np.cos(t)

        # initialization of the movie figure
        zeta0arr = np.reshape(zeta0,[Ny-1,Nx-1])
        u0arr = np.reshape(u0,[Ny-1,Nx-1])
        v0arr = np.reshape(v0,[Ny-1,Nx-1])

        
        
        imgzetacross = ax1.plot(zeta0arr.mean(0),'k.',markersize=1)
        ax1.title.set_text('zeta cross avg')
        ax1.set_ylim([-1.2,1.2])
        
        imgucross = ax2.plot(u0arr.mean(0),'k.',markersize=1)
        ax2.title.set_text('u cross avg')
        ax2.set_ylim([-0.2,0.2])
        
        imgvcross = ax3.plot(v0arr.mean(0),'k.',markersize=1)
        ax2.title.set_text('v cross avg')
        
        
      
        
        imgzeta = ax4.imshow(zeta0arr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        ax4.title.set_text('zeta')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgzeta,orientation='horizontal',ax=ax4)
     
        imgu=ax5.imshow(u0arr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        plt.gca().invert_yaxis()
        ax5.title.set_text('u')
        plt.colorbar(imgu,orientation='horizontal',ax=ax5)
        
        imgv=ax6.imshow(v0arr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        plt.gca().invert_yaxis()
        ax6.title.set_text('v')
        plt.colorbar(imgu,orientation='horizontal',ax=ax6)
        
        
        tlt = plt.suptitle('t = %3.3f' %(t))
        
        
        
        def animate(frame):
            '''
            This function updates the solution array
            '''
            global t, dt, Nx, Ny
            NF=1000
            t = NF*(frame+1)*dt
            
  
            
            
            zeta1=zetas*np.sin(t)+zetac*np.cos(t)
            u1=us*np.sin(t)+uc*np.cos(t)
            v1=vs*np.sin(t)+vc*np.cos(t)
            
                
            
            imgzetacross[0].set_ydata(np.reshape(zeta1,[Ny-1,Nx-1]).mean(0))
            imgucross[0].set_ydata(np.reshape(u1,[Ny-1,Nx-1]).mean(0))
            imgvcross[0].set_ydata(np.reshape(v1,[Ny-1,Nx-1]).mean(0))

            
            imgzeta.set_array(np.reshape(zeta1,[Ny-1,Nx-1]))
            imgu.set_array(np.reshape(u1,[Ny-1,Nx-1]))
            
            imgv.set_array(np.reshape(v1,[Ny-1,Nx-1]))


            
            
            tlt.set_text('t = %3.3f' %(t/3600))
            imgzeta.set_clim(zeta1.min(),zeta1.max())
            imgu.set_clim(u1.min(),u1.max())
            imgv.set_clim(v1.min(),v1.max())            
                                                
            return imgzetacross,imgucross,imgvcross,imgzeta,imgu,imgv
        
        # figure animation
        
        anim = animation.FuncAnimation(fig , animate , Nt , interval=50 , repeat=True)


# def uvsolveBENR(uStart,vStart,tStart,tEnd,Nt):
#     ht=(tEnd-tStart)/Nt
    
#     Utend=np.concatenate((uStart, vStart)) #time itteration variable
    
#     if Nt==1:
#         #Newton-Rapson Inner loop 

#         Utend=NewtonRapsonInnerloop(Utend,ht)
        
#     else:
#         for k in range(Nt): #tqdm(range(Nt)):
#             if Boolean_print:print('\n time itteration k= %i' %(k))
#             #Newton-Rapson Inner loop 
#             Uinnitalguess=np.copy(Utend)
#             Utend=NewtonRapsonInnerloop(Uinnitalguess,ht)
            
            
#     uEnd,vEnd=np.array_split(Utend,2)
#     return uEnd,vEnd
    

# if Boolean_Backward_Euler_calculations:
    
    
    
#     if Boolean_BENR_animation:
#         #animation
        
#         t = 0
#         plt.ion()
        
#         fig, (ax1, ax2) = plt.subplots(1, 2)
        
#         # Inital conditoin 
#         u0=np.copy(ICu0)
#         v0=np.copy(ICv0)
        
#         # initialization of the movie figure
#         u0arr = np.reshape(u0,[Nx-1,Ny-1])
#         v0arr = np.reshape(v0,[Nx-1,Ny-1])
        
        
#         imgu = ax1.imshow(u0arr,extent=[dx/2,L-dx/2,L-dy/2,dy/2],interpolation='none')
#         ax1.title.set_text('activator')
#         plt.gca().invert_yaxis()
        
#         plt.colorbar(imgu,orientation='horizontal',ax=ax1)
     
#         imgv=ax2.imshow(v0arr,extent=[dx/2,L-dx/2,L-dy/2,dy/2],interpolation='none')
        
#         plt.gca().invert_yaxis()
#         ax1.title.set_text('inhibitor')
#         plt.colorbar(imgv,orientation='horizontal',ax=ax2)
        
#         tlt = plt.suptitle('t = %3.3f' %(t))
        
#         def animate(frame):
#             '''
#             This function updates the solution array
#             '''
#             global t, dtNR, Nx, Ny, u0, u1, v0, v1
#             t = (frame+1)*dtNR
           
            
#             #if frame == 0:
#             #    #inital frame
#             #    u1 = np.copy(u0)
#             #else:
            
            
#             u1,v1 = np.copy(uvsolveBENR(u0,v0,t,t+dtNR,1))
            
#             u0 = np.copy(u1)
#             v0 = np.copy(v1)
            
#             #if np.max([np.abs(u0-u1),np.abs(v0-v1)])>100:
#             #     raise ValueError('Divergence \n BREAK \n ')
#             #     pass
                
            
            
#             imgu.set_array(np.reshape(u1,[Nx-1,Ny-1]))
#             imgv.set_array(np.reshape(v1,[Nx-1,Ny-1]))
            
#             tlt.set_text('t = %3.3f' %(t))
#             imgu.set_clim(u1.min(),u1.max())
#             imgv.set_clim(v1.min(),v1.max())
            
#             return imgu,imgv
        
#         # figure animation
        
#         anim = animation.FuncAnimation(fig , animate , NtNR , interval=50 , repeat=False)
#         #plt.pause(50*1e-3*Nt+1)
    
#         print('movie finished')
    
#     # # code to directly make the last image
#     else:
        
#         t = 0
#         NtNREnd=int((NtNR))
    
        
#         # Inital conditoin
#         u0=np.copy(ICu0)
#         v0=np.copy(ICv0)
        
#         # The run
#         print('\n \t --- Backward Euler Newton Rapson integration ---- \n')
#         timeFE=time.time()
#         uend,vend=uvsolveBENR(u0,v0,0,0+dtNR*NtNREnd,NtNREnd)
#         CPUtimeFE = time.time()-timeFE
#         uendarr = np.reshape(uend,[Nx-1,Ny-1])
#         vendarr = np.reshape(vend,[Nx-1,Ny-1])
        
#         plt.ion()
#         fig, (ax1, ax2) = plt.subplots(1, 2,sharex=True, sharey=True)
#         imgu = ax1.imshow(uendarr,extent=[dx/2,L-dx/2,L-dy/2,dy/2],interpolation='none')
#         ax1.title.set_text('activator')
        
        
#         plt.colorbar(imgu,orientation='horizontal',ax=ax1)
     
#         imgv=ax2.imshow(vendarr,extent=[dx/2,L-dx/2,L-dy/2,dy/2],interpolation='none')
        
#         plt.gca().invert_yaxis()
#         ax2.title.set_text('inhibitor')
#         plt.colorbar(imgv,orientation='horizontal',ax=ax2)
        
#         tlt = plt.suptitle('Backwards Euler Newton-Rapson Nt = %i \n t = %3.3f' %(NtNREnd,NtNREnd*dtNR))
        
        
#         print('Computation time for Backward Euler Newton Rapson. \n | Nt = %i \t CPU time: %f s \n' %(NtNR,CPUtimeFE))
        
    

    
        