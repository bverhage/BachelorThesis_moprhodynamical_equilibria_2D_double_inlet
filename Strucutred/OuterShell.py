# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:16:14 2021

@author: billy Verhage 
"""

if True:
    print('\t ---------------------------------------------------- ')
    print('\n \t Welcome to the code of BEP 2D FD Morphosdynamic al model \n ')
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as la
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
 
    import Model_parameters as P

    plt.close("all")
    
P.printparamters()  

BOOL_Total_Model, BOOL_Only_Water_model, BOOL_Only_Concentration_model = True,False,False



if BOOL_Total_Model:
    import Model_Numerical_Jacobian_total_model as TM
    import Model_Numerical_Jacobian_concentration_model as TM_con
    Uinnitalguess_con = np.concatenate((P.ICzeta0,P.ICzeta0,P.ICu0,P.ICu0,P.ICv0,P.ICv0,P.ICC0))
    
    print('\n \t ====Morphodynamical model====\n \t creating initial condition \n')
    
    #DeltaU=la.spsolve(TM_con.NumericalJacobian(Uinnitalguess_con),TM_con.F(Uinnitalguess_con))
    #Uinnitalguess_con=Uinnitalguess_con-DeltaU
    
    
    Uinnitalguess = np.concatenate((P.ICzeta0,P.ICzeta0,P.ICu0,P.ICu0,P.ICv0,P.ICv0,P.ICC0,P.ICh0))#TM_con.split_animation(Uinnitalguess_con)))
    
    
elif BOOL_Only_Water_model:
    import Model_Numerical_Jacobian_water_model as TM
    Uinnitalguess = np.concatenate((P.ICzeta0,P.ICzeta0,P.ICu0,P.ICu0,P.ICv0,P.ICv0))
    print('\t ====Water model====\n')

elif BOOL_Only_Concentration_model:
    import Model_Numerical_Jacobian_concentration_model as TM
    Uinnitalguess = np.concatenate((P.ICzeta0,P.ICzeta0,P.ICu0,P.ICu0,P.ICv0,P.ICv0,P.ICC0))
    print('\t ====Water+Concentration model====\n')
    





''' Initial condition '''


NJ=TM.NumericalJacobian(Uinnitalguess)
   





def NewtonRapsonInnerloop(Uinnitalguess:'np.ndarray'):
    print('\n \t  Starting Newton Rapson Method \t \n')
    epsilon=10**(-10)
    
    Uiend=np.copy(Uinnitalguess)
    
    i=0
    
    DeltaU=la.spsolve(NJ,TM.F(Uiend))
    Uiend=Uiend-DeltaU

    print('\t Newton Rapson loop \n i=0 \t ||F(U)||_max = %.2e < %.2e \n ' %(TM.MaxNormOfU(TM.F(Uiend)),epsilon))
    F_zetas,F_zetac,F_us,F_uc,F_vs,F_vc,F_C,F_h=TM.split_animation(TM.F(Uiend))
    print('\t | F(zetas) = %.2e \t | F(zetac) = %.2e \n \t | F(us) = %.2e \t | F(uc) = %.2e \n \t | F(vs) = %.2e \t | F(vc) = %.2e \n \t | F(C) = %.2e \t | F(h) = %.2e \n ' %(np.linalg.norm(F_zetas),np.linalg.norm(F_zetac),np.linalg.norm(F_us),np.linalg.norm(F_uc),np.linalg.norm(F_vs),np.linalg.norm(F_vc),np.linalg.norm(F_C),np.linalg.norm(F_h)))
    
    if TM.MaxNormOfU(TM.F(Uiend))>epsilon:
        Stopcondition=1
        
    else:
        Stopcondition=0
        
    
    while Stopcondition==1:
        
         
         i+=1
         Check=TM.MaxNormOfU(TM.F(Uiend))
         if Check>10**(20)*epsilon : # this is the fail save if the explodes.
             print('\n \t -----Divergence----- \n')
             break 
         
         if Check<=epsilon:
             print('\t Newton Rapson loop stoped at \n i=%i \t ||F(U)||_max = %.2e < %.2e\n' %(i,TM.MaxNormOfU(TM.F(Uiend)),epsilon))
             print ('\n')
             Stopcondition=0
             break
         
         elif Check>epsilon and Check<10**(20)*epsilon:

             Stopcondition=1
             DeltaU=la.spsolve(TM.NumericalJacobian(Uiend),TM.F(Uiend))
             Uiend=Uiend-DeltaU
             print('\t Newton Rapson loop \n i=%i \t ||F(U)||_max = %.2e < %.2e \n' %(i,TM.MaxNormOfU(TM.F(Uiend)),epsilon))
         zetas,zetac,us,uc,vs,vc,C,h=TM.split_animation(Uiend)
         
         F_zetas,F_zetac,F_us,F_uc,F_vs,F_vc,F_C,F_h=TM.split_animation(TM.F(Uiend))
         print('\t | F(zetas) = %.2e \t | F(zetac) = %.2e \n \t | F(us) = %.2e \t | F(uc) = %.2e \n \t | F(vs) = %.2e \t | F(vc) = %.2e \n \t | F(C) = %.2e \t | F(h) = %.2e \n ' %(np.linalg.norm(F_zetas),np.linalg.norm(F_zetac),np.linalg.norm(F_us),np.linalg.norm(F_uc),np.linalg.norm(F_vs),np.linalg.norm(F_vc),np.linalg.norm(F_C),np.linalg.norm(F_h)))

             
         if np.min(C)<0 :
            print('\n ------- \t  Negative Concentration! \t ---------\n')
            print('min(C) = %.2e\n ' %(np.min(C)))
        
         if np.min(1-h+P.epsilon*zetac)<0 or np.min(1-h+P.epsilon*zetas)<0 :
            print('\n ------- \t  NON physical water model! \t ---------\n')
            if np.min(1-h+P.epsilon*zetac)<np.min(1-h+P.epsilon*zetas):
                print('min(C) = %.2e\n ' %(np.min(1-h+P.epsilon*zetac)))
            else:
                print('min(C) = %.2e\n ' %(np.min(1-h+P.epsilon*zetas)))
                                           
            break    

         elif Check<=epsilon:
             print('\t Newton Rapson loop stoped at \n i=%i \t ||F(U)||_max = %.2e < %.2e\n' %(i,TM.MaxNormOfU(TM.F(Uiend)),epsilon))
             print ('\n')
             Stopcondition=0
             break
             
             
         if i>10:
            break
        

            

            
    return Uiend    

# the run
    
Ufinal=NewtonRapsonInnerloop(Uinnitalguess)
zetas,zetac,us,uc,vs,vc,C,h=TM.split_animation(Ufinal)


Check=np.where(TM.F(Ufinal)!=0)
if Check[0].size>0 :
    print('\t ------- F=!0 ------- \t \n')
    F_zetas,F_zetac,F_us,F_uc,F_vs,F_vc,F_C,F_h=TM.split_animation(TM.F(Ufinal))
    
    fig, ((ax1,ax3,ax5,ax7),(ax2,ax4,ax6,ax8)) = plt.subplots(2,4)
    plt.suptitle('F(U)') 
    plt.gca().invert_yaxis()
    imgzetas = ax1.imshow(P.reshape(F_zetas))
    ax1.title.set_text('$F(\zeta^{s})$')
    plt.colorbar(imgzetas,orientation='horizontal',ax=ax1)
    
    imgzetac = ax2.imshow(P.reshape(F_zetac))
    plt.gca().invert_yaxis()
    ax2.title.set_text('$F(\zeta^{c})$')
    plt.colorbar(imgzetac,orientation='horizontal',ax=ax2)
    
    imgus = ax3.imshow(P.reshape(F_us))
    plt.gca().invert_yaxis()
    ax3.title.set_text('$F(u^{s})$')
    plt.colorbar(imgus,orientation='horizontal',ax=ax3)
    
    imguc = ax4.imshow(P.reshape(F_uc))
    plt.gca().invert_yaxis()
    ax4.title.set_text('$F(u^{c})$')
    plt.colorbar(imguc,orientation='horizontal',ax=ax4)
    
    imgvs=ax5.imshow(P.reshape(F_vs))
    plt.gca().invert_yaxis()
    ax5.title.set_text('$F(vs)$')
    plt.colorbar(imgvs,orientation='horizontal',ax=ax5)
    
    imgvc=ax6.imshow(P.reshape(F_vc))
    plt.gca().invert_yaxis()
    ax6.title.set_text('$F(vc)$')
    plt.colorbar(imgvc,orientation='horizontal',ax=ax6)
    
    imgC=ax7.imshow(P.reshape(F_C))
    plt.gca().invert_yaxis()
    ax7.title.set_text('$F(C)$')
    plt.colorbar(imgC,orientation='horizontal',ax=ax7)
    
    imgh=ax8.imshow(P.reshape(F_h))
    plt.gca().invert_yaxis()
    ax8.title.set_text('$F(h)$')
    plt.colorbar(imgh,orientation='horizontal',ax=ax8)
    


''' the plots '''
def staticplots():
        plt.ion()
        
        fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3)
        

        # initialization of the movie figure
        zetasarr = P.reshape(zetas)
        zetacarr = P.reshape(zetac)
        usarr = P.reshape(us)
        ucarr = P.reshape(uc)
        vsarr = P.reshape(vs)
        vcarr = P.reshape(vc)
        
        
    
        
        imgzetas = ax1.imshow(zetasarr,extent=[P.dx/2,1-P.dx/2,1-P.dy/2,P.dy/2],interpolation='none',aspect='auto')
        
        ax1.title.set_text('zeta-sin')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgzetas,orientation='horizontal',ax=ax1)
        
        imgzetac = ax4.imshow(zetacarr,extent=[P.dx/2,1-P.dx/2,1-P.dy/2,P.dy/2],interpolation='none',aspect='auto')
        
        ax4.title.set_text('zeta-cos')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgzetac,orientation='horizontal',ax=ax4)
     
        imgus = ax2.imshow(usarr,extent=[P.dx/2,1-P.dx/2,1-P.dy/2,P.dy/2],interpolation='none',aspect='auto')
        
        ax2.title.set_text('u-sin')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgus,orientation='horizontal',ax=ax2)
        
        imguc = ax5.imshow(ucarr,extent=[P.dx/2,1-P.dx/2,1-P.dy/2,P.dy/2],interpolation='none',aspect='auto')
        
        ax5.title.set_text('u-cos')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imguc,orientation='horizontal',ax=ax5)
        
        imgvs = ax3.imshow(vsarr,extent=[P.dx/2,1-P.dx/2,1-P.dy/2,P.dy/2],interpolation='none',aspect='auto')
        
        ax3.title.set_text('v-sin')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgvs,orientation='horizontal',ax=ax3)
        
        imgvc = ax6.imshow(vcarr,extent=[P.dx/2,1-P.dx/2,1-P.dy/2,P.dy/2],interpolation='none',aspect='auto')
        
        ax6.title.set_text('v-cos')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgvc,orientation='horizontal',ax=ax6)
     
def Candhplots():
        plt.ion()
        plt.suptitle('Steady State bed, tidal average Concentration')
                     
        fig, ((ax1, ax2),(ax3, ax4),(ax5, ax6)) = plt.subplots(3, 2)
        
        # initialization of the movie figure
        Carr = P.reshape(C)
        harr = P.reshape(h)
        

        
        imgC = ax1.imshow(Carr,extent=[0,1,1,0],interpolation='none',aspect='auto')
        
        ax1.title.set_text('C')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgC,orientation='horizontal',ax=ax1)
        
        ax3.plot(Carr.mean(0))
        ax5.plot(Carr.mean(1))
        
        imgh = ax2.imshow(harr,extent=[0,1,1,0],interpolation='none',aspect='auto')
        
        ax2.title.set_text('h')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgh,orientation='horizontal',ax=ax2)
        
        ax4.plot(harr.mean(0))
        ax6.plot(harr.mean(1))


def Animation1():
        t = 0
    
        plt.ion()
        
        fig, ((ax1, ax2)) = plt.subplots(2)
        
        # Inital conditoin 
        zeta0=zetas*np.sin(t)+zetac*np.cos(t)
        u0=uc
        

        # initialization of the movie figure
        zeta0arr = zeta0
        u0arr = u0
        

        
        
        imgzetacross = ax1.plot(zeta0arr,'k.',markersize=1)
        ax1.title.set_text('zeta ')
        ax1.set_ylim([-P.Atilde,P.Atilde])
        
        imgucross = ax2.plot(u0arr,'k.',markersize=1)
        ax2.title.set_text('u')
        #ax2.set_ylim([-0.2,0.2])
        


        
        
        tlt = plt.suptitle('t = %3.3f' %(t))
        
        Tend=1
        NSteps=24*10
        anim_dt=Tend/NSteps
        
        def animate(frame):
            '''
            This function updates the solution array
            '''
            global t, Nx, Ny
            t = (frame+1)*anim_dt
                        
            zeta1=zetas*np.sin(2*np.pi*t)+zetac*np.cos(2*np.pi*t)
            u1=us*np.sin(2*np.pi*t)+uc*np.cos(2*np.pi*t)
            
                
            
            imgzetacross[0].set_ydata(zeta1)
            imgucross[0].set_ydata(u1)



            
            
            tlt.set_text('t = %3.3f' %(t))
        
                                                
            return imgzetacross,imgucross
        
        # figure animation
        
        return animation.FuncAnimation(fig , animate  , interval=50 , repeat=False)
  
    
def Animation2():
    t = 0
    
    plt.ion()
        
    fig, ((ax1,ax12, ax2)) = plt.subplots(3)
    # Inital conditoin 
    zeta0=zetas*np.sin(t)+zetac*np.cos(t)
    u0=uc
    v0=vc

    # initialization of the movie figure
    zeta0arr = P.reshape(zeta0)
    u0arr    = P.reshape(u0)
    v0arr    = P.reshape(v0)
        
    X = np.linspace(0, 1, P.Nx+1)
    Y = np.linspace(0, 1, P.Ny+1)
    U, V =v0arr, u0arr
    
    imgzeta = ax1.imshow(zeta0arr,extent=[P.dx/2,1-P.dx/2,1-P.dy/2,P.dy/2],interpolation='none',aspect='auto')
        
    ax1.title.set_text('zeta')
    plt.gca().invert_yaxis()
    
    plt.colorbar(imgzeta,orientation='horizontal',ax=ax1)
    

    imgzetacross = ax12.plot(zeta0arr.mean(0),linewidth=1,color='k')
    ax12.set_ylim([-(P.Atilde+1),(P.Atilde+1)])  
    
    ax2.quiver(X, Y, U, V, units='width')
    #ax2.quiverkey(q, 0.1, 0.1, 0.01, r'$2 \frac{m}{s}$', labelpos='E',
    #               coordinates='figure')
    
    tlt = plt.suptitle('t = %3.3f' %(t))
        
    Tend=1
    NSteps=24*10
    anim_dt=Tend/NSteps
        
    def animate(frame):
        '''
        This function updates the solution array
        '''
        t = (frame+1)*anim_dt
                        
        zeta1=zetas*np.sin(2*np.pi*t)+zetac*np.cos(2*np.pi*t)
        u1=us*np.sin(2*np.pi*t)+uc*np.cos(2*np.pi*t)
        v1=vs*np.sin(2*np.pi*t)+vc*np.cos(2*np.pi*t)
            
        imgzetacross[0].set_ydata(P.reshape(zeta1).mean(0))       
        imgzeta.set_array(P.reshape(zeta1))
        imgzeta.set_clim(zeta1.min(),zeta1.max())
        
        ax2.clear()
        ax2.quiver(X, Y, P.reshape(u1), P.reshape(v1), P.reshape(np.sqrt(u1**2+v1**2)),pivot='mid',units='dots',headwidth=0.1,linewidth=0.1,headlength=0.1)
        #ax2.quiverkey(q, 0.1, 0.1, 0.01)
            
        tlt.set_text('t = %3.3f' %(t))
      
                                                
        return imgzeta,imgzetacross,ax2
        
        # figure animation
        
    return animation.FuncAnimation(fig , animate  , interval=50 , repeat=True)

staticplots()

Animation2()

Candhplots()

    
print('\t ------------------------------')