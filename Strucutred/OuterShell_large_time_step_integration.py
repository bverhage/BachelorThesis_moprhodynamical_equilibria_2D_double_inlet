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
    import Model_functions as F
    from scipy import optimize
    
    plt.close("all")
    
P.printparamters()  

BOOL_Total_Model, BOOL_Only_Water_model, BOOL_Only_Concentration_model = False,False,True



if BOOL_Total_Model:
    import Model_Numerical_Jacobian_total_model as TM
    #import Model_Numerical_Jacobian_concentration_model as TM_con
    #Uinnitalguess_con = np.concatenate((P.ICzeta0,P.ICzeta0,P.ICu0,P.ICu0,P.ICv0,P.ICv0,P.ICC0))
    
    print('\n \t ====Morphodynamical model====\n')
    
    #DeltaU=la.spsolve(TM_con.NumericalJacobian(Uinnitalguess_con),TM_con.F(Uinnitalguess_con))
    #Uinnitalguess_con=Uinnitalguess_con-DeltaU
    
    
    Uinnitalguess = np.concatenate((P.ICzeta0,P.ICzeta0,P.ICu0,P.ICu0,P.ICv0,P.ICv0,P.ICC0,P.ICh0))#TM_con.split_animation(Uinnitalguess_con)))
    
    
elif BOOL_Only_Water_model:
    import Model_Numerical_Jacobian_water_model as TM
    Uinnitalguess = np.concatenate((P.ICzeta0,P.ICzeta0,P.ICu0,P.ICu0,P.ICv0,P.ICv0,P.ICC0,P.ICh0))
    print('\t ====Water model====\n')

elif BOOL_Only_Concentration_model:
    import Model_Numerical_Jacobian_concentration_model as TM
    Uinnitalguess = np.concatenate((P.ICzeta0,P.ICzeta0,P.ICu0,P.ICu0,P.ICv0,P.ICv0,P.ICC0,P.ICh0))
    print('\t ====Water+Concentration model====\n')
    





''' Initial condition '''


  

# the run
   



def NewtonRapsonInnerloop(Uinnitalguess:'np.ndarray'):
    print('\n \t  Starting Newton Rapson Method \t \n')
    epsilon=10**(-8)
    
    Uiend=np.copy(Uinnitalguess)
    
    i=0
    
    DeltaU=la.spsolve(TM.NumericalJacobian(Uiend),TM.F(Uiend))
    DeltaU=np.concatenate((DeltaU,P.ICC0))
            
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
             DeltaU=np.concatenate((DeltaU,P.ICC0))
            
             Uiend=Uiend-DeltaU
             print('\t Newton Rapson loop \n i=%i \t ||F(U)||_max = %.2e < %.2e \n' %(i,TM.MaxNormOfU(TM.F(Uiend)),epsilon))
         zetas,zetac,us,uc,vs,vc,C,h=TM.split_animation(Uiend)
         
         F_zetas=F.Fzetas(zetas,zetac,us,uc,vs,vc,C,h)
         F_zetac=F.Fzetac(zetas,zetac,us,uc,vs,vc,C,h)
         F_us=F.Fus(zetas,zetac,us,uc,vs,vc,C,h)
         F_uc=F.Fuc(zetas,zetac,us,uc,vs,vc,C,h)
         F_vs=F.Fvs(zetas,zetac,us,uc,vs,vc,C,h)
         F_vc=F.Fvc(zetas,zetac,us,uc,vs,vc,C,h)
         F_C=F.FC(zetas,zetac,us,uc,vs,vc,C,h)
         F_h=F.Fh(zetas,zetac,us,uc,vs,vc,C,h)
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
             
             
         if i>20:
            break
        

            

            
    return Uiend    

    
# Ufinal=NewtonRapsonInnerloop(Uinnitalguess)
# zetas,zetac,us,uc,vs,vc,C,h=TM.split_animation(Ufinal)

def time_integration(Uinnitalguess):
    
    tStart=0
    
    tEnd=1000000
    
    Nt=2000
    
    ht=(tEnd-tStart)/Nt
    U=Uinnitalguess
    
   
    
    for i in range(Nt):
            
        U=NewtonRapsonInnerloop(U)
        zetas,zetac,us,uc,vs,vc,C,h=TM.split(U)

        h=h+ht*P.Interior*F.Fh(zetas,zetac,us,uc,vs,vc,C,h)
        U = np.concatenate((zetas,zetac,us,uc,vs,vc,C,h))
        #uEnd=(I-ht*A).__pow__(Nt) * uStart
        
            
        if np.min(C)<0 :
            print('\n ------- \t  Negative Concentration! \t ---------\n')
            print('min(C) = %.2e\n ' %(np.min(C)))
                
            break
        if np.min(1-h+P.epsilon*zetac)<0 or np.min(1-h+P.epsilon*zetas)<0 :
            print('\n ------- \t  NON physical water model! \t ---------\n')
            
            break
            
    return U
    
    # def solveBE(uStart,tStart,tEnd,Nt):
    #     ht=(tEnd-tStart)/Nt
    #     uEnd=uStart
    #     for i in range(Nt):
    #         uEnd=la.spsolve((I+ht*A),uEnd)
    #     return uEnd
    
    # def solveTR(uStart,tStart,tEnd,Nt):
    #     ht=(tEnd-tStart)/Nt
    #     uEnd=uStart
    #     for i in range(Nt):
    #         uEnd=la.spsolve((I+0.5*ht*A),(I-0.5*ht*A).dot(uEnd))
    #     return uEnd

Ufinal=time_integration(Uinnitalguess)
zetas,zetac,us,uc,vs,vc,C,h=TM.split(Ufinal)  




F_zetas=F.Fzetas(zetas,zetac,us,uc,vs,vc,C,h)
F_zetac=F.Fzetac(zetas,zetac,us,uc,vs,vc,C,h)
F_us=F.Fus(zetas,zetac,us,uc,vs,vc,C,h)
F_uc=F.Fuc(zetas,zetac,us,uc,vs,vc,C,h)
F_vs=F.Fvs(zetas,zetac,us,uc,vs,vc,C,h)
F_vc=F.Fvc(zetas,zetac,us,uc,vs,vc,C,h)
F_C=F.FC(zetas,zetac,us,uc,vs,vc,C,h)
F_h=F.Fh(zetas,zetac,us,uc,vs,vc,C,h)
print('\t | F(zetas) = %.2e \t | F(zetac) = %.2e \n \t | F(us) = %.2e \t | F(uc) = %.2e \n \t | F(vs) = %.2e \t | F(vc) = %.2e \n \t | F(C) = %.2e \t | F(h) = %.2e \n ' %(np.linalg.norm(F_zetas),np.linalg.norm(F_zetac),np.linalg.norm(F_us),np.linalg.norm(F_uc),np.linalg.norm(F_vs),np.linalg.norm(F_vc),np.linalg.norm(F_C),np.linalg.norm(F_h)))

             
if np.min(C)<0 :
    print('\n ------- \t  Negative Concentration! \t ---------\n')
    print('min(C) = %.2e\n ' %(np.min(C)))
        
if np.min(1-h+P.epsilon*zetac)<0 or np.min(1-h+P.epsilon*zetas)<0 :
    print('\n ------- \t  NON physical water model! \t ---------\n')

                                           
Check=np.where(TM.F(Ufinal)!=0)
if Check[0].size>0 :
    print('\t ------- F=!0 ------- \t \n')
    
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
# def  generictotalplot(Ufinal):
#     F_zetas,F_zetac,F_us,F_uc,F_vs,F_vc,F_C,F_h=TM.split_animation(TM.F(Ufinal))
#     fig, ((ax1,ax3,ax5,ax7),(ax2,ax4,ax6,ax8)) = plt.subplots(2,4)
#     plt.suptitle('total plot') 
#     plt.gca().invert_yaxis()
#     imgzetas = ax1.imshow(P.reshape(F_zetas))
#     ax1.title.set_text('$\zeta^{s}$')
#     plt.colorbar(imgzetas,orientation='horizontal',ax=ax1)
    
#     imgzetac = ax2.imshow(P.reshape(F_zetac))
#     plt.gca().invert_yaxis()
#     ax2.title.set_text('$\zeta^{c}$')
#     plt.colorbar(imgzetac,orientation='horizontal',ax=ax2)
    
#     imgus = ax3.imshow(P.reshape(F_us))
#     plt.gca().invert_yaxis()
#     ax3.title.set_text('$u^{s}$')
#     plt.colorbar(imgus,orientation='horizontal',ax=ax3)
    
#     imguc = ax4.imshow(P.reshape(F_uc))
#     plt.gca().invert_yaxis()
#     ax4.title.set_text('$u^{c}$')
#     plt.colorbar(imguc,orientation='horizontal',ax=ax4)
    
#     imgvs=ax5.imshow(P.reshape(F_vs))
#     plt.gca().invert_yaxis()
#     ax5.title.set_text('$vs$')
#     plt.colorbar(imgvs,orientation='horizontal',ax=ax5)
    
#     imgvc=ax6.imshow(P.reshape(F_vc))
#     plt.gca().invert_yaxis()
#     ax6.title.set_text('$vc$')
#     plt.colorbar(imgvc,orientation='horizontal',ax=ax6)
    
#     imgC=ax7.imshow(P.reshape(F_C))
#     plt.gca().invert_yaxis()
#     ax7.title.set_text('$C$')
#     plt.colorbar(imgC,orientation='horizontal',ax=ax7)
    
#     imgh=ax8.imshow(P.reshape(F_h))
#     plt.gca().invert_yaxis()
#     ax8.title.set_text('$h$')
#     plt.colorbar(imgh,orientation='horizontal',ax=ax8)
    
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
        ax4.plot(P.reshape(P.ICh0).mean(0),'k--')
        ax4.plot(harr[0,:],'r-')
        ax4.plot(harr[int(P.Ny/2),:],'r-')
        ax6.plot(harr.mean(1))
        ax6.plot(P.reshape(P.ICh0).mean(1),'k--')
        ax6.plot(harr[:,int(P.Nx/2)],'r-')


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

def Sediment_transport_plot():
        # fC  =           P.a*P.k*(
        #                     A_xy*C + P.lambda_d*(
        #                                         C*( LxD*beta(h)*LxD*h + LyD*beta(h)*LyD*h )
        #                                         + beta(h)*(LxD*h*LxD*C+LyD*h*LyD*C)
        #                                         +C*beta(h)*A_xy*h
        #                                         )
        #                     )
        Sediment_transport_x = -P.Mutilde*P.Lx/P.Ly*F.LxD*h +P.delta_s*P.a*P.k*((P.Lx/P.Ly*F.LxD*C+P.lambda_d*C*F.beta(h)*F.LxD*h))
        #Sediment_transport_x = (P.Interior+P.NorthBoundary+P.SouthBoundary)*(P.a*P.k*(F.LxD*C+P.lambda_d*(C*F.beta(h)*F.LxD*h)))
        #Sediment_transport_x += P.WestBoundary*(P.a*P.k*(F.LxD_f*C+P.lambda_d*(C*F.beta(0)*F.LxD_f*h)))
        #Sediment_transport_x += P.EastBoundary*(P.a*P.k*(F.LxD_b*C+P.lambda_d*(C*F.beta(1-P.H2/P.H1)*F.LxD_b*h)))
        Sediment_transport_y = -P.Mutilde*P.Lx/P.Ly*F.LyD*h +P.delta_s*P.a*P.k*((P.Lx/P.Ly*F.LyD*C+P.lambda_d*C*F.beta(h)*F.LyD*h))
        #Sediment_transport_y = (P.Interior+P.WestBoundary+P.EastBoundary)*(P.a*P.k*(F.LyD*C+P.lambda_d*(C*F.beta(h)*F.LyD*h)))
        #Sediment_transport_y += P.SouthBoundary*(P.a*P.k*(F.LyD_f*C+P.lambda_d*(C*F.beta(h)*F.LyD_f*h)))
        #Sediment_transport_y += P.NorthBoundary*(P.a*P.k*(F.LyD_b*C+P.lambda_d*(C*F.beta(h)*F.LyD_b*h)))
        
        fig, ((ax1, ax2),(ax3, ax4),(ax5, ax6)) = plt.subplots(3, 2)
        
        # initialization of the movie figure
        Carr = P.reshape(Sediment_transport_x)
        harr = P.reshape(Sediment_transport_y)
        

        
        imgC = ax1.imshow(Carr,extent=[0,1,1,0],interpolation='none',aspect='auto')
        
        ax1.title.set_text('sedimen transport x direction')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgC,orientation='horizontal',ax=ax1)
        
        ax3.plot(Carr.mean(0))
        ax5.plot(Carr.mean(1))
        
        imgh = ax2.imshow(harr,extent=[0,1,1,0],interpolation='none',aspect='auto')
        
        ax2.title.set_text('sedimen transport y direction')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgh,orientation='horizontal',ax=ax2)
        
        ax4.plot(harr.mean(0))

        ax6.plot(harr.mean(1))


staticplots()

Animation2()

Candhplots()

Sediment_transport_plot()

if BOOL_Total_Model:
    X = np.linspace(0, 1, P.Nx+1)
    Y = np.linspace(0, 1, P.Ny+1)
    Z = P.reshape(h)

    
    fig, ax = plt.subplots()
    ax.plot(X,Z.mean(0),'k-')
    ax.plot(X,P.reshape(P.ICh0).mean(0),'k--')
 
    ax.legend(['2D-model','1D-model'])
    ax.set_title('steady-state bed profile width averaged $\overline{h}(x)$')
    plt.show()
    
    import matplotlib.cm as cm
    fig, ax = plt.subplots()
    im = ax.imshow(Z, interpolation='None', origin='lower',
                    cmap=cm.gray, extent=(0, 1, 0,1),aspect='auto')
    levels = np.arange(np.min(Z), np.max(Z), (np.max(Z)-np.min(Z))/9)
    CS = ax.contour(Z, levels, origin='lower', cmap='gist_yarg',
                    linewidths=2, extent=(0, 1, 0, 1))
    
    # Thicken the zero contour.

    
    ax.clabel(CS, levels,  # label every second level
              inline=1, fmt='%1.2f', fontsize=14)
    
    # make a colorbar for the contour lines
    
    ax.set_title('steady-state bed profile $h(x,y)$')
    
    # We can still add a colorbar for the image, too.
    CBI = fig.colorbar(im, orientation='horizontal')
    
    # This makes the original colorbar look a bit out of place,
    # so let's improve its position.
    
    PLOT_l, PLOT_b, PLOT_w, PLOT_h = ax.get_position().bounds


    plt.show()
    
if False:
    Num_eigenvalues=10
    EIGvals, EIGvecs=sp.linalg.eigs(NJ,Num_eigenvalues)
    print('\t \t ---------- EIGEN VALUE analysis on last jacobian---------  \t')
    print('\t  %i numerically determined eigen vals analysed ' %(Num_eigenvalues))
    print('\t  largest real part of  %1.2e \t largest real part of  %1.2e ' %(np.max(EIGvals.real),np.min(EIGvals.real)))
    
    fig, (ax1,ax2) = plt.subplots(1,2)
    
    ax1.boxplot(EIGvals.real)
    ax1.set_title('real eigen values')
    ax2.boxplot(EIGvals.imag)
    ax2.set_title('imaginary eigen values')
    
    generictotalplot(EIGvecs.real[:,0])
    

    
    
print('\t ------------------------------')
