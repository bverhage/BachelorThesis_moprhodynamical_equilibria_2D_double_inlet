# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:16:14 2021

@author: billy Verhage 
"""

if True:
    print('\t ---------------------------------------------------- ')
    print('\n \t Welcome to the code of BEP 2D FD Morphosdynamic al model \n ')

    import numpy as np
    import scipy.sparse.linalg as la
    import matplotlib.pyplot as plt
 
    import Model_parameters as P
    from scipy import optimize
    
    plt.close("all")
    
P.printparamters()  

BOOL_Total_Model, BOOL_Only_Water_model, BOOL_Only_Concentration_model = True,False,False



if BOOL_Total_Model:
    import Model_Numerical_Jacobian.Model_Numerical_Jacobian_total_model as TM
    #import Model_Numerical_Jacobian_concentration_model as TM_con
    #Uinnitalguess_con = np.concatenate((P.ICzeta0,P.ICzeta0,P.ICu0,P.ICu0,P.ICv0,P.ICv0,P.ICC0))
    
    print('\n \t ====Morphodynamical model====\n')
    
    path=r'DATA\Equilibria_variation_of_paramters\Varied-B'
    
    #n_LIST=np.array([10,11])#,24,22,20,18,16,14,12,10,8,6,4,2,0])
    n_LIST=np.arange(10,40,1)
    LIST=[]
    for n in n_LIST:
        if P.Nx==61:
            LIST.append(str(path+r'\UL%i_Nx60_test_Ny3.npy'%(n)))
        else:
            print('\n \t \t WRONG Nx \n ')  
    if P.Ny!=4:
        print('\n \t This experiment only works when Ny=3 \n \t please go to Model_pramaters.py and change Ny \n')
        
        
    

''' Initial condition '''


  

# the run
lambda_list=np.zeros(n_LIST.shape)
h_list=np.zeros(n_LIST.shape)
F_list=np.zeros(n_LIST.shape)
for i in range(np.size(LIST)):
    print('\n ====================================================== \n \n \t -- currently at %i grad-- \n \n \t loading file ' %(n_LIST[i]),LIST[i-1])
    if i==0:
        U = np.load(LIST[i])
    else:
        U = np.load(LIST[i-1])

    Fail1=False
    Fail2=False
    
    
    #Ufinal=optimize.fsolve(TM.F, Uinnitalguess-la.spsolve(TM.NumericalJacobian(Uinnitalguess),TM.F(Uinnitalguess)))
    def func(U):
        return TM.F(U,phi=np.pi*90/180,Ar=P.A, H2=P.H2 , B=P.Ly*n_LIST[i]/10)
    Ufinal=optimize.fsolve(func, U)
    j=0
    epsilon=10**(-8)
    Check=TM.MaxNormOfU(TM.F(U,phi=np.pi*90/180,Ar=P.A, H2=P.H2 ,B=P.Ly*n_LIST[i]/10))
    while Check>epsilon:
        
        print('\ fsolve loop \n j=%i \t ||F(U)||_max = %.2e < %.2e \n ' %(j,TM.MaxNormOfU(TM.F(Ufinal,phi=np.pi*90/180,Ar=P.A, H2=P.H2 ,B=P.Ly*n_LIST[i]/10)),epsilon))
        
        F_zetas,F_zetac,F_us,F_uc,F_vs,F_vc,F_C,F_h=TM.split_animation(TM.F(Ufinal,phi=np.pi*90/180,Ar=P.A, H2=P.H2 ,B=P.Ly*n_LIST[i]/10))
        
        print('\t | F(zetas) = %.2e \t | F(zetac) = %.2e \n \t | F(us) = %.2e \t | F(uc) = %.2e \n \t | F(vs) = %.2e \t | F(vc) = %.2e \n \t | F(C) = %.2e \t | F(h) = %.2e \n ' %(np.linalg.norm(F_zetas),np.linalg.norm(F_zetac),np.linalg.norm(F_us),np.linalg.norm(F_uc),np.linalg.norm(F_vs),np.linalg.norm(F_vc),np.linalg.norm(F_C),np.linalg.norm(F_h)))
        
        NJ=TM.NumericalJacobian(Ufinal,phi=np.pi*90/180,Ar=P.A, H2=P.H2 ,B=P.Ly*n_LIST[i]/10)
        DeltaU=la.spsolve(NJ,TM.F(Ufinal,phi=np.pi*90/180,Ar=P.A, H2=P.H2 ,B=P.Ly*n_LIST[i]/10))
    
        Ufinal_hat10=Ufinal-1*DeltaU
        Ufinal_hat075=Ufinal-0.75*DeltaU  
        Ufinal_hat05=Ufinal-0.5*DeltaU  
        Ufinal_hat025=Ufinal-0.25*DeltaU  
        Ufinal_hat01=Ufinal-0.1*DeltaU  
        Ufinal_hat001=Ufinal-0.01*DeltaU 
        
        F1=TM.MaxNormOfU(TM.F(Ufinal_hat10,phi=np.pi*90/180,Ar=P.A, H2=P.H2 ,B=P.Ly*n_LIST[i]/10));
        F2=TM.MaxNormOfU(TM.F(Ufinal_hat075,phi=np.pi*90/180,Ar=P.A, H2=P.H2 ,B=P.Ly*n_LIST[i]/10));
        F3=TM.MaxNormOfU(TM.F(Ufinal_hat05,phi=np.pi*90/180,Ar=P.A, H2=P.H2 ,B=P.Ly*n_LIST[i]/10));
        F4=TM.MaxNormOfU(TM.F(Ufinal_hat025,phi=np.pi*90/180,Ar=P.A, H2=P.H2 ,B=P.Ly*n_LIST[i]/10));
        F21=TM.MaxNormOfU(TM.F(Ufinal_hat01,phi=np.pi*90/180,Ar=P.A, H2=P.H2 ,B=P.Ly*n_LIST[i]/10))
        F22=TM.MaxNormOfU(TM.F(Ufinal_hat001,phi=np.pi*90/180,Ar=P.A, H2=P.H2 ,B=P.Ly*n_LIST[i]/10))
        if min(F1,F2,F3,F4,F21,F22)==F1:
             Ufinal=Ufinal_hat10
             print('\t a=1')
        elif min(F1,F2,F3,F4,F21,F22)==F2:
             Ufinal=Ufinal_hat075
             print('\t a=0.75')
        elif min(F1,F2,F3,F4,F21,F22)==F3:
             Ufinal=Ufinal_hat05
             print('\t a=0.5')
        elif min(F1,F2,F3,F4,F21,F22)==F4:
             Ufinal=Ufinal_hat025
             print('\t a=0.25')
        elif min(F1,F2,F3,F4,F21,F22)==F21:
             Ufinal=Ufinal_hat01
             print('\t a=0.1')  
        elif min(F1,F2,F3,F4,F21,F22)==F22:
             Ufinal=Ufinal_hat001
             print('\t a=0.01') 
        if min(F1,F2,F3,F4,F21,F22) > Check:
                 print('\n \t --- WARNING GOING THE WRONG WAY ---')     
        Ufinal=optimize.fsolve(func, Ufinal)
        
        if j>=200:
            Fail1=True
            break
        
        j+=1
        
        Check=TM.MaxNormOfU(TM.F(Ufinal,phi=np.pi*90/180,Ar=P.A, H2=P.H2 ,B=P.Ly*n_LIST[i]/10))
        
    zetas,zetac,us,uc,vs,vc,C,h=TM.split_animation(Ufinal)
    print('\t fsolve loop stoped \n \t ||F(U)||_max = %.2e \n' %(TM.MaxNormOfU(TM.F(Ufinal,phi=np.pi*90/180,Ar=P.A, H2=P.H2 ,B=P.Ly*n_LIST[i]/10))))
    F_zetas,F_zetac,F_us,F_uc,F_vs,F_vc,F_C,F_h=TM.split_animation(TM.F(Ufinal,phi=np.pi*90/180,Ar=P.A, H2=P.H2 ,B=P.Ly*n_LIST[i]/10))
    print('\t | F(zetas) = %.2e \t | F(zetac) = %.2e \n \t | F(us) = %.2e \t | F(uc) = %.2e \n \t | F(vs) = %.2e \t | F(vc) = %.2e \n \t | F(C) = %.2e \t | F(h) = %.2e \n ' %(np.linalg.norm(F_zetas),np.linalg.norm(F_zetac),np.linalg.norm(F_us),np.linalg.norm(F_uc),np.linalg.norm(F_vs),np.linalg.norm(F_vc),np.linalg.norm(F_C),np.linalg.norm(F_h)))
    
    
    #Num_eigenvalues=5
    
    #EIGvals, EIGvecs=sp.linalg.eigs(TM.NumericalJacobian(Ufinal,phi=np.pi*n_LIST[i]/180),Num_eigenvalues,which='LM')
    
    h_list[i]=np.max(h)
    F_list[i]=TM.MaxNormOfU(TM.F(Ufinal,phi=np.pi*90/180,Ar=P.A, H2=P.H2 ,B=P.Ly*n_LIST[i]/10))
    #lambda_list[i]=np.max(EIGvals.real)
    
    #print('with max eigen value %3.2f \n and min water depth %1.2f \n with F = %e  ' %(lambda_list[i],np.max(h), F_list[i]))
    
    SAVE_BOOL=True
    if np.min(C)<0 :
        print('\n ------- \t  Negative Concentration! \t ---------\n')
        print('min(C) = %.2e\n ' %(np.min(C)))
        SAVE_BOOL=False
            
    if np.min(1-h+P.epsilon*zetac)<0 or np.min(1-h+P.epsilon*zetas)<0 :
        print('\n ------- \t  NON physical water model! \t ---------\n')
        SAVE_BOOL=False

    if SAVE_BOOL==True:
        np.save(LIST[i],Ufinal)
        print('\t saved ',LIST[i] )
    
    

fig, (ax1,ax2,ax3)= plt.subplots(1,3)
ax1.plot(n_LIST,lambda_list,'k.')
ax1.set_xlabel('$\phi$')
ax1.set_ylabel('$\max(\lambda)$')
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.hlines(-1,np.min(n_LIST),np.max(n_LIST),'k')
ax1.set_title('$\max(\lambda)')
ax1.legend({'y=-1','$\lambda$'})

ax2.plot(n_LIST,h_list,'k.')
ax2.set_xlabel('$\phi$')
ax2.set_ylabel('h')
ax2.set_title('min water depth')

ax3.set_title('F(U)')
ax3.plot(n_LIST,F_list,'k.')
ax3.set_xlabel('$\phi$')
ax3.set_ylabel('$F(U)$')
             

                                           
# Check=np.where(TM.F(Ufinal)!=0)
# if Check[0].size>0 :
#     print('\t ------- F=!0 ------- \t \n')
#     F_zetas,F_zetac,F_us,F_uc,F_vs,F_vc,F_C,F_h=TM.split_animation(TM.F(Ufinal))
    
#     fig, ((ax1,ax3,ax5,ax7),(ax2,ax4,ax6,ax8)) = plt.subplots(2,4)
#     plt.suptitle('F(U)') 
#     plt.gca().invert_yaxis()
#     imgzetas = ax1.imshow(P.reshape(F_zetas))
#     ax1.title.set_text('$F(\zeta^{s})$')
#     plt.colorbar(imgzetas,orientation='horizontal',ax=ax1)
    
#     imgzetac = ax2.imshow(P.reshape(F_zetac))
#     plt.gca().invert_yaxis()
#     ax2.title.set_text('$F(\zeta^{c})$')
#     plt.colorbar(imgzetac,orientation='horizontal',ax=ax2)
    
#     imgus = ax3.imshow(P.reshape(F_us))
#     plt.gca().invert_yaxis()
#     ax3.title.set_text('$F(u^{s})$')
#     plt.colorbar(imgus,orientation='horizontal',ax=ax3)
    
#     imguc = ax4.imshow(P.reshape(F_uc))
#     plt.gca().invert_yaxis()
#     ax4.title.set_text('$F(u^{c})$')
#     plt.colorbar(imguc,orientation='horizontal',ax=ax4)
    
#     imgvs=ax5.imshow(P.reshape(F_vs))
#     plt.gca().invert_yaxis()
#     ax5.title.set_text('$F(vs)$')
#     plt.colorbar(imgvs,orientation='horizontal',ax=ax5)
    
#     imgvc=ax6.imshow(P.reshape(F_vc))
#     plt.gca().invert_yaxis()
#     ax6.title.set_text('$F(vc)$')
#     plt.colorbar(imgvc,orientation='horizontal',ax=ax6)
    
#     imgC=ax7.imshow(P.reshape(F_C))
#     plt.gca().invert_yaxis()
#     ax7.title.set_text('$F(C)$')
#     plt.colorbar(imgC,orientation='horizontal',ax=ax7)
    
#     imgh=ax8.imshow(P.reshape(F_h))
#     plt.gca().invert_yaxis()
#     ax8.title.set_text('$F(h)$')
#     plt.colorbar(imgh,orientation='horizontal',ax=ax8)
    


# ''' the plots '''
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
    
# def staticplots():
#         plt.ion()
        
#         fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3)
        

#         # initialization of the movie figure
#         zetasarr = P.reshape(zetas)
#         zetacarr = P.reshape(zetac)
#         usarr = P.reshape(us)
#         ucarr = P.reshape(uc)
#         vsarr = P.reshape(vs)
#         vcarr = P.reshape(vc)
        
        
    
        
#         imgzetas = ax1.imshow(zetasarr,extent=[P.dx/2,1-P.dx/2,1-P.dy/2,P.dy/2],interpolation='none',aspect='auto')
        
#         ax1.title.set_text('zeta-sin')
#         plt.gca().invert_yaxis()
        
#         plt.colorbar(imgzetas,orientation='horizontal',ax=ax1)
        
#         imgzetac = ax4.imshow(zetacarr,extent=[P.dx/2,1-P.dx/2,1-P.dy/2,P.dy/2],interpolation='none',aspect='auto')
        
#         ax4.title.set_text('zeta-cos')
#         plt.gca().invert_yaxis()
        
#         plt.colorbar(imgzetac,orientation='horizontal',ax=ax4)
     
#         imgus = ax2.imshow(usarr,extent=[P.dx/2,1-P.dx/2,1-P.dy/2,P.dy/2],interpolation='none',aspect='auto')
        
#         ax2.title.set_text('u-sin')
#         plt.gca().invert_yaxis()
        
#         plt.colorbar(imgus,orientation='horizontal',ax=ax2)
        
#         imguc = ax5.imshow(ucarr,extent=[P.dx/2,1-P.dx/2,1-P.dy/2,P.dy/2],interpolation='none',aspect='auto')
        
#         ax5.title.set_text('u-cos')
#         plt.gca().invert_yaxis()
        
#         plt.colorbar(imguc,orientation='horizontal',ax=ax5)
        
#         imgvs = ax3.imshow(vsarr,extent=[P.dx/2,1-P.dx/2,1-P.dy/2,P.dy/2],interpolation='none',aspect='auto')
        
#         ax3.title.set_text('v-sin')
#         plt.gca().invert_yaxis()
        
#         plt.colorbar(imgvs,orientation='horizontal',ax=ax3)
        
#         imgvc = ax6.imshow(vcarr,extent=[P.dx/2,1-P.dx/2,1-P.dy/2,P.dy/2],interpolation='none',aspect='auto')
        
#         ax6.title.set_text('v-cos')
#         plt.gca().invert_yaxis()
        
#         plt.colorbar(imgvc,orientation='horizontal',ax=ax6)
     
# def Candhplots():
#         plt.ion()
#         plt.suptitle('Steady State bed, tidal average Concentration')
                     
#         fig, ((ax1, ax2),(ax3, ax4),(ax5, ax6)) = plt.subplots(3, 2)
        
#         # initialization of the movie figure
#         Carr = P.reshape(C)
#         harr = P.reshape(h)
        

        
#         imgC = ax1.imshow(Carr,extent=[0,1,1,0],interpolation='none',aspect='auto')
        
#         ax1.title.set_text('C')
#         plt.gca().invert_yaxis()
        
#         plt.colorbar(imgC,orientation='horizontal',ax=ax1)
        
#         ax3.plot(Carr.mean(0))
#         ax5.plot(Carr.mean(1))
        
#         imgh = ax2.imshow(harr,extent=[0,1,1,0],interpolation='none',aspect='auto')
        
#         ax2.title.set_text('h')
#         plt.gca().invert_yaxis()
        
#         plt.colorbar(imgh,orientation='horizontal',ax=ax2)
        
#         ax4.plot(harr.mean(0))
#         ax4.plot(P.reshape(P.ICh0).mean(0),'k--')
#         ax4.plot(harr[0,:],'r-')
#         ax4.plot(harr[int(P.Ny/2),:],'r-')
#         ax6.plot(harr.mean(1))
#         ax6.plot(P.reshape(P.ICh0).mean(1),'k--')
#         ax6.plot(harr[:,int(P.Nx/2)],'r-')


# def Animation1():
#         t = 0
    
#         plt.ion()
        
#         fig, ((ax1, ax2)) = plt.subplots(2)
        
#         # Inital conditoin 
#         zeta0=zetas*np.sin(t)+zetac*np.cos(t)
#         u0=uc
        

#         # initialization of the movie figure
#         zeta0arr = zeta0
#         u0arr = u0
        

        
        
#         imgzetacross = ax1.plot(zeta0arr,'k.',markersize=1)
#         ax1.title.set_text('zeta ')
#         ax1.set_ylim([-P.Atilde,P.Atilde])
        
#         imgucross = ax2.plot(u0arr,'k.',markersize=1)
#         ax2.title.set_text('u')
#         #ax2.set_ylim([-0.2,0.2])
        


        
        
#         tlt = plt.suptitle('t = %3.3f' %(t))
        
#         Tend=1
#         NSteps=24*10
#         anim_dt=Tend/NSteps
        
#         def animate(frame):
#             '''
#             This function updates the solution array
#             '''
#             global t, Nx, Ny
#             t = (frame+1)*anim_dt
                        
#             zeta1=zetas*np.sin(2*np.pi*t)+zetac*np.cos(2*np.pi*t)
#             u1=us*np.sin(2*np.pi*t)+uc*np.cos(2*np.pi*t)
            
                
            
#             imgzetacross[0].set_ydata(zeta1)
#             imgucross[0].set_ydata(u1)



            
            
#             tlt.set_text('t = %3.3f' %(t))
        
                                                
#             return imgzetacross,imgucross
        
#         # figure animation
        
#         return animation.FuncAnimation(fig , animate  , interval=50 , repeat=False)
  
    
# def Animation2():
#     t = 0
    
#     plt.ion()
        
#     fig, ((ax1,ax12, ax2)) = plt.subplots(3)
#     # Inital conditoin 
#     zeta0=zetas*np.sin(t)+zetac*np.cos(t)
#     u0=uc
#     v0=vc

#     # initialization of the movie figure
#     zeta0arr = P.reshape(zeta0)
#     u0arr    = P.reshape(u0)
#     v0arr    = P.reshape(v0)
        
#     X = np.linspace(0, 1, P.Nx+1)
#     Y = np.linspace(0, 1, P.Ny+1)
#     U, V =v0arr, u0arr
    
#     imgzeta = ax1.imshow(zeta0arr,extent=[P.dx/2,1-P.dx/2,1-P.dy/2,P.dy/2],interpolation='none',aspect='auto')
        
#     ax1.title.set_text('zeta')
#     plt.gca().invert_yaxis()
    
#     plt.colorbar(imgzeta,orientation='horizontal',ax=ax1)
    

#     imgzetacross = ax12.plot(zeta0arr.mean(0),linewidth=1,color='k')
#     ax12.set_ylim([-(P.Atilde+1),(P.Atilde+1)])  
    
#     ax2.quiver(X, Y, U, V, units='width')
#     #ax2.quiverkey(q, 0.1, 0.1, 0.01, r'$2 \frac{m}{s}$', labelpos='E',
#     #               coordinates='figure')
    
#     tlt = plt.suptitle('t = %3.3f' %(t))
        
#     Tend=1
#     NSteps=24*10
#     anim_dt=Tend/NSteps
        
#     def animate(frame):
#         '''
#         This function updates the solution array
#         '''
#         t = (frame+1)*anim_dt
                        
#         zeta1=zetas*np.sin(2*np.pi*t)+zetac*np.cos(2*np.pi*t)
#         u1=us*np.sin(2*np.pi*t)+uc*np.cos(2*np.pi*t)
#         v1=vs*np.sin(2*np.pi*t)+vc*np.cos(2*np.pi*t)
            
#         imgzetacross[0].set_ydata(P.reshape(zeta1).mean(0))       
#         imgzeta.set_array(P.reshape(zeta1))
#         imgzeta.set_clim(zeta1.min(),zeta1.max())
        
#         ax2.clear()
#         ax2.quiver(X, Y, P.reshape(u1), P.reshape(v1), P.reshape(np.sqrt(u1**2+v1**2)),pivot='mid',units='dots',headwidth=0.1,linewidth=0.1,headlength=0.1)
#         #ax2.quiverkey(q, 0.1, 0.1, 0.01)
            
#         tlt.set_text('t = %3.3f' %(t))
      
                                                
#         return imgzeta,imgzetacross,ax2
        
#         # figure animation
        
#     return animation.FuncAnimation(fig , animate  , interval=50 , repeat=True)

# def Sediment_transport_plot():
#         # fC  =           P.a*P.k*(
#         #                     A_xy*C + P.lambda_d*(
#         #                                         C*( LxD*beta(h)*LxD*h + LyD*beta(h)*LyD*h )
#         #                                         + beta(h)*(LxD*h*LxD*C+LyD*h*LyD*C)
#         #                                         +C*beta(h)*A_xy*h
#         #                                         )
#         #                     )
#         Sediment_transport_x = -P.Mutilde*P.Lx/P.Ly*F.LxD*h +P.delta_s*P.a*P.k*((P.Lx/P.Ly*F.LxD*C+P.lambda_d*C*F.beta(h)*F.LxD*h))
#         #Sediment_transport_x = (P.Interior+P.NorthBoundary+P.SouthBoundary)*(P.a*P.k*(F.LxD*C+P.lambda_d*(C*F.beta(h)*F.LxD*h)))
#         #Sediment_transport_x += P.WestBoundary*(P.a*P.k*(F.LxD_f*C+P.lambda_d*(C*F.beta(0)*F.LxD_f*h)))
#         #Sediment_transport_x += P.EastBoundary*(P.a*P.k*(F.LxD_b*C+P.lambda_d*(C*F.beta(1-P.H2/P.H1)*F.LxD_b*h)))
#         Sediment_transport_y = -P.Mutilde*P.Lx/P.Ly*F.LyD*h +P.delta_s*P.a*P.k*((P.Lx/P.Ly*F.LyD*C+P.lambda_d*C*F.beta(h)*F.LyD*h))
#         #Sediment_transport_y = (P.Interior+P.WestBoundary+P.EastBoundary)*(P.a*P.k*(F.LyD*C+P.lambda_d*(C*F.beta(h)*F.LyD*h)))
#         #Sediment_transport_y += P.SouthBoundary*(P.a*P.k*(F.LyD_f*C+P.lambda_d*(C*F.beta(h)*F.LyD_f*h)))
#         #Sediment_transport_y += P.NorthBoundary*(P.a*P.k*(F.LyD_b*C+P.lambda_d*(C*F.beta(h)*F.LyD_b*h)))
        
#         fig, ((ax1, ax2),(ax3, ax4),(ax5, ax6)) = plt.subplots(3, 2)
        
#         # initialization of the movie figure
#         Carr = P.reshape(Sediment_transport_x)
#         harr = P.reshape(Sediment_transport_y)
        

        
#         imgC = ax1.imshow(Carr,extent=[0,1,1,0],interpolation='none',aspect='auto')
        
#         ax1.title.set_text('sedimen transport x direction')
#         plt.gca().invert_yaxis()
        
#         plt.colorbar(imgC,orientation='horizontal',ax=ax1)
        
#         ax3.plot(Carr.mean(0))
#         ax5.plot(Carr.mean(1))
        
#         imgh = ax2.imshow(harr,extent=[0,1,1,0],interpolation='none',aspect='auto')
        
#         ax2.title.set_text('sedimen transport y direction')
#         plt.gca().invert_yaxis()
        
#         plt.colorbar(imgh,orientation='horizontal',ax=ax2)
        
#         ax4.plot(harr.mean(0))

#         ax6.plot(harr.mean(1))


# staticplots()

# Animation2()

# Candhplots()

# Sediment_transport_plot()

# if BOOL_Total_Model:
#     X = np.linspace(0, 1, P.Nx+1)
#     Y = np.linspace(0, 1, P.Ny+1)
#     Z = P.reshape(h)

    
#     fig, ax = plt.subplots()
#     ax.plot(X,Z.mean(0),'k-')
#     ax.plot(X,P.reshape(P.ICh0).mean(0),'k--')
 
#     ax.legend(['2D-model','1D-model'])
#     ax.set_title('steady-state bed profile width averaged $\overline{h}(x)$')
#     plt.show()
    
#     import matplotlib.cm as cm
#     fig, ax = plt.subplots()
#     im = ax.imshow(Z, interpolation='None', origin='lower',
#                     cmap=cm.gray, extent=(0, 1, 0,1),aspect='auto')
#     levels = np.arange(np.min(Z), np.max(Z), (np.max(Z)-np.min(Z))/9)
#     CS = ax.contour(Z, levels, origin='lower', cmap='gist_yarg',
#                     linewidths=2, extent=(0, 1, 0, 1))
    
#     # Thicken the zero contour.

    
#     ax.clabel(CS, levels,  # label every second level
#               inline=1, fmt='%1.2f', fontsize=14)
    
#     # make a colorbar for the contour lines
    
#     ax.set_title('steady-state bed profile $h(x,y)$')
    
#     # We can still add a colorbar for the image, too.
#     CBI = fig.colorbar(im, orientation='horizontal')
    
#     # This makes the original colorbar look a bit out of place,
#     # so let's improve its position.
    
#     PLOT_l, PLOT_b, PLOT_w, PLOT_h = ax.get_position().bounds


#     plt.show()
    
# if True:
#     Num_eigenvalues=5
#     EIGvals, EIGvecs=sp.linalg.eigs(TM.NumericalJacobian(Ufinal),Num_eigenvalues,which='LR')
#     print('\t \t ---------- EIGEN VALUE analysis on last jacobian---------  \t')
#     print('\t  %i numerically determined eigen vals analysed ' %(Num_eigenvalues))
#     print('\t  largest real part of  %1.2e \t largest real part of  %1.2e ' %(np.max(EIGvals),np.min(EIGvals)))
    
#     # fig, (ax1,ax2) = plt.subplots(1,2)
    
#     # ax1.boxplot(EIGvals.real)
#     # ax1.set_title('real eigen values')
#     # ax2.boxplot(EIGvals.imag)
#     # ax2.set_title('imaginary eigen values')
    
#     # generictotalplot(EIGvecs.real[:,0])

#     plot_information=np.array([[P.phi.real],[np.max(h.real)],[TM.MaxNormOfU(TM.F(Ufinal.real))],[np.max(EIGvals.real)]])
    
#     fig, (ax1,ax2,ax3)= plt.subplots(1,3)
#     ax1.plot(plot_information[0,:],plot_information[3,:],'k.')
#     ax1.set_xlabel('$\phi$')
#     ax1.set_ylabel('$\max(\lambda$)')
#     ax1.hlines(0,np.min(plot_information[:,0]),np.max(plot_information[:,0]),'k')
    
#     ax2.plot(plot_information[0,:],plot_information[1,:],'k.')
#     ax2.set_xlabel('$\phi$')
#     ax2.set_ylabel('h')
    
#     ax3.plot(plot_information[0,:],plot_information[2,:],'k.')
#     ax3.set_xlabel('$\phi$')
#     ax3.set_ylabel('$F(U)$')
    
# if False:
#     #n_LIST=[90,88,86,84,82,80,78,76,74,72,70,68,66,64,62,60,58,56,54,52,50,48,46,44,42,40,38,36,34,32,30,28,26,24,22,20,18,16,14,12,10,8,6,4,2,0]
#     n_LIST=[90,88,86,84,82,80,78,76,74,72,70,68,66,64,62,60,58,56,54,52,50,48,46,44,42,40,38,36,34]
#     LIST=[]
#     for n in n_LIST:
#         LIST.append(str('Uphi%i_Ny9.npy'%(n)))
#     harr    = P.reshape(h)
#     h_list=np.zeros((np.size(LIST),np.size(harr.mean(0))))
#     Carr    = P.reshape(C)
#     C_list=np.zeros((np.size(LIST),np.size(Carr.mean(0))))
    
#     for i in range(np.size(LIST)):
#         U = np.load(LIST[i])
#         zetas,zetac,us,uc,vs,vc,C,h=TM.split_animation(U)
#         h_list[i,:]= P.reshape(h).mean(0)
#         C_list[i,:]= P.reshape(C).mean(0)
        
#     fig, (ax1,ax2 )= plt.subplots(1,2)
#     imgh=ax1.imshow(h_list, interpolation='None', origin='lower',
#                     cmap=cm.gray, extent=(0, 1, np.max(n_LIST),np.min(n_LIST)),aspect='auto')
#     #plt.set_ylabel([np.max(n_LIST),np.min(n_LIST)])
#     levels = np.arange(np.min(h_list), np.max(h_list), (np.max(h_list)-np.min(h_list))/15)
#     CS = ax1.contour(h_list, levels, origin='lower', cmap='gist_yarg',
#                     linewidths=2, extent=(0, 1, np.max(n_LIST),np.min(n_LIST)))
#     ax1.clabel(CS, levels,  # label every second level
#               inline=1, fmt='%1.2f', fontsize=14)
#     ax1.set_title('steady-state bed profile $h$ vs $\Delta\phi$')

    
#     # We can still add a colorbar for the image, too.
#     CBI = fig.colorbar(imgh, orientation='horizontal',ax=ax1)
    
#     imgC=ax2.imshow(C_list, interpolation='None', origin='lower',
#                     cmap=cm.gray, extent=(0, 1, np.max(n_LIST),np.min(n_LIST)),aspect='auto')
#     #plt.set_ylabel([np.max(n_LIST),np.min(n_LIST)])
#     levels = np.arange(np.min(C_list), np.max(C_list), (np.max(C_list)-np.min(C_list))/15)
#     CS = ax2.contour(C_list, levels, origin='lower', cmap='gist_yarg',
#                     linewidths=2, extent=(0, 1, np.max(n_LIST),np.min(n_LIST)))
#     ax2.clabel(CS, levels,  # label every second level
#               inline=1, fmt='%1.2f', fontsize=14)
#     ax2.set_title('steady-state bed profile $C$ vs $\Delta\phi$')

    
#     # We can still add a colorbar for the image, too.
#     CBI = fig.colorbar(imgC, orientation='horizontal',ax=ax2)
    
#     fig, (ax1)= plt.subplots(1)
#     plt.plot(n_LIST,h_list.max(1),'b*')
#     plt.plot( np.flip(np.arange(92-2*np.load('h_list_Ny3.npy').shape[0],92,2)),np.load('h_list_Ny3.npy').max(1),'k.')
#     plt.plot( np.flip(np.arange(92-2*np.load('h_list_Ny1.npy').shape[0],92,2)),np.load('h_list_Ny1.npy').max(1),'r.')
#     plt.xlabel('$\phi$')
#     plt.ylabel('h')

# if False:
#     fig, (ax1)= plt.subplots(1)
#     plt.plot( np.flip(np.arange(92-2*np.load('h_list_Ny3.npy').shape[0],92,2)),np.load('h_list_Ny3.npy').max(1),'k.')
#     plt.plot( np.flip(np.arange(92-2*np.load('h_list_Ny1.npy').shape[0],92,2)),np.load('h_list_Ny1.npy').max(1),'r.')
#     plt.xlabel('$\phi$')
#     plt.ylabel('h')
print('\t ------------------------------')
