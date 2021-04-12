# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:43:10 2021

@author: billy
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as animation
 
import Model_parameters as P
import Model_functions as F
from scipy import optimize

import Model_Numerical_Jacobian_total_model as TM

import matplotlib.cm as cm

 
Num_eigenvalues=1

#n_LIST=np.array([90])#,88,86,84,82,80,78,76,74,72,70,68,66,64,62,60,58,56,54,52,50,48,46,44,42,40,38,36,34,32,30,28,26])#,24,22,20,18,16,14,12,10,8,6,4,2,0])
n_LIST= n_LIST=np.flip(np.arange(0,90+1,1))

LIST=[]
for n in n_LIST:
    if P.Nx==31:
        LIST.append(str('Uphi%i_test.npy'%(n)))
    else:
        print('\n \t \t WRONG Nx \n ')
    

lambda_list=np.zeros(n_LIST.shape)
lambda_list_2=np.zeros(n_LIST.shape)
lambda_list_3=np.zeros(n_LIST.shape)
h_list=np.zeros(n_LIST.shape)
F_list=np.zeros(n_LIST.shape)

for i in range(np.size(LIST)):
    
    print('\n \t -- currently at %i grad-- \n ' %(n_LIST[i]))
    
    U = np.load(LIST[i])
   
    if P.Nx>2:
        
        zetas,zetac,us,uc,vs,vc,C,h=TM.split_animation(U)
        
        #zetas = np.reshape(np.block([[np.reshape(zetas,(3,P.Nx+1))],[np.reshape(zetas,(3,P.Nx+1))],[np.reshape(zetas,(3,P.Nx+1))]]),P.ICzeta0.shape)
        zetas = np.reshape(zetas,(3,P.Nx+1))
        newzetas = zetas[0,:]
        for j in range(P.Ny-1):
            newzetas=np.vstack([newzetas,zetas[1,:]])
        newzetas=np.vstack([newzetas,zetas[2,:]])
        zetas = np.reshape(newzetas,P.ICzeta0.shape)
        
        #zetac = np.reshape(np.block([[np.reshape(zetac,(3,P.Nx+1))],[np.reshape(zetac,(3,P.Nx+1))],[np.reshape(zetac,(3,P.Nx+1))]]),P.ICzeta0.shape)
        zetac = np.reshape(zetac,(3,P.Nx+1))
        newzetac = zetac[0,:]
        for j in range(P.Ny-1):
            newzetac=np.vstack([newzetac,zetac[1,:]])
        newzetac=np.vstack([newzetac,zetac[2,:]])
        zetac = np.reshape(newzetac,P.ICzeta0.shape)
        
        #us = np.reshape(np.block([[np.reshape(us,(3,P.Nx+1))],[np.reshape(us,(3,P.Nx+1))],[np.reshape(us,(3,P.Nx+1))]]),P.ICzeta0.shape)
        us = np.reshape(us,(3,P.Nx+1))
        newus = us[0,:]
        for j in range(P.Ny-1):
            newus=np.vstack([newus,us[1,:]])
        newus=np.vstack([newus,us[2,:]])
        us = np.reshape(newus,P.ICzeta0.shape)
        
        #uc = np.reshape(np.block([[np.reshape(uc,(3,P.Nx+1))],[np.reshape(uc,(3,P.Nx+1))],[np.reshape(uc,(3,P.Nx+1))]]),P.ICzeta0.shape)
        uc = np.reshape(uc,(3,P.Nx+1))
        newuc = uc[0,:]
        for j in range(P.Ny-1):
            newuc=np.vstack([newuc,uc[1,:]])
        newuc=np.vstack([newuc,uc[2,:]])
        uc = np.reshape(newuc,P.ICzeta0.shape)
        
        #vs = np.reshape(np.block([[np.reshape(vs,(3,P.Nx+1))],[np.reshape(vs,(3,P.Nx+1))],[np.reshape(vs,(3,P.Nx+1))]]),P.ICzeta0.shape)
        vs = np.reshape(vs,(3,P.Nx+1))
        newvs = vs[0,:]
        for j in range(P.Ny-1):
            newvs=np.vstack([newvs,vs[1,:]])
        newvs=np.vstack([newvs,vs[2,:]])
        vs = np.reshape(newvs,P.ICzeta0.shape)
        
        #vc = np.reshape(np.block([[np.reshape(vc,(3,P.Nx+1))],[np.reshape(vc,(3,P.Nx+1))],[np.reshape(vc,(3,P.Nx+1))]]),P.ICzeta0.shape)
        vc = np.reshape(vc,(3,P.Nx+1))
        newvc = vc[0,:]
        for j in range(P.Ny-1):
            newvc=np.vstack([newvc,vc[1,:]])
        newvc=np.vstack([newvc,vc[2,:]])
        vc = np.reshape(newvc,P.ICzeta0.shape)
        
        #C = np.reshape(np.block([[np.reshape(C,(3,P.Nx+1))],[np.reshape(C,(3,P.Nx+1))],[np.reshape(C,(3,P.Nx+1))]]),P.ICzeta0.shape)
        C = np.reshape(C,(3,P.Nx+1))
        newC = C[0,:]
        for j in range(P.Ny-1):
            newC=np.vstack([newC,C[1,:]])
        newC=np.vstack([newC,C[2,:]])
        C = np.reshape(newC,P.ICzeta0.shape)
        
        #h = np.reshape(np.block([[np.reshape(h,(3,P.Nx+1))],[np.reshape(h,(3,P.Nx+1))],[np.reshape(h,(3,P.Nx+1))]]),P.ICzeta0.shape)
        h = np.reshape(h,(3,P.Nx+1))
        newh = h[0,:]
        for j in range(P.Ny-1):
            newh=np.vstack([newh,h[1,:]])
        newh=np.vstack([newh,h[2,:]])
        h = np.reshape(newh,P.ICzeta0.shape)
        
        U = np.concatenate((zetas,zetac,us,uc,vs,vc,C,h))
        
        
    NJ=TM.NumericalJacobian(U,phi=np.pi*n_LIST[i]/180)
    F_list[i]=TM.MaxNormOfU(TM.F(U,phi=np.pi*n_LIST[i]/180))

    #EIGvals, EIGvecs=sp.linalg.eigsh( NJ.toarray(),NJ.shape[0])#,which='LM')

    EIGvals, EIGvecs=sp.linalg.eigs(NJ,k=1,which='LM')

    lambda_list[i]=np.max(np.real(EIGvals))
    
    zetas,zetac,us,uc,vs,vc,C,h=TM.split_animation(EIGvecs)
    
    eigen_vec_maxh=np.max(np.real(h))
    
    EIGvals, EIGvecs=sp.linalg.eigs(NJ,k=1,which='LR')
    
    lambda_list_2[i]=np.max(np.real(EIGvals))
    
    zetas,zetac,us,uc,vs,vc,C,h=TM.split_animation(EIGvecs)
    
    eigen_vec2_maxh=np.max(np.real(h))
    # EIGvals, EIGvecs=np.linalg.eig(NJ.toarray())
    
    # lambda_list_3[i]=np.max(np.real(EIGvals))
    
    zetas,zetac,us,uc,vs,vc,C,h=TM.split_animation(U)
    
    h_list[i]=np.max(h)
    
    if TM.MaxNormOfU(TM.F(U,phi=np.pi*n_LIST[i]/180)) <  TM.MaxNormOfU(TM.F(U-la.spsolve(NJ,TM.F(U,phi=np.pi*n_LIST[i]/180)),phi=np.pi*n_LIST[i]/180)):
        print('Diverging')
    else:
        print('Converging')
    
    print('with max eigen value %3.2e \n and min water depth %1.2f \n with F = %e  ' %(lambda_list[i]/(P.Nx*P.Ny),np.max(h), F_list[i]))
    print('\n eigen values \n \t LM eigen value %3.2e \t h of eigen vec %3.2e \n \t LR eigen value %3.2e \t h of eigen vec %3.2e  ' %(lambda_list[i]/(P.Nx*P.Ny),eigen_vec_maxh, lambda_list_2[i]/(P.Nx*P.Ny),eigen_vec2_maxh))

fig, (ax1,ax4,ax5,ax2,ax3)= plt.subplots(1,5)
ax1.plot(n_LIST,lambda_list/(P.Nx*P.Ny)**1,'b.')
ax1.set_xlabel('$\phi$')
ax1.set_ylabel('$\max(\lambda)$')
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax1.hlines(-1,np.min(n_LIST),np.max(n_LIST),'k')
ax1.set_title('$\max(\lambda)$')
#ax1.legend({'y=-1','$\lambda$'})

ax2.plot(n_LIST,h_list,'b.')
ax2.set_xlabel('$\phi$')
ax2.set_ylabel('h')
ax2.set_title('min water depth')

ax3.set_title('F(U)')
ax3.plot(n_LIST,F_list,'b.')
ax3.set_xlabel('$\phi$')
ax3.set_ylabel('$F(U)$')

ax4.plot(n_LIST,lambda_list_2/(P.Nx*P.Ny)**1,'b.')
ax4.set_xlabel('$\phi$')
ax4.set_ylabel('$\max(\lambda)$')
ax4.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax4.set_title('$\max(\lambda)$')


ax5.plot(n_LIST,lambda_list/(P.Nx*P.Ny)**2,'b.')
ax5.set_xlabel('$\phi$')
ax5.set_ylabel('$\max(\lambda)$')
ax5.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax5.set_title('$ \max(\lambda)/(N_x^2N_y^2)$')


# fig, (ax1,ax2 )= plt.subplots(1,2)
# imgh=ax1.imshow(h_list, interpolation='None', origin='lower',
#                 cmap=cm.gray, extent=(0, 1, np.max(n_LIST),np.min(n_LIST)),aspect='auto')
# #plt.set_ylabel([np.max(n_LIST),np.min(n_LIST)])
# levels = np.arange(np.min(h_list), np.max(h_list), (np.max(h_list)-np.min(h_list))/15)
# CS = ax1.contour(h_list, levels, origin='lower', cmap='gist_yarg',
#                 linewidths=2, extent=(0, 1, np.max(n_LIST),np.min(n_LIST)))
# ax1.clabel(CS, levels,  # label every second level
#           inline=1, fmt='%1.2f', fontsize=14)
# ax1.set_title('steady-state bed profile $h$ vs $\Delta\phi$')


# # We can still add a colorbar for the image, too.
# CBI = fig.colorbar(imgh, orientation='horizontal',ax=ax1)

# imgC=ax2.imshow(C_list, interpolation='None', origin='lower',
#                 cmap=cm.gray, extent=(0, 1, np.max(n_LIST),np.min(n_LIST)),aspect='auto')
# #plt.set_ylabel([np.max(n_LIST),np.min(n_LIST)])
# levels = np.arange(np.min(C_list), np.max(C_list), (np.max(C_list)-np.min(C_list))/15)
# CS = ax2.contour(C_list, levels, origin='lower', cmap='gist_yarg',
#                 linewidths=2, extent=(0, 1, np.max(n_LIST),np.min(n_LIST)))
# ax2.clabel(CS, levels,  # label every second level
#           inline=1, fmt='%1.2f', fontsize=14)
# ax2.set_title('steady-state bed profile $C$ vs $\Delta\phi$')


# # We can still add a colorbar for the image, too.
# CBI = fig.colorbar(imgC, orientation='horizontal',ax=ax2)

# fig, (ax1)= plt.subplots(1)
# plt.plot(n_LIST,h_list.max(1),'b*')
# plt.plot( np.flip(np.arange(92-2*np.load('h_list_Ny3.npy').shape[0],92,2)),np.load('h_list_Ny3.npy').max(1),'k.')
# plt.plot( np.flip(np.arange(92-2*np.load('h_list_Ny1.npy').shape[0],92,2)),np.load('h_list_Ny1.npy').max(1),'r.')
# plt.xlabel('$\phi$')
# plt.ylabel('h')

# Uinnitalguess = np.load('Uphi90.npy')



# NJ=TM.NumericalJacobian(Uinnitalguess)


# EIGvals, EIGvecs=sp.linalg.eigs(NJ,Num_eigenvalues)
# print('\t \t ---------- EIGEN VALUE analysis on last jacobian---------  \t')
# print('\t  %i numerically determined eigen vals analysed ' %(Num_eigenvalues))
# print('\t  largest real part of  %1.2e \t largest real part of  %1.2e ' %(np.max(EIGvals.real),np.min(EIGvals.real)))

# fig, (ax1,ax2) = plt.subplots(1,2)



# ax1.boxplot(EIGvals.real)
# ax1.set_title('real eigen values')
# ax2.boxplot(EIGvals.imag)
# ax2.set_title('imaginary eigen values')


# F_zetas,F_zetac,F_us,F_uc,F_vs,F_vc,F_C,F_h=TM.split_animation(TM.F(EIGvecs.real[:,0]))
# fig, ((ax1,ax3,ax5,ax7),(ax2,ax4,ax6,ax8)) = plt.subplots(2,4)
# plt.suptitle('total plot') 
# plt.gca().invert_yaxis()
# imgzetas = ax1.imshow(P.reshape(F_zetas))
# ax1.title.set_text('$\zeta^{s}$')
# plt.colorbar(imgzetas,orientation='horizontal',ax=ax1)

# imgzetac = ax2.imshow(P.reshape(F_zetac))
# plt.gca().invert_yaxis()
# ax2.title.set_text('$\zeta^{c}$')
# plt.colorbar(imgzetac,orientation='horizontal',ax=ax2)

# imgus = ax3.imshow(P.reshape(F_us))
# plt.gca().invert_yaxis()
# ax3.title.set_text('$u^{s}$')
# plt.colorbar(imgus,orientation='horizontal',ax=ax3)

# imguc = ax4.imshow(P.reshape(F_uc))
# plt.gca().invert_yaxis()
# ax4.title.set_text('$u^{c}$')
# plt.colorbar(imguc,orientation='horizontal',ax=ax4)

# imgvs=ax5.imshow(P.reshape(F_vs))
# plt.gca().invert_yaxis()
# ax5.title.set_text('$vs$')
# plt.colorbar(imgvs,orientation='horizontal',ax=ax5)

# imgvc=ax6.imshow(P.reshape(F_vc))
# plt.gca().invert_yaxis()
# ax6.title.set_text('$vc$')
# plt.colorbar(imgvc,orientation='horizontal',ax=ax6)

# imgC=ax7.imshow(P.reshape(F_C))
# plt.gca().invert_yaxis()
# ax7.title.set_text('$C$')
# plt.colorbar(imgC,orientation='horizontal',ax=ax7)

# imgh=ax8.imshow(P.reshape(F_h))
# plt.gca().invert_yaxis()
# ax8.title.set_text('$h$')
# plt.colorbar(imgh,orientation='horizontal',ax=ax8)