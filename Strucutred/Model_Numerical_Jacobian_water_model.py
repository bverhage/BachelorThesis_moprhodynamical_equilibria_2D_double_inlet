# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:16:14 2021

@author: billy Verhage 
"""


import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

import Model_functions as func
import Model_parameters as P

from tqdm import tqdm 

h=P.ICh0
C=P.ICC0

def split(U):
    zetas,zetac,us,uc,vs,vc=np.array_split(U,6)
    return zetas,zetac,us,uc,vs,vc

def split_animation(U):
    zetas,zetac,us,uc,vs,vc=np.array_split(U,6)
    return zetas,zetac,us,uc,vs,vc,C,h

def MaxNormOfU(U):
     ''' The max 2 norm of U=(u,v)^T w'''
     zetas,zetac,us,uc,vs,vc=split(U)
     return np.max([
         np.linalg.norm(zetas),
         np.linalg.norm(zetac),
         np.linalg.norm(us),
         np.linalg.norm(uc),
         np.linalg.norm(vs),
         np.linalg.norm(vc)
         ])
 


def F(U):
    zetas,zetac,us,uc,vs,vc=split(U)
    
    ans=np.concatenate((func.Fzetas(zetas,zetac,us,uc,vs,vc,P.ICC0,P.ICh0),
                        func.Fzetac(zetas,zetac,us,uc,vs,vc,P.ICC0,P.ICh0),
                           func.Fus(zetas,zetac,us,uc,vs,vc,P.ICC0,P.ICh0),
                           func.Fuc(zetas,zetac,us,uc,vs,vc,P.ICC0,P.ICh0),
                           func.Fvs(zetas,zetac,us,uc,vs,vc,P.ICC0,P.ICh0),
                           func.Fvc(zetas,zetac,us,uc,vs,vc,P.ICC0,P.ICh0)
                           ))

    return ans




I=func.I

def NumericalJacobian(U):
    zetas,zetac,us,uc,vs,vc=split(U)
    print('\n \t Numerical Jacobian Inner loop')




    J11=np.zeros(I.shape);J12=np.zeros(I.shape);J13=np.zeros(I.shape);J14=np.zeros(I.shape);J15=np.zeros(I.shape);J16=np.zeros(I.shape);
    J21=np.zeros(I.shape);J22=np.zeros(I.shape);J23=np.zeros(I.shape);J24=np.zeros(I.shape);J25=np.zeros(I.shape);J26=np.zeros(I.shape);
    J31=np.zeros(I.shape);J32=np.zeros(I.shape);J33=np.zeros(I.shape);J34=np.zeros(I.shape);J35=np.zeros(I.shape);J36=np.zeros(I.shape);
    J41=np.zeros(I.shape);J42=np.zeros(I.shape);J43=np.zeros(I.shape);J44=np.zeros(I.shape);J45=np.zeros(I.shape);J46=np.zeros(I.shape);
    J51=np.zeros(I.shape);J52=np.zeros(I.shape);J53=np.zeros(I.shape);J54=np.zeros(I.shape);J55=np.zeros(I.shape);J56=np.zeros(I.shape);
    J61=np.zeros(I.shape);J62=np.zeros(I.shape);J63=np.zeros(I.shape);J64=np.zeros(I.shape);J65=np.zeros(I.shape);J66=np.zeros(I.shape);

    
    for i in tqdm(range(0,(P.Nx+1)*(P.Ny+1))):
        h_small=1e-12*I.toarray()[:,i]
        
        for NJ_func in {func.Fzetas,func.Fzetac,func.Fus,func.Fuc,func.Fvs,func.Fvc}:
            if NJ_func == func.Fzetas:
                J1=J11;     J2=J12;     J3=J13;     J4=J14;     J5=J15;     J6=J16;
            if NJ_func == func.Fzetac:
                J1=J21;     J2=J22;     J3=J23;     J4=J24;     J5=J25;     J6=J26;
            if NJ_func == func.Fus:
                J1=J31;     J2=J32;     J3=J33;     J4=J34;     J5=J35;     J6=J36; 
            if NJ_func == func.Fuc:
                J1=J41;     J2=J42;     J3=J43;     J4=J44;     J5=J45;     J6=J46;  
            if NJ_func == func.Fvs:
                J1=J51;     J2=J52;     J3=J53;     J4=J54;     J5=J55;     J6=J56;
            if NJ_func == func.Fvc:
                J1=J61;     J2=J62;     J3=J63;     J4=J64;     J5=J65;     J6=J66; 

                

            J1[:,i] = (NJ_func(zetas+h_small,zetac,us,uc,vs,vc,C,h)-NJ_func(zetas-h_small,zetac,us,uc,vs,vc,C,h))/(2*np.linalg.norm(h_small))
            J2[:,i] = (NJ_func(zetas,zetac+h_small,us,uc,vs,vc,C,h)-NJ_func(zetas,zetac-h_small,us,uc,vs,vc,C,h))/(2*np.linalg.norm(h_small))
            J3[:,i] = (NJ_func(zetas,zetac,us+h_small,uc,vs,vc,C,h)-NJ_func(zetas,zetac,us-h_small,uc,vs,vc,C,h))/(2*np.linalg.norm(h_small))
            J4[:,i] = (NJ_func(zetas,zetac,us,uc+h_small,vs,vc,C,h)-NJ_func(zetas,zetac,us,uc-h_small,vs,vc,C,h))/(2*np.linalg.norm(h_small))
            J5[:,i] = (NJ_func(zetas,zetac,us,uc,vs+h_small,vc,C,h)-NJ_func(zetas,zetac,us,uc,vs-h_small,vc,C,h))/(2*np.linalg.norm(h_small))
            J6[:,i] = (NJ_func(zetas,zetac,us,uc,vs,vc+h_small,C,h)-NJ_func(zetas,zetac,us,uc,vs,vc-h_small,C,h))/(2*np.linalg.norm(h_small))

         
    J=sp.bmat([
                [sp.csr_matrix(J11), sp.csr_matrix(J12), sp.csr_matrix(J13), sp.csr_matrix(J14), sp.csr_matrix(J15), sp.csr_matrix(J16)], 
                [sp.csr_matrix(J21), sp.csr_matrix(J22), sp.csr_matrix(J23), sp.csr_matrix(J24), sp.csr_matrix(J25), sp.csr_matrix(J26)],
                [sp.csr_matrix(J31), sp.csr_matrix(J32), sp.csr_matrix(J33), sp.csr_matrix(J34), sp.csr_matrix(J35), sp.csr_matrix(J36)], 
                [sp.csr_matrix(J41), sp.csr_matrix(J42), sp.csr_matrix(J43), sp.csr_matrix(J44), sp.csr_matrix(J45), sp.csr_matrix(J46)],
                [sp.csr_matrix(J51), sp.csr_matrix(J52), sp.csr_matrix(J53), sp.csr_matrix(J54), sp.csr_matrix(J55), sp.csr_matrix(J56)],
                [sp.csr_matrix(J61), sp.csr_matrix(J62), sp.csr_matrix(J63), sp.csr_matrix(J64), sp.csr_matrix(J65), sp.csr_matrix(J66)]
                ],format='csr')
    return J



 


def plotjacobian(NJ,BOOL=False):
    plt.figure();
    plt.title('Numericla Jacobian')
    for j in range(1,6):
        if BOOL:
            for x in np.where(P.NorthBoundary == 1)[0]: plt.axvline((j-1)*(P.Nx+1)*(P.Ny+1)+x,color='#4dff4d',linewidth=2)
            for y in np.where(P.NorthBoundary == 1)[0]: plt.axhline((j-1)*(P.Nx+1)*(P.Ny+1)+y,color='#4dff4d',linewidth=2)
            for x in np.where(P.EastBoundary == 1)[0]: plt.axvline((j-1)*(P.Nx+1)*(P.Ny+1)+x,color='#ff4d4d',linewidth=2)
            for y in np.where(P.EastBoundary == 1)[0]: plt.axhline((j-1)*(P.Nx+1)*(P.Ny+1)+y,color='#ff4d4d',linewidth=2)
            for x in np.where(P.SouthBoundary == 1)[0]: plt.axvline((j-1)*(P.Nx+1)*(P.Ny+1)+x,color='#ff4da6',linewidth=2)
            for y in np.where(P.SouthBoundary == 1)[0]: plt.axhline((j-1)*(P.Nx+1)*(P.Ny+1)+y,color='#ff4da6',linewidth=2)
            for x in np.where(P.WestBoundary == 1)[0]: plt.axvline((j-1)*(P.Nx+1)*(P.Ny+1)+x,color='#4dffff',linewidth=2)
            for y in np.where(P.WestBoundary == 1)[0]: plt.axhline((j-1)*(P.Nx+1)*(P.Ny+1)+y,color='#4dffff',linewidth=2)
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color='#4dff4d', lw=2, label='Noth Bounary'),
                               Line2D([0], [0], color='#ff4d4d', lw=2, label='East Bounary'),
                               Line2D([0], [0], color='#ff4da6', lw=2, label='South Bounary'),
                               Line2D([0], [0], color='#4dffff', lw=2, label='West Bounary')]
            plt.legend(handles=legend_elements,loc='upper left')
        plt.axvline(x=j*(P.Nx+1)*(P.Ny+1),color='k')
        plt.axhline(y=j*(P.Nx+1)*(P.Ny+1),color='k')
    
    plt.spy(NJ)


