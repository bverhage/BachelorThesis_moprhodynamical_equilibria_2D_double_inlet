# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:16:14 2021

@author: billy Verhage 
"""
import sys
sys.path.append("..") 

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

import Model_functions as func
import Model_parameters as P

from tqdm import tqdm 

def split(U):
    zetas,zetac,us,uc,vs,vc,C,h=np.array_split(U,8)
    return zetas,zetac,us,uc,vs,vc,C,h

def split_animation(U):
    zetas,zetac,us,uc,vs,vc,C,h=np.array_split(U,8)
    return zetas,zetac,us,uc,vs,vc,C,h

def MaxNormOfU(U):
     ''' The max 2 norm of U=(u,v)^T w'''
     zetas,zetac,us,uc,vs,vc,C,h=np.array_split(U,8)
     return np.max([
         np.linalg.norm(zetas),
         np.linalg.norm(zetac),
         np.linalg.norm(us),
         np.linalg.norm(uc),
         np.linalg.norm(vs),
         np.linalg.norm(vc),
         np.linalg.norm(C),
         np.linalg.norm(h)
         ])
 


def F(U, phi:float=P.phi, Ar:float=P.Atilde, H2:float=P.H2, B:float=P.Ly):
    zetas,zetac,us,uc,vs,vc,C,h=np.array_split(U,8)
    
    ans=np.concatenate((func.Fzetas(zetas,zetac,us,uc,vs,vc,C,h,phi,Ar,H2,B),
                        func.Fzetac(zetas,zetac,us,uc,vs,vc,C,h,phi,Ar,H2,B),
                           func.Fus(zetas,zetac,us,uc,vs,vc,C,h,phi,Ar,H2,B),
                           func.Fuc(zetas,zetac,us,uc,vs,vc,C,h,phi,Ar,H2,B),
                           func.Fvs(zetas,zetac,us,uc,vs,vc,C,h,phi,Ar,H2,B),
                           func.Fvc(zetas,zetac,us,uc,vs,vc,C,h,phi,Ar,H2,B),
                            func.FC(zetas,zetac,us,uc,vs,vc,C,h,phi,Ar,H2,B),
                            func.Fh(zetas,zetac,us,uc,vs,vc,C,h,phi,Ar,H2,B)
                           ))

    return ans

I=func.I

def NumericalJacobian(U, phi:float=P.phi, Ar:float=P.Atilde, H2:float=P.H2, B:float=P.Ly):
    zetas,zetac,us,uc,vs,vc,C,h=np.array_split(U,8)
    print('\n \t Numerical Jacobian Inner loop')
    # J11=sp.csr_matrix(I.shape);J12=sp.csr_matrix(I.shape);J13=sp.csr_matrix(I.shape);J14=sp.csr_matrix(I.shape);J15=sp.csr_matrix(I.shape);J16=sp.csr_matrix(I.shape);
    # J21=sp.csr_matrix(I.shape);J22=sp.csr_matrix(I.shape);J23=sp.csr_matrix(I.shape);J24=sp.csr_matrix(I.shape);J25=sp.csr_matrix(I.shape);J26=sp.csr_matrix(I.shape);
    # J31=sp.csr_matrix(I.shape);J32=sp.csr_matrix(I.shape);J33=sp.csr_matrix(I.shape);J34=sp.csr_matrix(I.shape);J35=sp.csr_matrix(I.shape);J36=sp.csr_matrix(I.shape);
    # J41=sp.csr_matrix(I.shape);J42=sp.csr_matrix(I.shape);J43=sp.csr_matrix(I.shape);J44=sp.csr_matrix(I.shape);J45=sp.csr_matrix(I.shape);J46=sp.csr_matrix(I.shape);
    # J51=sp.csr_matrix(I.shape);J52=sp.csr_matrix(I.shape);J53=sp.csr_matrix(I.shape);J54=sp.csr_matrix(I.shape);J55=sp.csr_matrix(I.shape);J56=sp.csr_matrix(I.shape);
    # J61=sp.csr_matrix(I.shape);J62=sp.csr_matrix(I.shape);J63=sp.csr_matrix(I.shape);J64=sp.csr_matrix(I.shape);J65=sp.csr_matrix(I.shape);J66=sp.csr_matrix(I.shape);

    
    # J11,J12,J13,J14,J15,J16,J17,J18 = ANALYTICAL.AnalyticalJacobian_zetas(U)
    # J21,J22,J23,J24,J25,J26,J27,J28 = ANALYTICAL.AnalyticalJacobian_zetac(U)
    # J31,J32,J33,J34,J35,J36,J37,J38 = ANALYTICAL.AnalyticalJacobian_us(U)
    # J41,J42,J43,J44,J45,J46,J47,J48 = ANALYTICAL.AnalyticalJacobian_uc(U)
    # J51,J52,J53,J54,J55,J56,J57,J58 = ANALYTICAL.AnalyticalJacobian_vs(U)
    # J61,J62,J63,J64,J65,J66,J67,J68 = ANALYTICAL.AnalyticalJacobian_vc(U)
    J11=np.zeros(I.shape);J12=np.zeros(I.shape);J13=np.zeros(I.shape);J14=np.zeros(I.shape);J15=np.zeros(I.shape);J16=np.zeros(I.shape);J17=np.zeros(I.shape);J18=np.zeros(I.shape);
    J21=np.zeros(I.shape);J22=np.zeros(I.shape);J23=np.zeros(I.shape);J24=np.zeros(I.shape);J25=np.zeros(I.shape);J26=np.zeros(I.shape);J27=np.zeros(I.shape);J28=np.zeros(I.shape);
    J31=np.zeros(I.shape);J32=np.zeros(I.shape);J33=np.zeros(I.shape);J34=np.zeros(I.shape);J35=np.zeros(I.shape);J36=np.zeros(I.shape);J37=np.zeros(I.shape);J38=np.zeros(I.shape);
    J41=np.zeros(I.shape);J42=np.zeros(I.shape);J43=np.zeros(I.shape);J44=np.zeros(I.shape);J45=np.zeros(I.shape);J46=np.zeros(I.shape);J47=np.zeros(I.shape);J48=np.zeros(I.shape);
    J51=np.zeros(I.shape);J52=np.zeros(I.shape);J53=np.zeros(I.shape);J54=np.zeros(I.shape);J55=np.zeros(I.shape);J56=np.zeros(I.shape);J57=np.zeros(I.shape);J58=np.zeros(I.shape);
    J61=np.zeros(I.shape);J62=np.zeros(I.shape);J63=np.zeros(I.shape);J64=np.zeros(I.shape);J65=np.zeros(I.shape);J66=np.zeros(I.shape);J67=np.zeros(I.shape);J68=np.zeros(I.shape);
    J71=np.zeros(I.shape);J72=np.zeros(I.shape);J73=np.zeros(I.shape);J74=np.zeros(I.shape);J75=np.zeros(I.shape);J76=np.zeros(I.shape);J77=np.zeros(I.shape);J78=np.zeros(I.shape);
    J81=np.zeros(I.shape);J82=np.zeros(I.shape);J83=np.zeros(I.shape);J84=np.zeros(I.shape);J85=np.zeros(I.shape);J86=np.zeros(I.shape);J87=np.zeros(I.shape);J88=np.zeros(I.shape);
    
    for i in tqdm(range(0,(P.Nx+1)*(P.Ny+1))):
        h_small=1e-12*I.toarray()[:,i]
        
        for NJ_func in {func.Fzetas,func.Fzetac,func.Fus,func.Fuc,func.Fvs,func.Fvc,func.FC,func.Fh}:
            if NJ_func == func.Fzetas:
                #zetas       zetac       us           uc           vs           vc           C             h
                J1=J11;      J2=J12;     J3=J13;      J4=J14;      J5=J15;      J6=J16;      J7=J17;      J8=J18;
                BOOL_1=True;BOOL_2=True;BOOL_3=True;BOOL_4=True;BOOL_5=True;BOOL_6=True;BOOL_7=True;BOOL_8=True;
            if NJ_func == func.Fzetac:
                #zetas      zetac       us          uc          vs           vc          C           h
                J1=J21;     J2=J22;     J3=J23;     J4=J24;     J5=J25;     J6=J26;     J7=J27;     J8=J28;
                BOOL_1=True;BOOL_2=True;BOOL_3=True;BOOL_4=True;BOOL_5=True;BOOL_6=True;BOOL_7=True;BOOL_8=True;
            if NJ_func == func.Fus:
                #zetas      zetac       us          uc          vs           vc          C           h
                J1=J31;     J2=J32;     J3=J33;     J4=J34;     J5=J35;     J6=J36;     J7=J37;     J8=J38;
                BOOL_1=True;BOOL_2=True;BOOL_3=True;BOOL_4=True;BOOL_5=True;BOOL_6=True;BOOL_7=True;BOOL_8=True;
            if NJ_func == func.Fuc:
                #zetas      zetac       us          uc          vs           vc          C           h
                J1=J41;     J2=J42;     J3=J43;     J4=J44;     J5=J45;     J6=J46;     J7=J47;     J8=J48;
                BOOL_1=True;BOOL_2=True;BOOL_3=True;BOOL_4=True;BOOL_5=True;BOOL_6=True;BOOL_7=True;BOOL_8=True;
            if NJ_func == func.Fvs:
                #zetas      zetac       us          uc          vs           vc          C           h
                J1=J51;     J2=J52;     J3=J53;     J4=J54;     J5=J55;     J6=J56;     J7=J57;     J8=J58;
                BOOL_1=True;BOOL_2=True;BOOL_3=True;BOOL_4=True;BOOL_5=True;BOOL_6=True;BOOL_7=True;BOOL_8=True;
            if NJ_func == func.Fvc:
                #zetas      zetac       us          uc          vs           vc          C           h
                J1=J61;     J2=J62;     J3=J63;     J4=J64;     J5=J65;     J6=J66;     J7=J67;     J8=J68;
                BOOL_1=True;BOOL_2=True;BOOL_3=True;BOOL_4=True;BOOL_5=True;BOOL_6=True;BOOL_7=True;BOOL_8=True;
            if NJ_func == func.FC:
                #zetas      zetac       us          uc          vs           vc          C           h
                J1=J71;     J2=J72;     J3=J63;     J4=J74;     J5=J75;     J6=J76;     J7=J77;     J8=J78;
                BOOL_1=True;BOOL_2=True;BOOL_3=True;BOOL_4=True;BOOL_5=True;BOOL_6=True;BOOL_7=True;BOOL_8=True;               
            if NJ_func == func.Fh:
                #zetas      zetac       us          uc          vs           vc          C           h
                J1=J81;     J2=J82;     J3=J83;     J4=J84;     J5=J85;     J6=J86;     J7=J87;     J8=J88;
                BOOL_1=True;BOOL_2=True;BOOL_3=True;BOOL_4=True;BOOL_5=True;BOOL_6=True;BOOL_7=True;BOOL_8=True;
                
            # J1[:,i] = np.array([(NJ_func(zetas+h_small,zetac,us,uc,vs,vc)-NJ_func(zetas-h_small,zetac,us,uc,vs,vc))/(2*np.linalg.norm(h_small))]).T
            # J2[:,i] = np.array([(NJ_func(zetas,zetac+h_small,us,uc,vs,vc)-NJ_func(zetas,zetac-h_small,us,uc,vs,vc))/(2*np.linalg.norm(h_small))]).T
            # J3[:,i] = np.array([(NJ_func(zetas,zetac,us+h_small,uc,vs,vc)-NJ_func(zetas,zetac,us-h_small,uc,vs,vc))/(2*np.linalg.norm(h_small))]).T
            # J4[:,i] = np.array([(NJ_func(zetas,zetac,us,uc+h_small,vs,vc)-NJ_func(zetas,zetac,us,uc-h_small,vs,vc))/(2*np.linalg.norm(h_small))]).T
            # J5[:,i] = np.array([(NJ_func(zetas,zetac,us,uc,vs+h_small,vc)-NJ_func(zetas,zetac,us,uc,vs-h_small,vc))/(2*np.linalg.norm(h_small))]).T
            # J6[:,i] = np.array([(NJ_func(zetas,zetac,us,uc,vs,vc+h_small)-NJ_func(zetas,zetac,us,uc,vs,vc-h_small))/(2*np.linalg.norm(h_small))]).T
            if BOOL_1: J1[:,i] = (NJ_func(zetas+h_small,zetac,us,uc,vs,vc,C,h,phi,Ar,H2,B)-NJ_func(zetas-h_small,zetac,us,uc,vs,vc,C,h,phi,Ar,H2,B))/(2*np.linalg.norm(h_small))
            if BOOL_2: J2[:,i] = (NJ_func(zetas,zetac+h_small,us,uc,vs,vc,C,h,phi,Ar,H2,B)-NJ_func(zetas,zetac-h_small,us,uc,vs,vc,C,h,phi,Ar,H2,B))/(2*np.linalg.norm(h_small))
            if BOOL_3: J3[:,i] = (NJ_func(zetas,zetac,us+h_small,uc,vs,vc,C,h,phi,Ar,H2,B)-NJ_func(zetas,zetac,us-h_small,uc,vs,vc,C,h,phi,Ar,H2,B))/(2*np.linalg.norm(h_small))
            if BOOL_4: J4[:,i] = (NJ_func(zetas,zetac,us,uc+h_small,vs,vc,C,h,phi,Ar,H2,B)-NJ_func(zetas,zetac,us,uc-h_small,vs,vc,C,h,phi,Ar,H2,B))/(2*np.linalg.norm(h_small))
            if BOOL_5: J5[:,i] = (NJ_func(zetas,zetac,us,uc,vs+h_small,vc,C,h,phi,Ar,H2,B)-NJ_func(zetas,zetac,us,uc,vs-h_small,vc,C,h,phi,Ar,H2,B))/(2*np.linalg.norm(h_small))
            if BOOL_6: J6[:,i] = (NJ_func(zetas,zetac,us,uc,vs,vc+h_small,C,h,phi,Ar,H2,B)-NJ_func(zetas,zetac,us,uc,vs,vc-h_small,C,h,phi,Ar,H2,B))/(2*np.linalg.norm(h_small))
            if BOOL_7: J7[:,i] = (NJ_func(zetas,zetac,us,uc,vs,vc,C+h_small,h,phi,Ar,H2,B)-NJ_func(zetas,zetac,us,uc,vs,vc,C-h_small,h,phi,Ar,H2,B))/(2*np.linalg.norm(h_small))
            if BOOL_8: J8[:,i] = (NJ_func(zetas,zetac,us,uc,vs,vc,C,h+h_small,phi,Ar,H2,B)-NJ_func(zetas,zetac,us,uc,vs,vc,C,h-h_small,phi,Ar,H2,B))/(2*np.linalg.norm(h_small))
         
    J=sp.bmat([
                [sp.csr_matrix(J11), sp.csr_matrix(J12), sp.csr_matrix(J13), sp.csr_matrix(J14), sp.csr_matrix(J15), sp.csr_matrix(J16), sp.csr_matrix(J17), sp.csr_matrix(J18)],
                [sp.csr_matrix(J21), sp.csr_matrix(J22), sp.csr_matrix(J23), sp.csr_matrix(J24), sp.csr_matrix(J25), sp.csr_matrix(J26), sp.csr_matrix(J27), sp.csr_matrix(J28)],
                [sp.csr_matrix(J31), sp.csr_matrix(J32), sp.csr_matrix(J33), sp.csr_matrix(J34), sp.csr_matrix(J35), sp.csr_matrix(J36), sp.csr_matrix(J37), sp.csr_matrix(J38)],
                [sp.csr_matrix(J41), sp.csr_matrix(J42), sp.csr_matrix(J43), sp.csr_matrix(J44), sp.csr_matrix(J45), sp.csr_matrix(J46), sp.csr_matrix(J47), sp.csr_matrix(J48)],
                [sp.csr_matrix(J51), sp.csr_matrix(J52), sp.csr_matrix(J53), sp.csr_matrix(J54), sp.csr_matrix(J55), sp.csr_matrix(J56), sp.csr_matrix(J57), sp.csr_matrix(J58)],
                [sp.csr_matrix(J61), sp.csr_matrix(J62), sp.csr_matrix(J63), sp.csr_matrix(J64), sp.csr_matrix(J65), sp.csr_matrix(J66), sp.csr_matrix(J67), sp.csr_matrix(J68)],
                [sp.csr_matrix(J71), sp.csr_matrix(J72), sp.csr_matrix(J73), sp.csr_matrix(J74), sp.csr_matrix(J75), sp.csr_matrix(J76), sp.csr_matrix(J77), sp.csr_matrix(J78)],
                [sp.csr_matrix(J81), sp.csr_matrix(J82), sp.csr_matrix(J83), sp.csr_matrix(J84), sp.csr_matrix(J85), sp.csr_matrix(J86), sp.csr_matrix(J87), sp.csr_matrix(J88)]
                ],format='csr')
    return J



def plotjacobian(NJ,BOOL=False):
    plt.figure();
    plt.title('Numericla Jacobian')
    for j in range(1,8):
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


