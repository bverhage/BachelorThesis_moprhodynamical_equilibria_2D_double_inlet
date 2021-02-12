# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:16:14 2021

@author: billy Verhage 
"""

import numpy as np


 
''' Model pramters ''' 

Nx = 30+1                # interior points + 1
Ny = int((Nx-1)*1/3)+1

dx = 1/Nx
dy = 1/Ny
    
    
'''  System Parameters '''

Lx = 59*10**3 # [ m ]

Ly =10**3 # [ m ]

g = 9.81 # [ m/s^2 ]

CD = 0.0025 # [ - ]

sigma = 1.4*10**-4 #[1/s]

fhat = 0#7.1*10**-1 # [ - ]

''' Inlets '''
''' Mars diep inlet''' 

H1 = 12 # [ m ]
A = 0.74 # [ m ]

''' Vlie inlet''' 
H2 = 6 # [ m ]

A2=0.84 # [ m ]

Atilde = A2/A  # [ - ]

phi = 41/180*np.pi # [ - ]


'''  Sediment  Parameters '''
k_h=10**2 # [ m^2/s ]

alpha = 0.5*10**-2 # [ kg s m^-4 ]

omega_s=0.015 # [ m/s ]

k_v = 0.1 # [ m**-2/s ]
'''  bed Parameters '''
rho_s = 2650 # [ kg m^-3 ]
p = 0.4 # [ - ]
mu_hat = 1.4*10**-4 # [ m^2/s ]
''' Model pramters ''' 



U_const = A*Lx*sigma/H1

r_hat=8*U_const*CD/(3*np.pi)


''' Scaled parameters ''' 

epsilon = A/H1

r = r_hat/(H1*sigma)

k = k_h/(Lx**2*sigma)#2.052*10**(-4)

Mutilde = mu_hat/(Lx**2*sigma)

lambda_L = np.sqrt(H1*g)/(Lx*sigma)#sigma*Lx/(np.sqrt(g*H1))

a = k_v*sigma*omega_s**-2


delta_s = alpha*U_const**2/(rho_s*(1-p)*H1*sigma)
    
lambda_d = H1*omega_s/k_v

def printparamters():    
    print('Model paramters \n N_x = %i \t N_y = %i \n ' %(Nx,Ny))   
    print('\n dimensionless paramteters \n')
    print('| epsilon = %.2e  \t | r = %.2e  |  k = %.2e \t | mu = %.2e \t |' %(epsilon, r, k, Mutilde))   
    print('| lambda_L = %.2e \t | a = %.2e  | delta_s = %.2e \t | lambda_d = %.2e \t |' %(lambda_L, a, delta_s, lambda_d))   
    print('\n \n \t Boundary condition \n')
    
    print('x=0 \t | zetas = 0 \t \t  \t  | zetac = 1 \t  \t  \t  | h = 0' )
    print('x=1 \t | zetas = %.2f sin(%.2f pi)\t  | zetac = %.2f cos(%.2f pi)\t  | h = %.2f' %(Atilde,phi/np.pi,Atilde,phi/np.pi,1-H2/H1))



def bedprofile(x,y):

     return (1-H2/H1)*x

def func0(x,y):
    return 0


def westboundary(x,y):
    if x==0 and 0<y<1:
        return 1

    else: return 0
    
def nearwestboundary(x,y):
    if x==dx and 0<y<1:
        return 1

    else: return 0


def eastboundary(x,y):
    if x==1 and 0<y<1: return 1
    else:return 0


def neareastboundary(x,y):
    if x==(Nx-1)*dx and 0<y<1: return 1
    else:return 0

def northboundary(x,y):
    if y==1  and 0<x<1:
        return 1

    else: return 0
    
def nearnorthboundary(x,y):
    if y==(Ny-1)*dy and 0<x<1 :
        return 1

    else: return 0


def southboundary(x,y):
    if y==0 and 0<x<1: return 1
    else:return 0


def nearsouthboundary(x,y):
    if y==dy and 0<x<1: return 1
    else:return 0
    
def NWcornerpoint(x,y):
    if y==1 and x==0: return 1
    else: return 0

def NEcornerpoint(x,y):
    if y==1 and x==1: return 1
    else: return 0
    
def SWcornerpoint(x,y):
    if y==0 and x==0: return 1
    else: return 0

def SEcornerpoint(x,y):
    if y==0 and x==1: return 1
    else: return 0
    
    
def create(Nx:int,Ny:int,func):
    Fvec=np.zeros((Nx+1)*(Ny+1))
    for j in range(Ny+1):
        for i in range(Nx+1):
            x=dx*(i)
            y=dy*(j)
            Fvec[i+j*(Nx+1)]=func(x,y)
    return Fvec

ICzeta0 = create(Nx,Ny,func0)
ICu0    = create(Nx,Ny,func0)
ICv0    = create(Nx,Ny,func0)
ICC0    = create(Nx,Ny,func0)
ICh0    = create(Nx,Ny,bedprofile)


WestBoundary     = create(Nx,Ny,westboundary)
NearWestBoundary = create(Nx,Ny,nearwestboundary)
EastBoundary     = create(Nx,Ny,eastboundary)
NearEastBoundary = create(Nx,Ny,neareastboundary)

NorthBoundary    = create(Nx,Ny,northboundary)
NearNorthBoundary= create(Nx,Ny,nearnorthboundary)
SouthBoundary    = create(Nx,Ny,southboundary)
NearSouthBoundary= create(Nx,Ny,nearsouthboundary)

NWCorner = create(Nx,Ny,NWcornerpoint)
NECorner = create(Nx,Ny,NEcornerpoint)
SWCorner = create(Nx,Ny,SWcornerpoint)
SECorner = create(Nx,Ny,SEcornerpoint)

Interior = np.ones(ICzeta0.shape)-WestBoundary-EastBoundary-NorthBoundary-SouthBoundary-NWCorner-NECorner-SWCorner-SECorner

Encolsure = Interior+WestBoundary+NearWestBoundary+EastBoundary+EastBoundary+NearEastBoundary

    
def reshape(LEXarray):
    return np.reshape(LEXarray,[Ny+1,Nx+1])

