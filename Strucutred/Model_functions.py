# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:16:14 2021

@author: billy Verhage 
"""

if True:

    import numpy as np
    import scipy.sparse as sp

    import Model_parameters as P
    
    

if True:
    
   def FDLaplacian2D(Nx,Ny,dx,dy):

        diagonalsx = [-1*np.ones((1,Nx+2)),np.ones((1,Nx+1))]
        diagonalsy = [-1*np.ones((1,Ny+2)),np.ones((1,Ny+1))]
        
        Dx =1/(dx)*sp.diags(diagonalsx, [-1, 0], shape=(Nx+2,Nx+1))
        Dy =1/(dy)*sp.diags(diagonalsy, [-1, 0], shape=(Ny+2,Ny+1))
        
        LXX=Dx.transpose().dot(Dx)
        LYY=Dy.transpose().dot(Dy)
   

        Ans = sp.kron(LYY,sp.eye(Nx+1))+sp.kron(sp.eye(Ny+1),LXX)
        return Ans

   def LX(Nx,Ny,dx,dy):
        
        diagonalsx = [-1*np.ones((1,Nx)),np.ones((1,Nx))]


        Dx =1/(2*dx)*sp.diags(diagonalsx, [-1 ,1], shape=(Nx+1,Nx+1))
        
        Ans = sp.kron(sp.eye(Ny+1),Dx)
        
        return Ans

       
   def LY(Nx,Ny,dx,dy):
        
        diagonalsy = [-1*np.ones((1,Ny)),np.ones((1,Ny))]


        Dy =1/(2*dy)*sp.diags(diagonalsy, [-1, 1], shape=(Ny+1,Ny+1))

        Ans = sp.kron(Dy,sp.eye(Nx+1))

        return Ans

        
        
 
# 2D FD Laplacian and identity matrices

'''  Used matrix identies '''

LxD  =  LX(P.Nx,P.Ny,P.dx,P.dy)

LyD  =  LY(P.Nx,P.Ny,P.dx,P.dy)

A_x  =  FDLaplacian2D(P.Nx,P.Ny,P.dx,P.dy)


I = sp.eye((P.Ny+1)*(P.Nx+1))

I_xoffR = sp.diags(np.ones((P.Ny+1)*(P.Nx+1)-1), offsets=1)
I_xoffL = sp.diags(np.ones((P.Ny+1)*(P.Nx+1)-1), offsets=-1)
I_yoffR = sp.diags(np.ones((P.Ny+1)*(P.Nx+1)-(P.Nx+1)), offsets=(P.Nx+1))
I_yoffL = sp.diags(np.ones((P.Ny+1)*(P.Nx+1)-(P.Nx+1)), offsets=-(P.Nx+1))

''' Initial condition '''




 
def cornerfix(NP_array):
    ans= NP_array*(-P.NWCorner -P.NECorner -P.SWCorner -P.SECorner)
    return ans
    
def beta(h):
    return 1/(1-np.exp(-P.lambda_d*(1-h)))


def Fzetas(zetas,zetac,us,uc,vs,vc,C,h):
    ''' interior '''
    
    fzetas =-zetas-(1-h)*(LxD*uc+LyD*vc)+uc*(LxD*h)+vc*(LyD*h)
    
    fzetas = P.Interior*fzetas
    
    
    
    ''' west boundary '''
    'x=0' 
    fzetas += P.WestBoundary*(
                            -zetas + 0
                            )
    
    ''' east boundary '''
    'x=1' 

    fzetas += P.EastBoundary*(
                        -zetas + P.Atilde*np.sin(P.phi)
                        )

    ''' South Boundary '''
    'y=0' 
    fzetas += P.SouthBoundary*1/P.dy*(-zetas+I_yoffR*zetas)
    
    ''' North Boundary ''' 
    'y=1'
    fzetas += P.NorthBoundary*1/P.dy*(-I_yoffL*zetas+zetas)


    ''' quick corner fix'''
    
    fzetas += P.NWCorner*(-zetas+0)+P.SWCorner*(-zetas+0)+P.NECorner*(-zetas + P.Atilde*np.sin(P.phi))+P.SECorner*(-zetas + P.Atilde*np.sin(P.phi))
    #fzetas += cornerfix(zetas)
    return fzetas

def Fzetac(zetas,zetac,us,uc,vs,vc,C,h):
    ''' interior '''
    fzetac =-zetac+(1-h)*(LxD*us+LyD*vs)-us*(LxD*h)-vs*(LyD*h)
    
    fzetac = P.Interior*fzetac
    
    ''' west boundary '''
    'x=0' 
    fzetac += P.WestBoundary*(
                            -zetac + 1
  
                            )
    ''' east boundary '''
    'x=1' 

    fzetac += P.EastBoundary*(
                                -zetac + P.Atilde*np.cos(P.phi)
                            )
    ''' South Boundary '''
    'y=0' 
    fzetac += P.SouthBoundary*(1/P.dy*(-zetac+I_yoffR*zetac))
    
    ''' North Boundary ''' 
    'y=1' 
    fzetac += P.NorthBoundary*1/P.dy*(-I_yoffL*zetac +zetac)

    ''' quick corner fix'''
    
    fzetac += P.NWCorner*(-zetac+1)+P.SWCorner*(-zetac+1)+P.NECorner*(-zetac + P.Atilde*np.cos(P.phi))+P.SECorner*(-zetac + P.Atilde*np.cos(P.phi))
    #fzetac += cornerfix(zetac)
    return fzetac


def Fus(zetas,zetac,us,uc,vs,vc,C,h):
    ''' interior '''
    fus   = -us +P.fhat*vc-np.divide(P.r, 1-h)*uc-P.lambda_L**(2)*LxD*zetas
    
    fus = P.Interior*fus
    
    ''' west boundary '''
    'x=0' 
    ' KNOWN : zetas = 0 zetac = A  '
    ' UNKOWN : us, uc '
    fus   +=  P.WestBoundary*(
                            -us
                            +P.fhat*vc
                            -np.divide(P.r, 1-h)*uc
                            -P.lambda_L**(2)*1/(P.dx)*(-0 +I_xoffR*zetas)
                            )
    ''' east boundary ''' 
    'x=1' 

    fus += P.EastBoundary*(
                             -us
                             +P.fhat*vc
                             -np.divide(P.r, 1-h)*uc
                             -P.lambda_L**(2)*1/P.dx*(-I_xoffL*zetas +P.Atilde*np.sin(P.phi)) 
                             )

         
    ''' South Boundary '''
    'y=0' 
    fus += P.SouthBoundary*(-us+P.fhat*vc-np.divide(P.r, 1-h)*uc-P.lambda_L**(2)*LxD*zetas)
    
    ''' North Boundary ''' 
    'y=1' 
    fus += P.NorthBoundary*(-us+P.fhat*vc-np.divide(P.r, 1-h)*uc-P.lambda_L**(2)*LxD*zetas)
    
    
    # ''' quick corner fix'''
    
    fus += P.NWCorner*( -us
                      -np.divide(P.r, 1-h)*uc
                      -P.lambda_L**(2)*1/P.dx*(-0 +I_xoffR*zetas)
                      )
    fus += P.SWCorner*( -us
                      -np.divide(P.r, 1-h)*uc
                      -P.lambda_L**(2)*1/P.dx*(-0 +I_xoffR*zetas))
    
    fus += P.NECorner*( -us
                      -np.divide(P.r, 1-h)*uc
                      -P.lambda_L**(2)*1/P.dx*(-I_xoffL*zetas +P.Atilde*np.sin(P.phi))
                      )
    fus += P.SECorner*( -us
                      -np.divide(P.r, 1-h)*uc
                      -P.lambda_L**(2)*1/P.dx*(-I_xoffL*zetas +P.Atilde*np.sin(P.phi))
                      )
    #fus += cornerfix(us)
    return fus
         
def Fuc(zetas,zetac,us,uc,vs,vc,C,h):
    ''' interior '''
    fuc   = -uc -P.fhat*vs +np.divide(P.r, 1-h)*us +P.lambda_L**(2)*LxD*zetac
    
    fuc = P.Interior*fuc 
    
    ''' west boundary '''
    'x=0' 
    ' KNOWN : zetas = 0 zetac = A  '
    ' UNKOWN : us, uc '
    fuc     +=  P.WestBoundary*(
                                -uc
                                -P.fhat*vs
                                +np.divide(P.r, 1-h)*us
                                
                                +P.lambda_L**(2)*1/P.dx*( -1 +I_xoffR*zetac)
                                )
                            
    ''' east boundary '''
    'x=1' 

    fuc += P.EastBoundary*(
                         -uc
                         -P.fhat*vs
                         +np.divide(P.r, 1-h)*us
                         +P.lambda_L**(2)*1/P.dx*( -I_xoffL*zetac  + P.Atilde*np.cos(P.phi))
                        )

    ''' South Boundary '''
    'y=0' 
    fuc += P.SouthBoundary*(-uc-P.fhat*vs+np.divide(P.r, 1-h)*us+P.lambda_L**(2)*LxD*zetac)
    
    ''' North Boundary ''' 
    'y=1' 
    fuc += P.NorthBoundary*(-uc-P.fhat*vs+np.divide(P.r, 1-h)*us+P.lambda_L**(2)*LxD*zetac)
    
    # ''' quick corner fix'''
    
    fuc += P.NWCorner*( -uc 
                      +np.divide(P.r, 1-h)*us
                      +P.lambda_L**(2)*1/P.dx*(-1 + I_xoffR*zetac))
    
    fuc += P.SWCorner*( -uc
                        +np.divide(P.r, 1-h)*us
                        +P.lambda_L**(2)*1/P.dx*(-1 +I_xoffR*zetac))
    
    fuc += P.NECorner*(-uc
                      +np.divide(P.r, 1-h)*us
                      +P.lambda_L**(2)*1/P.dx*(-I_xoffL*zetac  + P.Atilde*np.cos(P.phi)))
    fuc += P.SECorner*(-uc
                      +np.divide(P.r, 1-h)*us
                      +P.lambda_L**(2)*1/P.dx*(-I_xoffL*zetac  + P.Atilde*np.cos(P.phi)))
    #fuc += cornerfix(uc)
    return fuc

def Fvs(zetas,zetac,us,uc,vs,vc,C,h):
    ''' interior '''
    fvs   = -vs -P.fhat*uc -np.divide(P.r, 1-h)*vc -P.lambda_L**(2)*LyD*zetas
    
    fvs = P.Interior*fvs
    
    ''' west boundary '''
    ' KNOWN : zetas = 0 zetac = A  '
    ' UNKOWN : us, uc '
    fvs   +=  P.WestBoundary*(-vs-P.fhat*uc -np.divide(P.r, 1-h)*vc -P.lambda_L**(2)*LyD*zetas)
                            
    #                         )
    # ''' east boundary ''' 

    fvs += P.EastBoundary*(-vs-P.fhat*uc -np.divide(P.r, 1-h)*vc -P.lambda_L**(2)*LyD*zetas)
    #                          )
         
    ''' South Boundary '''
    fvs += P.SouthBoundary*(-vs+0)
    
    ''' North Boundary ''' 
    fvs += P.NorthBoundary*(-vs+0)
    ''' quick corner fix'''
    
    fvs += P.NWCorner*(-vs-P.fhat*uc -np.divide(P.r, 1-h)*vc -P.lambda_L**(2)*1/P.dy*(-I_yoffL*zetas+zetas))
    
    fvs += P.SWCorner*(-vs-P.fhat*uc -np.divide(P.r, 1-h)*vc -P.lambda_L**(2)*1/P.dy*(-zetas+I_yoffR*zetas))
    
    fvs += P.NECorner*(-vs-P.fhat*uc -np.divide(P.r, 1-h)*vc -P.lambda_L**(2)*1/P.dy*(-I_yoffL*zetas+zetas))
    
    fvs += P.SECorner*(-vs-P.fhat*uc -np.divide(P.r, 1-h)*vc -P.lambda_L**(2)*1/P.dy*(-zetas+I_yoffR*zetas))
    return fvs

def Fvc(zetas,zetac,us,uc,vs,vc,C,h):
    ''' interior '''
    fvc   = -vc +P.fhat*us   +np.divide(P.r, 1-h)*vs +P.lambda_L**(2)*LyD*zetac
    
    fvc = P.Interior*fvc
    
    ''' west boundary '''
    ' KNOWN : zetas = 0 zetac = A  '
    ' UNKOWN : us, uc '
    fvc   +=  P.WestBoundary*(-vc+P.fhat*us+np.divide(P.r,1-h)*vs+P.lambda_L**(2)*LyD*zetac)
                            
    #                         )
    # ''' east boundary ''' 

    fvc += P.EastBoundary*(-vc+P.fhat*us+np.divide(P.r,1-h)*vs+P.lambda_L**(2)*LyD*zetac)
    #                          )
         
    # ''' South Boundary '''
    fvc += P.SouthBoundary*(-vc+0)
    
    ''' North Boundary ''' 
    fvc += P.NorthBoundary*(-vc+0)
    
    # ''' quick corner fix'''
    
    fvc += P.NWCorner*(-vc+P.fhat*us+np.divide(P.r,1-h)*vs+P.lambda_L**(2)*1/P.dy*(-I_yoffL*zetac+zetac))
    
    fvc += P.SWCorner*(-vc+P.fhat*us+np.divide(P.r,1-h)*vs+P.lambda_L**(2)*1/P.dy*(-zetac+I_yoffR*zetac))
    
    fvc += P.NECorner*(-vc+P.fhat*us+np.divide(P.r,1-h)*vs+P.lambda_L**(2)*1/P.dy*(-I_yoffL*zetac+zetac))
    
    fvc += P.SECorner*(-vc+P.fhat*us+np.divide(P.r,1-h)*vs+P.lambda_L**(2)*1/P.dy*(-zetac+I_yoffR*zetac))
    return fvc

def FC(zetas,zetac,us,uc,vs,vc,C,h):
    ''' interior '''
    fC  = +P.epsilon*P.a*P.k*A_x*C
    fC += P.epsilon*P.a*P.k*P.lambda_d*(beta(h)*LxD*h*LxD*C+ C*( beta(h)*A_x*h + LxD*beta(h)*LxD*h ))
    fC += -P.epsilon*beta(h)*C+us*us+vs*vs+uc*uc+vc*vc
    
    fC = P.Interior*fC
    
    ''' west boundary '''
    ' KNOWN : zetas = 0 zetac = A  '
    ' UNKOWN : us, uc '
    fC   +=  P.WestBoundary*(-P.epsilon*beta(h)*C+us*us+vs*vs+uc*uc+vc*vc)
                            
    #                         )
    # ''' east boundary ''' 

    fC   += P.EastBoundary*(-P.epsilon*beta(h)*C+us*us+vs*vs+uc*uc+vc*vc)
    #                          )
         
    ''' South Boundary '''
    fC += P.SouthBoundary*(1/P.dy*(-C+I_yoffR*C))
    
    ''' North Boundary ''' 
    fC += P.NorthBoundary*(1/P.dy*(-I_yoffL*C+C))
    
    # ''' quick corner fix'''
    
    fC += P.NWCorner*(-P.epsilon*beta(h)*C+us*us+vs*vs+uc*uc+vc*vc)#1/P.dy*(-I_yoffL*C+C))
    
    fC += P.SWCorner*(-P.epsilon*beta(h)*C+us*us+vs*vs+uc*uc+vc*vc)#1/P.dy*(-C+I_yoffR*C))
    
    fC += P.NECorner*(-P.epsilon*beta(h)*C+us*us+vs*vs+uc*uc+vc*vc)#1/P.dy*(-I_yoffL*C+C))
    
    fC += P.SECorner*(-P.epsilon*beta(h)*C+us*us+vs*vs+uc*uc+vc*vc)#1/P.dy*(-C+I_yoffR*C))
    return fC

def Fh(zetas,zetac,us,uc,vs,vc,C,h):
    ''' interior '''
    fh  = +P.Mutilde*A_x*h
    fh += -P.delta_s*(-P.epsilon*beta(h)*C+us*us+vs*vs+uc*uc+vc*vc)
    #fh += delta_s*(epsilon*a*k*A_x*C+epsilon*a*k*lambda_d*( beta(h)*LxD*h*LxD*C+ C*(beta(h)*A_x*h+LxD*beta(h)*LxD*h) ))
 #
    
    fh = P.Interior*fh
    
    ''' west boundary '''
    ' KNOWN : zetas = 0 zetac = A  '
    ' UNKOWN : us, uc '
    fh   +=  P.WestBoundary*(-h+0)
                            
    #                         )
    # ''' east boundary ''' 

    fh += P.EastBoundary*(-h+1-P.H2/P.H1)
    #                          )
         
    # ''' South Boundary '''
    fh += P.SouthBoundary*(1/P.dy*(-h+I_yoffR*h))
    
    ''' North Boundary ''' 
    fh += P.NorthBoundary*(1/P.dy*(-I_yoffL*h+h))
    
    # ''' quick corner fix'''
    
    fh += P.NWCorner*(-h+0)
    
    fh += P.SWCorner*(-h+0)
    
    fh += P.NECorner*(-h+1-P.H2/P.H1)
    
    fh += P.SECorner*(-h+1-P.H2/P.H1)
    return fh

