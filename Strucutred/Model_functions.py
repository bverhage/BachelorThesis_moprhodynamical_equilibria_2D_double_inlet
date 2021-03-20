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
    
   def FDLaplacian2D(Nx,Ny,dx,dy,WestNeuman=False,NorthNeuman=False,EastNeuman=False,SouthNeuman=False):
        
        #diagonalsx = [-1*np.ones((1,Nx-1-1)),np.zeros((1,Nx-1)),np.ones((1,Nx-1-1))]
        #diagonalsy = [-1*np.ones((1,Ny-1-1)),np.zeros((1,Ny-1)),np.ones((1,Ny-1-1))]
        diagonalsx = [-1*np.ones((1,Nx+2)),np.ones((1,Nx+1))]
        diagonalsy = [-1*np.ones((1,Ny+2)),np.ones((1,Ny+1))]
        

 
            
        
        Dx =1/(dx)*sp.diags(diagonalsx, [-1, 0], shape=(Nx+2,Nx+1))
        Dy =1/(dy)*sp.diags(diagonalsy, [-1, 0], shape=(Ny+2,Ny+1))
        
        
        LXX=Dx.transpose().dot(Dx)
        if WestNeuman==True:
            LXX[0,0]=-1/dx**2;LXX[0,1]=2/dx**2;LXX[0,2]=-1/dx**2
        if EastNeuman==True:
            LXX[-1,-1]=-1/dx**2;LXX[-1,-2]=2/dx**2;LXX[-1,-3]=-1/dx**2
        LYY=Dy.transpose().dot(Dy)
        if NorthNeuman==True:
            LYY[0,0]=-1/dy**2;LYY[0,1]=2/dy**2;LYY[0,2]=-1/dy**2
        if SouthNeuman==True:
            LYY[-1,-1]=-1/dy**2;LYY[-1,-2]=2/dy**2;LYY[-1,-3]=-1/dy**2
        Ans = -sp.kron(LYY,sp.eye(Nx+1))-sp.kron(sp.eye(Ny+1),LXX)    
        return Ans
    
   def FDLaplacianx(Nx,Ny,dx,dy,WestNeuman=False,NorthNeuman=False,EastNeuman=False,SouthNeuman=False):
        
        #diagonalsx = [-1*np.ones((1,Nx-1-1)),np.zeros((1,Nx-1)),np.ones((1,Nx-1-1))]
        #diagonalsy = [-1*np.ones((1,Ny-1-1)),np.zeros((1,Ny-1)),np.ones((1,Ny-1-1))]
        diagonalsx = [-1*np.ones((1,Nx+2)),np.ones((1,Nx+1))]


 
            
        
        Dx =1/(dx)*sp.diags(diagonalsx, [-1, 0], shape=(Nx+2,Nx+1))
 
        
        
        LXX=Dx.transpose().dot(Dx)
        if WestNeuman==True:
            LXX[0,0]=-1/dx**2;LXX[0,1]=2/dx**2;LXX[0,2]=-1/dx**2
        if EastNeuman==True:
            LXX[-1,-1]=-1/dx**2;LXX[-1,-2]=2/dx**2;LXX[-1,-3]=-1/dx**2

        Ans = -sp.kron(sp.eye(Ny+1),LXX)    
        return Ans
    
   def FDLaplaciany(Nx,Ny,dx,dy,WestNeuman=False,NorthNeuman=False,EastNeuman=False,SouthNeuman=False):
        

        diagonalsy = [-1*np.ones((1,Ny+2)),np.ones((1,Ny+1))]
        

 

        Dy =1/(dy)*sp.diags(diagonalsy, [-1, 0], shape=(Ny+2,Ny+1))
        

        LYY=Dy.transpose().dot(Dy)
        if NorthNeuman==True:
            LYY[0,0]=-1/dy**2;LYY[0,1]=2/dy**2;LYY[0,2]=-1/dy**2
        if SouthNeuman==True:
            LYY[-1,-1]=-1/dy**2;LYY[-1,-2]=2/dy**2;LYY[-1,-3]=-1/dy**2
        Ans = -sp.kron(LYY,sp.eye(Nx+1))  
        return Ans
    
   def FDLaplacian2D_yforward(Nx,Ny,dx,dy):

        diagonalsx = [-1*np.ones((1,Nx+2)),np.ones((1,Nx+1))]
        diagonalsy = [-1*np.ones((1,Ny+1)),2*np.ones((1,Ny)),-1*np.ones((1,Ny-1))]
        
        Dx =1/(dx)*sp.diags(diagonalsx, [-1, 0], shape=(Nx+2,Nx+1))
        LYY =1/(dy**2)*sp.diags(diagonalsy, [0,1,2], shape=(Ny+1,Ny+1))
        
        LXX=Dx.transpose().dot(Dx)

        Ans = sp.kron(LYY,sp.eye(Nx+1))+sp.kron(sp.eye(Ny+1),LXX)
        return Ans
    
   def FDLaplacian2D_ybackward(Nx,Ny,dx,dy):

        diagonalsx = [-1*np.ones((1,Nx+2)),np.ones((1,Nx+1))]
        diagonalsy = [-1*np.ones((1,Ny-1)),2*np.ones((1,Ny)),-1*np.ones((1,Ny+1))]
        
        Dx =1/(dx)*sp.diags(diagonalsx, [-1, 0], shape=(Nx+2,Nx+1))
        LYY =1/(dy**2)*sp.diags(diagonalsy, [-2,-1,0], shape=(Ny+1,Ny+1))
        
        LXX=Dx.transpose().dot(Dx)

        Ans = sp.kron(LYY,sp.eye(Nx+1))+sp.kron(sp.eye(Ny+1),LXX)
        return Ans
    
   def FDLaplacian2D_xforward(Nx,Ny,dx,dy):

        diagonalsx = [-1*np.ones((1,Nx+1)),2*np.ones((1,Nx)),-1*np.ones((1,Nx-1))]
        diagonalsy = [-1*np.ones((1,Ny+2)),np.ones((1,Ny+1))]
        
        LXX =1/(dx**2)*sp.diags(diagonalsx, [0,1,2], shape=(Nx+1,Nx+1))
        Dy =1/(dy)*sp.diags(diagonalsy, [-1, 0], shape=(Ny+2,Ny+1))
        

        LYY=Dy.transpose().dot(Dy)
        
        Ans = sp.kron(LYY,sp.eye(Nx+1))+sp.kron(sp.eye(Ny+1),LXX)
        return Ans
    
   def FDLaplacian2D_xbackward(Nx,Ny,dx,dy):

        diagonalsx = [-1*np.ones((1,Nx-1)),2*np.ones((1,Nx)),-1*np.ones((1,Nx+1))]
        diagonalsy = [-1*np.ones((1,Ny+2)),np.ones((1,Ny+1))]
        
        LXX =1/(dx**2)*sp.diags(diagonalsx, [-2,-1,0], shape=(Nx+1,Nx+1))
        Dy =1/(dy)*sp.diags(diagonalsy, [-1, 0], shape=(Ny+2,Ny+1))
        

        LYY=Dy.transpose().dot(Dy)
        
        Ans = sp.kron(LYY,sp.eye(Nx+1))+sp.kron(sp.eye(Ny+1),LXX)
        return Ans
    
   def FDLaplacian2D_xbackward_ybackward(Nx,Ny,dx,dy):

        diagonalsx = [-1*np.ones((1,Nx-1)),2*np.ones((1,Nx)),-1*np.ones((1,Nx+1))]
        diagonalsy = [-1*np.ones((1,Ny-1)),2*np.ones((1,Ny)),-1*np.ones((1,Ny+1))]
        
        LXX =1/(dx**2)*sp.diags(diagonalsx, [-2,-1,0], shape=(Nx+1,Nx+1))
        LYY =1/(dy**2)*sp.diags(diagonalsy, [-2,-1,0], shape=(Ny+1,Ny+1))
        
        Ans = sp.kron(LYY,sp.eye(Nx+1))+sp.kron(sp.eye(Ny+1),LXX)
        return Ans
    
   def FDLaplacian2D_xforward_ybackward(Nx,Ny,dx,dy):

        diagonalsx = [-1*np.ones((1,Nx+1)),2*np.ones((1,Nx)),-1*np.ones((1,Nx-1))]
        diagonalsy = [-1*np.ones((1,Ny-1)),2*np.ones((1,Ny)),-1*np.ones((1,Ny+1))]
        
        LXX =1/(dx**2)*sp.diags(diagonalsx, [0,1,2], shape=(Nx+1,Nx+1))
        LYY =1/(dy**2)*sp.diags(diagonalsy, [-2,-1,0], shape=(Ny+1,Ny+1))
        
    
        Ans = sp.kron(LYY,sp.eye(Nx+1))+sp.kron(sp.eye(Ny+1),LXX)
        return Ans
    
   def FDLaplacian2D_xbackward_yforward(Nx,Ny,dx,dy):

        diagonalsx = [-1*np.ones((1,Nx-1)),2*np.ones((1,Nx)),-1*np.ones((1,Nx+1))]
        diagonalsy = [-1*np.ones((1,Ny+1)),2*np.ones((1,Ny)),-1*np.ones((1,Ny-1))]
        
        LXX =1/(dx**2)*sp.diags(diagonalsx, [-2,-1,0], shape=(Nx+1,Nx+1))
        LYY =1/(dy**2)*sp.diags(diagonalsy, [0,1,2], shape=(Ny+1,Ny+1))

        Ans = sp.kron(LYY,sp.eye(Nx+1))+sp.kron(sp.eye(Ny+1),LXX)
        return Ans
    
   def FDLaplacian2D_xforward_yforward(Nx,Ny,dx,dy):

        diagonalsx = [-1*np.ones((1,Nx+1)),2*np.ones((1,Nx)),-1*np.ones((1,Nx-1))]
        diagonalsy = [-1*np.ones((1,Ny+1)),2*np.ones((1,Ny)),-1*np.ones((1,Ny-1))]
        
        LXX =1/(dx**2)*sp.diags(diagonalsx, [0,1,2], shape=(Nx+1,Nx+1))
        LYY =1/(dy**2)*sp.diags(diagonalsy, [0,1,2], shape=(Ny+1,Ny+1))

        
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
    
   def LX_forward(Nx,Ny,dx,dy):
        
        diagonalsx = [-1*np.ones((1,Nx+1)),np.ones((1,Nx))]


        Dx =1/(dx)*sp.diags(diagonalsx, [0, 1], shape=(Nx+1,Nx+1))
        
        Ans = sp.kron(sp.eye(Ny+1),Dx)
        
        return Ans

       
   def LY_forward(Nx,Ny,dx,dy):
        
        diagonalsy = [-1*np.ones((1,Ny+1)),np.ones((1,Ny))]


        Dy =1/(dy)*sp.diags(diagonalsy, [0, 1], shape=(Ny+1,Ny+1))

        Ans = sp.kron(Dy,sp.eye(Nx+1))

        return Ans
    
   def LX_backward(Nx,Ny,dx,dy):
        
        diagonalsx = [-1*np.ones((1,Nx)),np.ones((1,Nx+1))]


        Dx =1/(dx)*sp.diags(diagonalsx, [-1 ,0], shape=(Nx+1,Nx+1))
        
        Ans = sp.kron(sp.eye(Ny+1),Dx)
        
        return Ans

       
   def LY_backward(Nx,Ny,dx,dy):
        
        diagonalsy = [-1*np.ones((1,Ny)),np.ones((1,Ny+1))]


        Dy =1/(dy)*sp.diags(diagonalsy, [-1, 0], shape=(Ny+1,Ny+1))

        Ans = sp.kron(Dy,sp.eye(Nx+1))

        return Ans
        
 
# 2D FD Laplacian and identity matrices

'''  Used matrix identies '''

LxD  =  LX(P.Nx,P.Ny,P.dx,P.dy)

LyD  =  LY(P.Nx,P.Ny,P.dx,P.dy)

A_xy   =  FDLaplacian2D(P.Nx,P.Ny,P.dx,P.dy)

A_x   =  FDLaplacianx(P.Nx,P.Ny,P.dx,P.dy)
A_y   =  FDLaplaciany(P.Nx,P.Ny,P.dx,P.dy)

# if True:
#     A_xy_yf = FDLaplacian2D(P.Nx,P.Ny,P.dx,P.dy,False,False,False,True)#_yforward(P.Nx,P.Ny,P.dx,P.dy)
#     A_xy_yb = FDLaplacian2D(P.Nx,P.Ny,P.dx,P.dy,False,True,False,False)#_ybackward(P.Nx,P.Ny,P.dx,P.dy)
#     A_xy_xf = FDLaplacian2D(P.Nx,P.Ny,P.dx,P.dy,True,False,False,False)#_xforward(P.Nx,P.Ny,P.dx,P.dy)
#     A_xy_xb = FDLaplacian2D(P.Nx,P.Ny,P.dx,P.dy,False,False,True,False)#_xbackward(P.Nx,P.Ny,P.dx,P.dy)
    
    
#     A_xy_xf_yf = FDLaplacian2D(P.Nx,P.Ny,P.dx,P.dy,True,False,False,True)#_xforward_yforward(P.Nx,P.Ny,P.dx,P.dy)
#     A_xy_xb_yf = FDLaplacian2D(P.Nx,P.Ny,P.dx,P.dy,False,False,True,True)#_xbackward_yforward(P.Nx,P.Ny,P.dx,P.dy)
#     A_xy_xf_yb = FDLaplacian2D(P.Nx,P.Ny,P.dx,P.dy,True,True,False,False)#_xforward_ybackward(P.Nx,P.Ny,P.dx,P.dy)
#     A_xy_xb_yb = FDLaplacian2D(P.Nx,P.Ny,P.dx,P.dy,False,True,True,False)#_xbackward_ybackward(P.Nx,P.Ny,P.dx,P.dy)
# else:
#     A_xy_yf = -FDLaplacian2D_yforward(P.Nx,P.Ny,P.dx,P.dy)
#     A_xy_yb = -FDLaplacian2D_ybackward(P.Nx,P.Ny,P.dx,P.dy)
#     A_xy_xf = -FDLaplacian2D_xforward(P.Nx,P.Ny,P.dx,P.dy)
#     A_xy_xb = -FDLaplacian2D_xbackward(P.Nx,P.Ny,P.dx,P.dy)
    
    
#     A_xy_xf_yf = -FDLaplacian2D_xforward_yforward(P.Nx,P.Ny,P.dx,P.dy)
#     A_xy_xb_yf = -FDLaplacian2D_xbackward_yforward(P.Nx,P.Ny,P.dx,P.dy)
#     A_xy_xf_yb = -FDLaplacian2D_xforward_ybackward(P.Nx,P.Ny,P.dx,P.dy)
#     A_xy_xb_yb = -FDLaplacian2D_xbackward_ybackward(P.Nx,P.Ny,P.dx,P.dy) 
I = sp.eye((P.Ny+1)*(P.Nx+1))


LxD_f = LX_forward(P.Nx,P.Ny,P.dx,P.dy)
LxD_b = LX_backward(P.Nx,P.Ny,P.dx,P.dy)

LyD_f = LY_forward(P.Nx,P.Ny,P.dx,P.dy)
LyD_b = LY_backward(P.Nx,P.Ny,P.dx,P.dy)


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
    fzetas += (P.WestBoundary+P.NWCorner+P.SWCorner)*(
                            -zetas + 0
                            )
    
    ''' east boundary '''
    'x=1' 

    fzetas += (P.EastBoundary+P.NECorner+P.SECorner)*(
                        -zetas + P.Atilde*np.sin(P.phi)
                        )

    ''' South Boundary '''
    'y=0' 
    fzetas += P.SouthBoundary*(
                               -zetas-(1-h)*(LxD*uc+LyD_f*vc)+uc*(LxD*h)+vc*(LyD_f*h) #-zetas-(1-h)*(LxD*uc)+uc*(LxD*h)
                                )

    
    ''' North Boundary ''' 
    'y=1'
    fzetas += P.NorthBoundary*(
                                -zetas-(1-h)*(LxD*uc+LyD_b*vc)+uc*(LxD*h)+vc*(LyD_b*h) #-zetas-(1-h)*(LxD*uc)+uc*(LxD*h)
                                )


    ''' quick corner fix'''
    
    #fzetas += P.NWCorner*(-zetas+0)+P.SWCorner*(-zetas+0)+P.NECorner*(-zetas + P.Atilde*np.sin(P.phi))+P.SECorner*(-zetas + P.Atilde*np.sin(P.phi))
    #fzetas += cornerfix(zetas)
    return fzetas

def Fzetac(zetas,zetac,us,uc,vs,vc,C,h):
    ''' interior '''
    fzetac =-zetac+(1-h)*(LxD*us+LyD*vs)-us*(LxD*h)-vs*(LyD*h)
    
    fzetac = P.Interior*fzetac
    
    ''' west boundary '''
    'x=0' 
    fzetac += (P.WestBoundary+P.NWCorner+P.SWCorner)*(
                            -zetac + 1
                             )
    ''' east boundary '''
    'x=1' 

    fzetac += (P.EastBoundary+P.NECorner+P.SECorner)*(
                                -zetac + P.Atilde*np.cos(P.phi)
                            )
    ''' South Boundary '''
    'y=0' 
    fzetac += P.SouthBoundary*(
                               -zetac+(1-h)*(LxD*us+LyD_f*vs)-us*(LxD*h)-vs*(LyD_f*h) #-zetac+(1-h)*(LxD*us)-us*(LxD*h)
                                )
    
    ''' North Boundary ''' 
    'y=1' 
    fzetac += P.NorthBoundary*(
                              -zetac+(1-h)*(LxD*us+LyD_b*vs)-us*(LxD*h)-vs*(LyD_b*h)# -zetac+(1-h)*(LxD*us)-us*(LxD*h)
                               )#LyD_b*zetac

    ''' quick corner fix'''
    
    #fzetac += P.NWCorner*(-zetac+1)+P.SWCorner*(-zetac+1)+P.NECorner*(-zetac + P.Atilde*np.cos(P.phi))+P.SECorner*(-zetac + P.Atilde*np.cos(P.phi))
    #fzetac += cornerfix(zetac)
    return fzetac


def Fus(zetas,zetac,us,uc,vs,vc,C,h):
    ''' interior '''
    fus   = -us -np.divide(P.r, 1-h)*uc-P.lambda_L**(2)*LxD*zetas
    
    fus = P.Interior*fus
    
    ''' west boundary '''
    'x=0' 
    ' KNOWN : zetas = 0 zetac = A  '
    ' UNKOWN : us, uc '
    fus   += (P.WestBoundary+P.NWCorner+P.SWCorner)*(
                            -us-np.divide(P.r, 1 - 0 )*uc-P.lambda_L**(2)*(LxD_f*zetas)
                            )
    ''' east boundary ''' 
    'x=1' 

    fus += (P.EastBoundary+P.NECorner+P.SECorner)*(
                             -us-np.divide(P.r, 1 - (1-P.H2/P.H1) )*uc-P.lambda_L**(2)*(LxD_b*zetas)
                             )

         
    ''' South Boundary '''
    'y=0' 
    fus += P.SouthBoundary*(
                            -us-np.divide(P.r, 1 - h)*uc-P.lambda_L**(2)*LxD*zetas
                            )
    
    ''' North Boundary ''' 
    'y=1' 
    fus += P.NorthBoundary*(
                            -us-np.divide(P.r, 1-h)*uc-P.lambda_L**(2)*LxD*zetas
                            )
    
    
    ''' quick corner fix'''
    
    # fus += P.NWCorner*(
    #                   -us
    #                   -np.divide(P.r, 1-h)*uc
    #                   -P.lambda_L**(2)*(LxD_f*zetas)
    #                   )
    # fus += P.SWCorner*(
    #                   -us
    #                   -np.divide(P.r, 1-h)*uc
    #                   -P.lambda_L**(2)*(LxD_f*zetas))
    
    # fus += P.NECorner*(
    #                   -us
    #                   -np.divide(P.r, 1-h)*uc
    #                   -P.lambda_L**(2)*(LxD_b*zetas)
    #                   )
    # fus += P.SECorner*(
    #                   -us
    #                   -np.divide(P.r, 1-h)*uc
    #                   -P.lambda_L**(2)*(LxD_b*zetas)
    #                   )
    return fus
         
def Fuc(zetas,zetac,us,uc,vs,vc,C,h):
    ''' interior '''
    fuc   = -uc  +np.divide(P.r, 1-h)*us +P.lambda_L**(2)*LxD*zetac
    
    fuc = P.Interior*fuc 
    
    ''' west boundary '''
    'x=0' 
    ' KNOWN : zetas = 0 zetac = A  '
    ' UNKOWN : us, uc '
    fuc     +=  (P.WestBoundary+P.NWCorner+P.SWCorner)*(
                                -uc+np.divide(P.r, 1 - 0)*us+P.lambda_L**(2)*LxD_f*zetac
                                )
                            
    ''' east boundary '''
    'x=1' 

    fuc += (P.EastBoundary+P.NECorner+P.SECorner)*(
                         -uc+np.divide(P.r, 1 - (1-P.H2/P.H1))*us+P.lambda_L**(2)*LxD_b*zetac
                        )

    ''' South Boundary '''
    'y=0' 
    fuc += P.SouthBoundary*(
                            -uc+np.divide(P.r, 1 - h)*us+P.lambda_L**(2)*LxD*zetac
                            )
    
    ''' North Boundary ''' 
    'y=1' 
    fuc += P.NorthBoundary*(
                            -uc+np.divide(P.r, 1 - h)*us+P.lambda_L**(2)*LxD*zetac
                            )
    
    # ''' quick corner fix'''
    
    # fuc += P.NWCorner*( 
    #                   -uc + np.divide(P.r, 1-h)*us + P.lambda_L**(2)*LxD_f*zetac
    #                   )
    
    # fuc += P.SWCorner*( 
    #                     -uc + np.divide(P.r, 1-h)*us + P.lambda_L**(2)*LxD_f*zetac
    #                     )
    
    # fuc += P.NECorner*(
    #                   -uc + np.divide(P.r, 1-h)*us + P.lambda_L**(2)*LxD_b*zetac
    #                   )
    # fuc += P.SECorner*(
    #                   -uc + np.divide(P.r, 1-h)*us + P.lambda_L**(2)*LxD_b*zetac
    #                   )
    #fuc += cornerfix(uc)
    return fuc

def Fvs(zetas,zetac,us,uc,vs,vc,C,h):
    ''' interior '''
    fvs   = -vs  -np.divide(P.r, 1 - h)*vc +(P.Lx/P.Ly)**2*-P.lambda_L**(2)*LyD*zetas
    
    fvs = P.Interior*fvs
    
    ''' west boundary '''
    ' KNOWN : zetas = 0 zetac = A  '
    ' UNKOWN : us, uc '
    fvs   +=  P.WestBoundary*(-vs -np.divide(P.r, 1-0)*vc +(P.Lx/P.Ly)**2*-P.lambda_L**(2)*LyD*zetas)
                            
    #                         )
    # ''' east boundary ''' 

    fvs += P.EastBoundary*(-vs -np.divide(P.r, 1-(1-P.H2/P.H1))*vc +(P.Lx/P.Ly)**2*-P.lambda_L**(2)*LyD*zetas)
    #                          )
         
    ''' South Boundary '''
    fvs += (P.SouthBoundary+P.SWCorner+P.SECorner)*(-vs+0)
    
    ''' North Boundary ''' 
    fvs += (P.NorthBoundary+P.NWCorner+P.NECorner)*(-vs+0)
    '''  corner '''
    
    # fvs += P.NWCorner*(
    #                      -vs-np.divide(P.r, 1-h)*vc -(P.Lx/P.Ly)**2*P.lambda_L**(2)*LyD_b*zetas
    #                 )
    
    # fvs += P.SWCorner*(
    #                      -vs-np.divide(P.r, 1-h)*vc -(P.Lx/P.Ly)**2*P.lambda_L**(2)*LyD_f*zetas
    #                     )
    
    # fvs += P.NECorner*(
    #                      -vs-np.divide(P.r, 1-h)*vc -(P.Lx/P.Ly)**2*P.lambda_L**(2)*LyD_b*zetas
    #                     )
    
    # fvs += P.SECorner*(
    #                      -vs-np.divide(P.r, 1-h)*vc -(P.Lx/P.Ly)**2*P.lambda_L**(2)*LyD_f*zetas
    #                     )
    return fvs

def Fvc(zetas,zetac,us,uc,vs,vc,C,h):
    ''' interior '''
    fvc  = -vc   +np.divide(P.r, 1-h)*vs +(P.Lx/P.Ly)**2*P.lambda_L**(2)*LyD*zetac
    
    fvc  = P.Interior*fvc
    
    ''' west boundary '''
    ' KNOWN : zetas = 0 zetac = A  '
    ' UNKOWN : us, uc '
    fvc   +=  P.WestBoundary*(
                             -vc+np.divide(P.r,1-0)*vs+(P.Lx/P.Ly)**2*P.lambda_L**(2)*LyD*zetac
                             )
                            
    #                         )
    ''' east boundary ''' 

    fvc += P.EastBoundary*(
                            -vc+np.divide(P.r,1-(1-P.H2/P.H1))*vs+(P.Lx/P.Ly)**2*P.lambda_L**(2)*LyD*zetac
                            )
    #                          )
         
    ''' South Boundary '''
    fvc += (P.SouthBoundary+P.SWCorner+P.SECorner)*(-vc+0)
    
    ''' North Boundary ''' 
    fvc += (P.NorthBoundary+P.NWCorner+P.NECorner)*(-vc+0)
    
    # # ''' quick corner fix'''
    
    # fvc += P.NWCorner*(
    #                     -vc+np.divide(P.r,1-h)*vs+(P.Lx/P.Ly)**2*P.lambda_L**(2)*LyD_b*zetac
    #                     )
    
    # fvc += P.SWCorner*(
    #                     -vc+np.divide(P.r,1-h)*vs+(P.Lx/P.Ly)**2*P.lambda_L**(2)*LyD_f*zetac
    #                     )
    
    # fvc += P.NECorner*(
    #                     -vc+np.divide(P.r,1-h)*vs+(P.Lx/P.Ly)**2*P.lambda_L**(2)*LyD_b*zetac
    #                     )
    
    # fvc += P.SECorner*(
    #                     -vc+np.divide(P.r,1-h)*vs+(P.Lx/P.Ly)**2*P.lambda_L**(2)*LyD_f*zetac
    #                     )
    return fvc

def FC(zetas,zetac,us,uc,vs,vc,C,h):
    ''' interior '''
    fC  =           P.a*P.k*(
                            A_x*C+(P.Lx/P.Ly)**2*A_y*C 
                                    # + 0*P.lambda_d*(
                                    #             C*( (LxD*(beta(h)))*(LxD*h) + (P.Lx/P.Ly)**2*LyD*beta(h)*LyD*h )
                                    #             + beta(h)*( LxD*h*LxD*C + (P.Lx/P.Ly)**2*LyD*h*LyD*C)
                                    #             +C*beta(h)*( A_x*h + (P.Lx/P.Ly)**2*A_y*h )
                                    #             )
                            )
    fC += 0.5*(us*us+uc*uc+(P.Ly/P.Lx)**2*(vs*vs+vc*vc))-1*beta(h)*C
    
    fC = P.Interior*fC
    
    ''' west boundary '''
    ' Lx => Lx_f  '
    ' A_xy => A_xy_xf '
    fC   +=  P.WestBoundary*(#P.a*P.k*(LxD_f*C+P.lambda_d*C*beta(h)*LxD_f*h))
                            # P.a*P.k*(
                            # A_xy_xf*C+ P.lambda_d*(
                            #                     C*(LxD_f*beta(h)*LxD_f*h + LyD*beta(h)*LyD*h)
                            #                     + beta(h)*(LxD_f*h*LxD_f*C+LyD*h*LyD*C)
                            #                     +C*beta(h)*A_xy_xf*h
                            #                     )
                            # )
                             -1*beta(h)*C+0.5*(us*us+uc*uc+(P.Ly/P.Lx)**2*(vs*vs+vc*vc)))
                            
    ''' east boundary ''' 

    fC   += P.EastBoundary*(#-P.a*P.k*(LxD_b*C+P.lambda_d*C*beta(h)*LxD_b*h))
                            # P.a*P.k*(
                            # A_xy_xb*C+ P.lambda_d*(
                            #                     C*(LxD_b*beta(h)*LxD_b*h + LyD*beta(h)*LyD*h)
                            #                     + beta(h)*(LxD_b*h*LxD_b*C+LyD*h*LyD*C)
                            #                     +C*beta(h)*A_xy_xb*h
                            #                     )
                            # )
                            (-1*beta(h)*C+0.5*(us*us+uc*uc+(P.Ly/P.Lx)**2*(vs*vs+vc*vc))))
         
    ''' South Boundary '''
    fC += (P.SouthBoundary+P.SWCorner+P.SECorner)*( #P.a*P.k*((P.Lx/P.Ly)*LyD_f*C+P.lambda_d*C*beta(h)*LyD_f*h))
                            # P.a*P.k*(
                            # A_xy_yf*C+ P.lambda_d*(
                            #                     C*(LxD*beta(h)*LxD*h + LyD_f*beta(h)*LyD_f*h)
                            #                     + beta(h)*(LxD*h*LxD*C+LyD_f*h*LyD_f*C)
                            #                     +C*beta(h)*A_xy_yf*h
                            #                     )
                            # )
                           (-1*beta(h)*C+0.5*(us*us+uc*uc+0*(P.Ly/P.Lx)**2*(vs*vs+vc*vc))))
    
    ''' North Boundary ''' 
    fC += (P.NorthBoundary+P.NWCorner+P.NECorner)*(#-P.a*P.k*(LyD_b*C+P.lambda_d*C*beta(h)*LyD_b*h))
                            # P.a*P.k*(
                            #     A_xy_yb*C+ P.lambda_d*(
                            #                      C*(LxD*beta(h)*LxD*h + LyD_b*beta(h)*LyD_b*h)
                            #                      + beta(h)*(LxD*h*LxD*C+LyD_b*h*LyD_b*C)
                            #                      +C*beta(h)*A_xy_yb*h
                            #                      )
                            #  )
                           (-1*beta(h)*C+0.5*(us*us+uc*uc+0*(P.Ly/P.Lx)**2*(vs*vs+vc*vc))))
    
    # ''' quick corner fix'''
    
    # fC += P.NWCorner*(#-P.a*P.k*(LyD_b*C+P.lambda_d*C*beta(h)*LyD_b*h))
    #                         # P.a*P.k*(
    #                         #     A_xy_xf_yb*C+ P.lambda_d*(
    #                         #                      C*(LxD_f*beta(h)*LxD_f*h + LyD_b*beta(h)*LyD_b*h)
    #                         #                      + beta(h)*(LxD_f*h*LxD_f*C+LyD_b*h*LyD_b*C)
    #                         #                      +C*beta(h)*A_xy_xf_yb*h
    #                         #                      )
    #                         #  )
    #                  -1*beta(0)*C+0.5*(us*us+uc*uc+0*(P.Ly/P.Lx)**2*(vs*vs+vc*vc)))
    
    # fC += P.SWCorner*(# P.a*P.k*(LyD_f*C+P.lambda_d*C*beta(h)*LyD_f*h))
    #                        # P.a*P.k*(
    #                        #      A_xy_xf_yf*C+ P.lambda_d*(
    #                        #                       C*(LxD_f*beta(h)*LxD_f*h + LyD_f*beta(h)*LyD_f*h)
    #                        #                       + beta(h)*(LxD_f*h*LxD_f*C+LyD_f*h*LyD_f*C)
    #                        #                       +C*beta(h)*A_xy_xf_yf*h
    #                        #                       )
    #                        #   )

    #                     -1*beta(0)*C+0.5*(us*us+uc*uc+0*(P.Ly/P.Lx)**2*(vs*vs+vc*vc)))
    
    # fC += P.NECorner*(#-P.a*P.k*(LyD_b*C+P.lambda_d*C*beta(h)*LyD_b*h))
    #                         # P.a*P.k*(
    #                         #     A_xy_xb_yb*C+ P.lambda_d*(
    #                         #                      C*(LxD_b*beta(h)*LxD_b*h + LyD_b*beta(h)*LyD_b*h)
    #                         #                      + beta(h)*(LxD_b*h*LxD_b*C+LyD_b*h*LyD_b*C)
    #                         #                      +C*beta(h)*A_xy_xb_yb*h
    #                         #                      )
    #                         #  )
    #                    -1*beta(1-P.H2/P.H1)*C+0.5*(us*us+uc*uc+0*(P.Ly/P.Lx)**2*(vs*vs+vc*vc)))
    
    # fC += P.SECorner*(# P.a*P.k*(LyD_f*C+P.lambda_d*C*beta(h)*LyD_f*h))
    #                         # P.a*P.k*(
    #                         #     A_xy_xb_yf*C+ P.lambda_d*(
    #                         #                      C*(LxD_b*beta(h)*LxD_b*h + LyD_f*beta(h)*LyD_f*h)
    #                         #                      + beta(h)*(LxD_b*h*LxD_b*C+LyD_f*h*LyD_f*C)
    #                         #                      +C*beta(h)*A_xy_xb_yf*h
    #                         #                      )
    #                         #     )
    #                   -1*beta(1-P.H2/P.H1)*C+0.5*(us*us+uc*uc+0*(P.Ly/P.Lx)**2*(vs*vs+vc*vc)))
    return fC

def Fh(zetas,zetac,us,uc,vs,vc,C,h):
    # if N=0 (how it should be) the system is not stable
    # 
    N=0#2.7
    
    ''' interior '''
    fh  = -P.Mutilde*(A_x*h+(P.Lx/P.Ly)**2*A_y*h) #-P.Mutilde*A_xy*h
    if False:
        fh  += -10**(-N)*P.delta_s*(-1*beta(h)*C+0.5*(us*us+uc*uc+(P.Ly/P.Lx)**2*(vs*vs+vc*vc)))
    else:
        fh  += -10**(-N)*P.delta_s*P.a*P.k*(
                            A_x*C+(P.Lx/P.Ly)**2*A_y*C 
                                    # + 0*P.lambda_d*(
                                    #             C*( (LxD*(beta(h)))*(LxD*h) + (P.Lx/P.Ly)**2*LyD*beta(h)*LyD*h )
                                    #             + beta(h)*( LxD*h*LxD*C + (P.Lx/P.Ly)**2*LyD*h*LyD*C)
                                    #             +C*beta(h)*( A_x*h + (P.Lx/P.Ly)**2*A_y*h )
                                    #             )
                            )
    fh = P.Interior*fh
    
    ''' west boundary '''
    ' KNOWN : zetas = 0 zetac = A  '
    ' UNKOWN : us, uc '
    fh   +=  (P.WestBoundary+P.NWCorner+P.SWCorner)*(-h+0)
                            
    #                         )
    ''' east boundary ''' 

    fh += (P.EastBoundary+P.NECorner+P.SECorner)*(-h+1-P.H2/P.H1)
    #                          )
         
    ''' South Boundary '''
    fh += P.SouthBoundary*(LyD_f*h)
                            #-P.Mutilde*P.Lx/P.Ly*LyD_f*h +10**(-N)*P.delta_s*P.a*P.k*((P.Lx/P.Ly*LyD_f*C+P.lambda_d*C*beta(h)*LyD_f*h)))
                            # +10**(-N)*P.delta_s*P.a*P.k*(
                            # A_xy_yf*C + P.lambda_d*(
                            #                     C*(LxD*beta(h)*LxD*h + LyD_f*beta(h)*LyD_f*h)
                            #                     + beta(h)*(LxD*h*LxD*C+LyD_f*h*LyD_f*C)
                            #                     +C*beta(h)*A_xy_yf*h
                            #                     )
                            # ))
    
    ''' North Boundary ''' 
    fh += P.NorthBoundary*(LyD_b*h)
        #                   -P.Mutilde*P.Lx/P.Ly*LyD_b*h +10**(-N)*P.delta_s*P.a*P.k*((P.Lx/P.Ly*LyD_b*C+P.lambda_d*C*beta(h)*LyD_b*h)))
                            
                            # +10**(-N)*P.delta_s*P.a*P.k*(
                            #   A_xy_yb*C + P.lambda_d*(
                            #                       C*(LxD*beta(h)*LxD*h + LyD_b*beta(h)*LyD_b*h)
                            #                       + beta(h)*(LxD*h*LxD*C+LyD_b*h*LyD_b*C)
                            #                       +C*beta(h)*A_xy_yb*h
                            #                       )
                            #   )
                            #   )
    
    # ''' quick corner fix'''
    
    # fh += P.NWCorner*(-h+0)
    
    # fh += P.SWCorner*(-h+0)
    
    # fh += P.NECorner*(-h+1-P.H2/P.H1)
    
    # fh += P.SECorner*(-h+1-P.H2/P.H1)
    return fh

