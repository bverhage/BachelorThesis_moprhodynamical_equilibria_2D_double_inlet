# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:16:14 2021

@author: billy
"""

if True:
    print('----------------------------------------------- ')
    print('\n \t Welcome to the code of NR-BEP.py \n ')
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
    
if True:
    Lx = 1*59*10**3 #m
    
    Ly = 1*10**3 #m
    
    H1 = 12#m
    
    H2 = 6
    
    A = 0.74*1/H1
    
    Atilde = 0.84*1/H1
    
    phi = 41/180*np.pi
    
    g = 9.81 #m/s^2
    
    CD = 0.0025
    sigma = 1.4*10**-4 #/s
    
    lamnda = sigma*Lx/(np.sqrt(g*H1))
    Nx = 50+1
    Ny = int((Nx-1)*1/2)+1
    
    dx = 1/Nx
    dy = 1/Ny
    
    a = 6.222*10**(-2)
    r = -(8*CD*A*Lx)/(3*np.pi*H1**2)/(H1*sigma)
    
    
    fhat = 7.1*10**-1
    
    k = 2.052*10**(-4)
    
    Mutilde = 10**(-6)
    
    epsilon = A/H1
    
    lamndad = 1.8
    
    delta_s = 0#9.5*10**(-4)  
    
    BOOL_two_open_ends = True
    
    BOOL_print_NR = False
    
    
    
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
        if y==0 and x==0: return 1
        else: return 0
    
    def NEcornerpoint(x,y):
        if y==0 and x==1: return 1
        else: return 0
        
    def SWcornerpoint(x,y):
        if y==1 and x==0: return 1
        else: return 0
    
    def SEcornerpoint(x,y):
        if y==1 and x==1: return 1
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
    h    = create(Nx,Ny,bedprofile)


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
    colsLX, rowsLX, valsLX = sp.find(Encolsure>0)
    
def reshape(LEXarray):
    return np.reshape(LEXarray,[Ny+1,Nx+1])

    print('\n Boundary checked commensing Newton Rapsons algorithem')

if True:
    
   def FDLaplacian2D(Nx,Ny,dx,dy):

        diagonalsx = [-1*np.ones((1,Nx+2)),np.ones((1,Nx+1))]
        diagonalsy = [-1*np.ones((1,Ny+2)),np.ones((1,Ny+1))]
        
        Dx =1/(dx)*sp.diags(diagonalsx, [-1, 0], shape=(Nx+2,Nx+1))
        Dy =1/(dy)*sp.diags(diagonalsy, [-1, 0], shape=(Ny+2,Ny+1))
        
        LXX=Dx.transpose().dot(Dx)
        LYY=Dy.transpose().dot(Dy)*0
   

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

LxD  =  LX(Nx,Ny,dx,dy)

LyD  =  LY(Nx,Ny,dx,dy)

A_x  =  FDLaplacian2D(Nx,Ny,dx,dy)


I = sp.eye((Ny+1)*(Nx+1))

I_xoffR = sp.diags(np.ones((Ny+1)*(Nx+1)-1), offsets=1)
I_xoffL = sp.diags(np.ones((Ny+1)*(Nx+1)-1), offsets=-1)
I_yoffR = sp.diags(np.ones((Ny+1)*(Nx+1)-(Nx+1)), offsets=(Nx+1))
I_yoffL = sp.diags(np.ones((Ny+1)*(Nx+1)-(Nx+1)), offsets=-(Nx+1))

''' Initial condition '''

Uinnitalguess=np.concatenate((ICzeta0,ICzeta0,ICu0,ICu0,ICv0,ICv0))


'''

   part 2 the Newton rapson method 
   
   first we need to define some functions:
       
'''


def MaxNormOfU(U):
     ''' The max 2 norm of U=(u,v)^T w'''
     zetas,zetac,us,uc,vs,vc=np.array_split(U,6)
     return np.max([
         np.linalg.norm(zetas),
         np.linalg.norm(zetac),
         np.linalg.norm(us),
         np.linalg.norm(uc),
         np.linalg.norm(vs),
         np.linalg.norm(vc)
         ])
 
def cornerfix(NP_array):
    ans= NP_array*(-NWCorner -NECorner -SWCorner -SECorner)
    return ans
    
def beta(h):
    return 1/(1-np.exp(-lamndad*(1-h)))


def Fzetas(zetas,zetac,us,uc,vs,vc):
    ''' interior '''
    
    fzetas =-zetas-(1-h)*(LxD*uc+LyD*vc)+uc*(LxD*h)+vc*(LyD*h)
    
    fzetas = Interior*fzetas
    
    
    
    ''' west boundary '''
    'x=0' 
    fzetas += WestBoundary*(
                            -zetas + 0
                            )
    
    ''' east boundary '''
    'x=1' 
    if BOOL_two_open_ends:
        fzetas += EastBoundary*(
                                -zetas + Atilde*np.sin(phi)

                                )
    else:
        fzetas += EastBoundary*(
                                1/dx*(-I_xoffL+I)*zetas 
                                )
    ''' South Boundary '''
    'y=0' 
    fzetas += SouthBoundary*1/dy*(-zetas+I_yoffR*zetas)
    
    ''' North Boundary ''' 
    'y=1'
    fzetas += NorthBoundary*1/dy*(-I_yoffL*zetas+zetas)


    ''' quick corner fix'''
    
    fzetas += NWCorner*(-zetas+0)+SWCorner*(-zetas+0)+NECorner*(-zetas + Atilde*np.sin(phi))+SECorner*(-zetas + Atilde*np.sin(phi))
    #fzetas += cornerfix(zetas)
    return fzetas

def Fzetac(zetas,zetac,us,uc,vs,vc):
    ''' interior '''
    fzetac =-zetac+(1-h)*(LxD*us+LyD*vs)-us*(LxD*h)-vs*(LyD*h)
    
    fzetac = Interior*fzetac
    
    ''' west boundary '''
    'x=0' 
    fzetac += WestBoundary*(
                            -zetac + A
  
                            )
    ''' east boundary '''
    'x=1' 
    if BOOL_two_open_ends:
        fzetac += EastBoundary*(
                                -zetac + Atilde*np.cos(phi)

            )
    else:
        fzetac += EastBoundary*(
                                1/dx*(-I_xoffL+I)*zetac
                                )
    ''' South Boundary '''
    'y=0' 
    fzetac += SouthBoundary*1/dy*(-zetac+I_yoffR*zetac)
    
    ''' North Boundary ''' 
    'y=1' 
    fzetac += NorthBoundary*1/dy*(-I_yoffL*zetac +zetac)

    ''' quick corner fix'''
    
    fzetac += NWCorner*(-zetac+A)+SWCorner*(-zetac+A)+NECorner*(-zetac + Atilde*np.cos(phi))+SECorner*(-zetac + Atilde*np.cos(phi))
    #fzetac += cornerfix(zetac)
    return fzetac


def Fus(zetas,zetac,us,uc,vs,vc):
    ''' interior '''
    fus   = -us +fhat*vc-np.divide(r, 1-h)*uc-lamnda**(-2)*LxD*zetas
    
    fus = Interior*fus
    
    ''' west boundary '''
    'x=0' 
    ' KNOWN : zetas = 0 zetac = A  '
    ' UNKOWN : us, uc '
    fus   +=  WestBoundary*(
                            -us
                            +fhat*vc
                            -np.divide(r, 1-h)*uc
                            -lamnda**(-2)*1/(dx)*(-0 +I_xoffR*zetas)
                            )
    ''' east boundary ''' 
    'x=1' 
    if BOOL_two_open_ends:
         ' KNOWN : zetas = 0 zetac = A  '
         ' UNKOWN : us, uc '
         fus += EastBoundary*(
                             -us
                             +fhat*vc
                             -np.divide(r, 1-h)*uc
                             -lamnda**(-2)*1/dx*(-I_xoffL*zetas +Atilde*np.sin(phi)) 
                             )
    else:
         ' KNOWN :  d/dx zetas=0  d/dx zetac = 0 , us=0 , uc = 0 '
         ' UNKOWN : zetas, zetac '
         fus += EastBoundary*(
                             -us+0
                        
                             )
         
    ''' South Boundary '''
    'y=0' 
    fus += SouthBoundary*(-us+fhat*vc-np.divide(r, 1-h)*uc-lamnda**(-2)*LxD*zetas)
    
    ''' North Boundary ''' 
    'y=1' 
    fus += NorthBoundary*(-us+fhat*vc-np.divide(r, 1-h)*uc-lamnda**(-2)*LxD*zetas)
    
    
    # ''' quick corner fix'''
    
    fus += NWCorner*(-us
                      -np.divide(r, 1-h)*uc
                      -lamnda**(-2)*1/dx*(-0 +I_xoffR*zetas))
    fus += SWCorner*( -us
                      -np.divide(r, 1-h)*uc
                      -lamnda**(-2)*1/dx*(-0 +I_xoffR*zetas))
    
    fus += NECorner*(-us
                      -np.divide(r, 1-h)*uc
                      -lamnda**(-2)*1/dx*(-I_xoffL*zetas +Atilde*np.sin(phi)))
    fus += SECorner*(-us
                      -np.divide(r, 1-h)*uc
                      -lamnda**(-2)*1/dx*(-I_xoffL*zetas +Atilde*np.sin(phi)))
    #fus += cornerfix(us)
    return fus
         
def Fuc(zetas,zetac,us,uc,vs,vc):
    ''' interior '''
    fuc   = -uc -fhat*vs +np.divide(r, 1-h)*us +lamnda**(-2)*LxD*zetac
    
    fuc = Interior*fuc 
    
    ''' west boundary '''
    'x=0' 
    ' KNOWN : zetas = 0 zetac = A  '
    ' UNKOWN : us, uc '
    fuc     +=  WestBoundary*(
                                -uc
                                -fhat*vs
                                +np.divide(r, 1-h)*us
                                
                                +lamnda**(-2)*1/dx*( -A +I_xoffR*zetac)
                                )
                            
    ''' east boundary '''
    'x=1' 
    if BOOL_two_open_ends:
        ' KNOWN  : zetas = Atilde*sin zetac = Atilde*cos '
        ' UNKOWN : us , uc '
        fuc += EastBoundary*(
                                -uc
                                -fhat*vs
                                +np.divide(r, 1-h)*us
                                +lamnda**(-2)*1/dx*( -I_xoffL*zetac  + Atilde*np.cos(phi))
                                
                                )
    else:
        fuc += EastBoundary*(
                            -uc+0
                            
                            )
    ''' South Boundary '''
    'y=0' 
    fuc += SouthBoundary*(-uc-fhat*vs+np.divide(r, 1-h)*us+lamnda**(-2)*1/dx*LxD*zetac)
    
    ''' North Boundary ''' 
    'y=1' 
    fuc += NorthBoundary*(-uc-fhat*vs+np.divide(r, 1-h)*us+lamnda**(-2)*1/dx*LxD*zetac)
    
    # ''' quick corner fix'''
    
    fuc += NWCorner*( -uc 
                      +np.divide(r, 1-h)*us
                      +lamnda**(-2)*1/dx*(-A + I_xoffR*zetac))
    
    fuc += SWCorner*( -uc
                        +np.divide(r, 1-h)*us
                        +lamnda**(-2)*1/dx*(-A +I_xoffR*zetac))
    
    fuc += NECorner*(-uc
                      +np.divide(r, 1-h)*us
                      +lamnda**(-2)*1/dx*(-I_xoffL*zetac  + Atilde*np.cos(phi)))
    fuc += SECorner*(-uc
                      +np.divide(r, 1-h)*us
                      +lamnda**(-2)*1/dx*(-I_xoffL*zetac  + Atilde*np.cos(phi)))
    #fuc += cornerfix(uc)
    return fuc

def Fvs(zetas,zetac,us,uc,vs,vc):
    ''' interior '''
    fvs   = -vs -fhat*uc -np.divide(r, 1-h)*vc -lamnda**(-2)*LyD*zetas
    
    fvs = Interior*fvs
    
    ''' west boundary '''
    ' KNOWN : zetas = 0 zetac = A  '
    ' UNKOWN : us, uc '
    fvs   +=  WestBoundary*(-vs-fhat*uc -np.divide(r, 1-h)*vc -lamnda**(-2)*LyD*zetas)
                            
    #                         )
    # ''' east boundary ''' 

    fvs += EastBoundary*(-vs-fhat*uc -np.divide(r, 1-h)*vc -lamnda**(-2)*LyD*zetas)
    #                          )
         
    ''' South Boundary '''
    fvs += SouthBoundary*(-vs+0)
    
    ''' North Boundary ''' 
    fvs += NorthBoundary*(-vs+0)
    ''' quick corner fix'''
    
    fvs += NWCorner*( -vs-fhat*uc -np.divide(r, 1-h)*vc -lamnda**(-2)*LyD*zetas)
    
    fvs += SWCorner*( -vs-fhat*uc -np.divide(r, 1-h)*vc -lamnda**(-2)*LyD*zetas)
    
    fvs += NECorner*(-vs-fhat*uc -np.divide(r, 1-h)*vc -lamnda**(-2)*LyD*zetas)
    
    fvs += SECorner*(-vs-fhat*uc -np.divide(r, 1-h)*vc -lamnda**(-2)*LyD*zetas)
    return fvs

def Fvc(zetas,zetac,us,uc,vs,vc):
    ''' interior '''
    fvc   = -vc +fhat*us   +np.divide(r, 1-h)*vs +lamnda**(-2)*LyD*zetac
    
    fvc = Interior*fvc
    
    ''' west boundary '''
    ' KNOWN : zetas = 0 zetac = A  '
    ' UNKOWN : us, uc '
    fvc   +=  WestBoundary*(-vc+fhat*us+np.divide(r,1-h)*vs+lamnda**(-2)*LyD*zetac)
                            
    #                         )
    # ''' east boundary ''' 

    fvc += EastBoundary*(-vc+fhat*us+np.divide(r,1-h)*vs+lamnda**(-2)*LyD*zetac)
    #                          )
         
    # ''' South Boundary '''
    fvc += SouthBoundary*(-vc+0)
    
    ''' North Boundary ''' 
    fvc += NorthBoundary*(-vc+0)
    
    # ''' quick corner fix'''
    
    fvc += NWCorner*( -vc+fhat*us+np.divide(r,1-h)*vs+lamnda**(-2)*LyD*zetac)
    
    fvc += SWCorner*( -vc+fhat*us+np.divide(r,1-h)*vs+lamnda**(-2)*LyD*zetac)
    
    fvc += NECorner*(-vc+fhat*us+np.divide(r,1-h)*vs+lamnda**(-2)*LyD*zetac)
    
    fvc += SECorner*(-vc+fhat*us+np.divide(r,1-h)*vs+lamnda**(-2)*LyD*zetac)
    return fvc


def F(U):
    zetas,zetac,us,uc,vs,vc=np.array_split(U,6)
    
    ans=np.concatenate((Fzetas(zetas,zetac,us,uc,vs,vc),
                        Fzetac(zetas,zetac,us,uc,vs,vc),
                           Fus(zetas,zetac,us,uc,vs,vc),
                           Fuc(zetas,zetac,us,uc,vs,vc),
                           Fvs(zetas,zetac,us,uc,vs,vc),
                           Fvc(zetas,zetac,us,uc,vs,vc)))

    return ans




def NumericalJacobian(U):
    zetas,zetac,us,uc,vs,vc=np.array_split(U,6)
    
    # J11=sp.csr_matrix(I.shape);J12=sp.csr_matrix(I.shape);J13=sp.csr_matrix(I.shape);J14=sp.csr_matrix(I.shape);J15=sp.csr_matrix(I.shape);J16=sp.csr_matrix(I.shape);
    # J21=sp.csr_matrix(I.shape);J22=sp.csr_matrix(I.shape);J23=sp.csr_matrix(I.shape);J24=sp.csr_matrix(I.shape);J25=sp.csr_matrix(I.shape);J26=sp.csr_matrix(I.shape);
    # J31=sp.csr_matrix(I.shape);J32=sp.csr_matrix(I.shape);J33=sp.csr_matrix(I.shape);J34=sp.csr_matrix(I.shape);J35=sp.csr_matrix(I.shape);J36=sp.csr_matrix(I.shape);
    # J41=sp.csr_matrix(I.shape);J42=sp.csr_matrix(I.shape);J43=sp.csr_matrix(I.shape);J44=sp.csr_matrix(I.shape);J45=sp.csr_matrix(I.shape);J46=sp.csr_matrix(I.shape);
    # J51=sp.csr_matrix(I.shape);J52=sp.csr_matrix(I.shape);J53=sp.csr_matrix(I.shape);J54=sp.csr_matrix(I.shape);J55=sp.csr_matrix(I.shape);J56=sp.csr_matrix(I.shape);
    # J61=sp.csr_matrix(I.shape);J62=sp.csr_matrix(I.shape);J63=sp.csr_matrix(I.shape);J64=sp.csr_matrix(I.shape);J65=sp.csr_matrix(I.shape);J66=sp.csr_matrix(I.shape);



    J11=np.zeros(I.shape);J12=np.zeros(I.shape);J13=np.zeros(I.shape);J14=np.zeros(I.shape);J15=np.zeros(I.shape);J16=np.zeros(I.shape);
    J21=np.zeros(I.shape);J22=np.zeros(I.shape);J23=np.zeros(I.shape);J24=np.zeros(I.shape);J25=np.zeros(I.shape);J26=np.zeros(I.shape);
    J31=np.zeros(I.shape);J32=np.zeros(I.shape);J33=np.zeros(I.shape);J34=np.zeros(I.shape);J35=np.zeros(I.shape);J36=np.zeros(I.shape);
    J41=np.zeros(I.shape);J42=np.zeros(I.shape);J43=np.zeros(I.shape);J44=np.zeros(I.shape);J45=np.zeros(I.shape);J46=np.zeros(I.shape);
    J51=np.zeros(I.shape);J52=np.zeros(I.shape);J53=np.zeros(I.shape);J54=np.zeros(I.shape);J55=np.zeros(I.shape);J56=np.zeros(I.shape);
    J61=np.zeros(I.shape);J62=np.zeros(I.shape);J63=np.zeros(I.shape);J64=np.zeros(I.shape);J65=np.zeros(I.shape);J66=np.zeros(I.shape);
    
    for i in tqdm(range(0,(Nx+1)*(Ny+1))):
        h_small=1e-8*I.toarray()[:,i]
        
        for NJ_func in {Fzetas,Fzetac,Fus,Fuc,Fvs,Fvc}:
            if NJ_func == Fzetas:
                J1=J11;     J2=J12;     J3=J13;     J4=J14;     J5=J15;     J6=J16;
            if NJ_func == Fzetac:
                J1=J21;     J2=J22;     J3=J23;     J4=J24;     J5=J25;     J6=J26;
            if NJ_func == Fus:
                J1=J31;     J2=J32;     J3=J33;     J4=J34;     J5=J35;     J6=J36;
            if NJ_func == Fuc:
                J1=J41;     J2=J42;     J3=J43;     J4=J44;     J5=J45;     J6=J46;
            if NJ_func == Fvs:
                J1=J51;     J2=J52;     J3=J53;     J4=J54;     J5=J55;     J6=J56; 
            if NJ_func == Fvc:
                J1=J61;     J2=J62;     J3=J63;     J4=J64;     J5=J65;     J6=J66;

                
            # J1[:,i] = np.array([(NJ_func(zetas+h_small,zetac,us,uc,vs,vc)-NJ_func(zetas-h_small,zetac,us,uc,vs,vc))/(2*np.linalg.norm(h_small))]).T
            # J2[:,i] = np.array([(NJ_func(zetas,zetac+h_small,us,uc,vs,vc)-NJ_func(zetas,zetac-h_small,us,uc,vs,vc))/(2*np.linalg.norm(h_small))]).T
            # J3[:,i] = np.array([(NJ_func(zetas,zetac,us+h_small,uc,vs,vc)-NJ_func(zetas,zetac,us-h_small,uc,vs,vc))/(2*np.linalg.norm(h_small))]).T
            # J4[:,i] = np.array([(NJ_func(zetas,zetac,us,uc+h_small,vs,vc)-NJ_func(zetas,zetac,us,uc-h_small,vs,vc))/(2*np.linalg.norm(h_small))]).T
            # J5[:,i] = np.array([(NJ_func(zetas,zetac,us,uc,vs+h_small,vc)-NJ_func(zetas,zetac,us,uc,vs-h_small,vc))/(2*np.linalg.norm(h_small))]).T
            # J6[:,i] = np.array([(NJ_func(zetas,zetac,us,uc,vs,vc+h_small)-NJ_func(zetas,zetac,us,uc,vs,vc-h_small))/(2*np.linalg.norm(h_small))]).T
            J1[:,i] = (NJ_func(zetas+h_small,zetac,us,uc,vs,vc)-NJ_func(zetas-h_small,zetac,us,uc,vs,vc))/(2*np.linalg.norm(h_small))
            J2[:,i] = (NJ_func(zetas,zetac+h_small,us,uc,vs,vc)-NJ_func(zetas,zetac-h_small,us,uc,vs,vc))/(2*np.linalg.norm(h_small))
            J3[:,i] = (NJ_func(zetas,zetac,us+h_small,uc,vs,vc)-NJ_func(zetas,zetac,us-h_small,uc,vs,vc))/(2*np.linalg.norm(h_small))
            J4[:,i] = (NJ_func(zetas,zetac,us,uc+h_small,vs,vc)-NJ_func(zetas,zetac,us,uc-h_small,vs,vc))/(2*np.linalg.norm(h_small))
            J5[:,i] = (NJ_func(zetas,zetac,us,uc,vs+h_small,vc)-NJ_func(zetas,zetac,us,uc,vs-h_small,vc))/(2*np.linalg.norm(h_small))
            J6[:,i] = (NJ_func(zetas,zetac,us,uc,vs,vc+h_small)-NJ_func(zetas,zetac,us,uc,vs,vc-h_small))/(2*np.linalg.norm(h_small))

         
    J=sp.bmat([
                [sp.csr_matrix(J11), sp.csr_matrix(J12), sp.csr_matrix(J13), sp.csr_matrix(J14), sp.csr_matrix(J15), sp.csr_matrix(J16)],
                [sp.csr_matrix(J21), sp.csr_matrix(J22), sp.csr_matrix(J23), sp.csr_matrix(J24), sp.csr_matrix(J25), sp.csr_matrix(J26)],
                [sp.csr_matrix(J31), sp.csr_matrix(J32), sp.csr_matrix(J33), sp.csr_matrix(J34), sp.csr_matrix(J35), sp.csr_matrix(J36)],
                [sp.csr_matrix(J41), sp.csr_matrix(J42), sp.csr_matrix(J43), sp.csr_matrix(J44), sp.csr_matrix(J45), sp.csr_matrix(J46)],
                [sp.csr_matrix(J51), sp.csr_matrix(J52), sp.csr_matrix(J53), sp.csr_matrix(J54), sp.csr_matrix(J55), sp.csr_matrix(J56)],
                [sp.csr_matrix(J61), sp.csr_matrix(J62), sp.csr_matrix(J63), sp.csr_matrix(J64), sp.csr_matrix(J65), sp.csr_matrix(J66)]


                ],format='csr')
    return J



 
NJ=NumericalJacobian(Uinnitalguess)

if False:
    plt.figure();
    plt.title('Numericla Jacobian')
    for j in range(1,6):
        if False:
            for x in np.where(NorthBoundary == 1)[0]: plt.axvline((j-1)*(Nx+1)*(Ny+1)+x,color='#4dff4d',linewidth=2)
            for y in np.where(NorthBoundary == 1)[0]: plt.axhline((j-1)*(Nx+1)*(Ny+1)+y,color='#4dff4d',linewidth=2)
            for x in np.where(EastBoundary == 1)[0]: plt.axvline((j-1)*(Nx+1)*(Ny+1)+x,color='#ff4d4d',linewidth=2)
            for y in np.where(EastBoundary == 1)[0]: plt.axhline((j-1)*(Nx+1)*(Ny+1)+y,color='#ff4d4d',linewidth=2)
            for x in np.where(SouthBoundary == 1)[0]: plt.axvline((j-1)*(Nx+1)*(Ny+1)+x,color='#ff4da6',linewidth=2)
            for y in np.where(SouthBoundary == 1)[0]: plt.axhline((j-1)*(Nx+1)*(Ny+1)+y,color='#ff4da6',linewidth=2)
            for x in np.where(WestBoundary == 1)[0]: plt.axvline((j-1)*(Nx+1)*(Ny+1)+x,color='#4dffff',linewidth=2)
            for y in np.where(WestBoundary == 1)[0]: plt.axhline((j-1)*(Nx+1)*(Ny+1)+y,color='#4dffff',linewidth=2)
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color='#4dff4d', lw=2, label='Noth Bounary'),
                               Line2D([0], [0], color='#ff4d4d', lw=2, label='East Bounary'),
                               Line2D([0], [0], color='#ff4da6', lw=2, label='South Bounary'),
                               Line2D([0], [0], color='#4dffff', lw=2, label='West Bounary')]
            plt.legend(handles=legend_elements,loc='upper left')
        plt.axvline(x=j*(Nx+1)*(Ny+1),color='k')
        plt.axhline(y=j*(Nx+1)*(Ny+1),color='k')
    
    plt.spy(NJ)





def NewtonRapsonInnerloop(Uinnitalguess:'np.ndarray'):
    epsilon=10**(-6)
    
    Uiend=np.copy(Uinnitalguess)
    
    i=0
    
    DeltaU=la.spsolve(NJ,F(Uiend))
    Uiend=Uiend-DeltaU

    print('\t Newton Rapson loop \n i=0 \t ||F(U)|| = %f < %f \n ' %(MaxNormOfU(F(Uiend)),epsilon))
    
    if MaxNormOfU(F(Uiend))>epsilon:
        Stopcondition=1
        
    else:
        Stopcondition=0
        
    
    while Stopcondition==1:
         
         i+=1
         Check=MaxNormOfU(F(Uiend))
         if Check>10**(20)*epsilon : # this is the fail save if the explodes.
             print('\n \t -----Divergence----- \n')
             break 
         if Check>epsilon and Check<10**(20)*epsilon:
             
             DeltaU=la.spsolve(NumericalJacobian(Uiend),F(Uiend))
             Uiend=Uiend-DeltaU
             Stopcondition=1
             print('\t Newton Rapson loop \n i=%i \t ||F(U)|| = %f < %f\n' %(i,MaxNormOfU(F(Uiend)),epsilon))

         if Check<=epsilon:
             print('\t Newton Rapson loop stoped at \n i=%i \t ||F(U)|| = %f < %f\n' %(i,MaxNormOfU(F(Uiend)),epsilon))
             print ('\n')
             Stopcondition=0
             break
             
             
         if i>20:
            break
    return Uiend    

# the run
    
Ufinal=NewtonRapsonInnerloop(Uinnitalguess)
zetas,zetac,us,uc,vs,vc=np.array_split(Ufinal,6)

Check=np.where(F(Ufinal)!=0)
if Check[0].size>0 :
    print('\t ------- F=!0 ------- \t \n')
    F_zetas,F_zetac,F_us,F_uc,F_vs,F_vc=np.array_split(F(Ufinal),6)
    
    fig, ((ax1,ax3,ax5),(ax2,ax4,ax6)) = plt.subplots(2,3)
    plt.gca().invert_yaxis()
    imgzetas = ax1.imshow(reshape(F_zetas))
    ax1.title.set_text('$F(\zeta^{s})$')
    plt.colorbar(imgzetas,orientation='horizontal',ax=ax1)
    
    imgzetac = ax2.imshow(reshape(F_zetac))
    plt.gca().invert_yaxis()
    ax2.title.set_text('$F(\zeta^{c})$')
    plt.colorbar(imgzetac,orientation='horizontal',ax=ax2)
    
    imgus = ax3.imshow(reshape(F_us))
    plt.gca().invert_yaxis()
    ax3.title.set_text('$F(u^{s})$')
    plt.colorbar(imgus,orientation='horizontal',ax=ax3)
    
    imguc = ax4.imshow(reshape(F_uc))
    plt.gca().invert_yaxis()
    ax4.title.set_text('$F(u^{c})$')
    plt.colorbar(imguc,orientation='horizontal',ax=ax4)
    
    imgvs=ax5.imshow(reshape(F_vs))
    plt.gca().invert_yaxis()
    ax5.title.set_text('$F(vs)$')
    plt.colorbar(imgvs,orientation='horizontal',ax=ax5)
    
    imgvc=ax6.imshow(reshape(F_vc))
    plt.gca().invert_yaxis()
    ax6.title.set_text('$F(vc)$')
    plt.colorbar(imgvc,orientation='horizontal',ax=ax6)
    


''' the plots '''
def staticplots():
        plt.ion()
        
        fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3)
        

        # initialization of the movie figure
        zetasarr = np.reshape(zetas,[Ny+1,Nx+1])
        zetacarr = np.reshape(zetac,[Ny+1,Nx+1])
        usarr = np.reshape(us,[Ny+1,Nx+1])
        ucarr = np.reshape(uc,[Ny+1,Nx+1])
        vsarr = np.reshape(vs,[Ny+1,Nx+1])
        vcarr = np.reshape(vc,[Ny+1,Nx+1])
        
        
    
        
        imgzetas = ax1.imshow(zetasarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        ax1.title.set_text('zeta-sin')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgzetas,orientation='horizontal',ax=ax1)
        
        imgzetac = ax4.imshow(zetacarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        ax4.title.set_text('zeta-cos')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgzetac,orientation='horizontal',ax=ax4)
     
        imgus = ax2.imshow(usarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        ax2.title.set_text('u-sin')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgus,orientation='horizontal',ax=ax2)
        
        imguc = ax5.imshow(ucarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        ax5.title.set_text('u-cos')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imguc,orientation='horizontal',ax=ax5)
        
        imgvs = ax3.imshow(vsarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        ax3.title.set_text('v-sin')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgvs,orientation='horizontal',ax=ax3)
        
        imgvc = ax6.imshow(vcarr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        ax6.title.set_text('v-cos')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgvc,orientation='horizontal',ax=ax6)
     


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
        ax1.set_ylim([-Atilde,Atilde])
        
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
    zeta0arr = reshape(zeta0)
    u0arr    = reshape(u0)
    v0arr    = reshape(v0)
        
    X = np.linspace(0, 1, Nx+1)
    Y = np.linspace(0, 1, Ny+1)
    U, V =v0arr, u0arr
    
    imgzeta = ax1.imshow(zeta0arr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
    ax1.title.set_text('zeta')
    plt.gca().invert_yaxis()
    
    plt.colorbar(imgzeta,orientation='horizontal',ax=ax1)
    

    imgzetacross = ax12.plot(zeta0arr.mean(0),linewidth=1,color='k')
    ax12.set_ylim([-(Atilde+A),(Atilde+A)])  
    
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
        global t, Nx, Ny
        t = (frame+1)*anim_dt
                        
        zeta1=zetas*np.sin(2*np.pi*t)+zetac*np.cos(2*np.pi*t)
        u1=us*np.sin(2*np.pi*t)+uc*np.cos(2*np.pi*t)
        v1=vs*np.sin(2*np.pi*t)+vc*np.cos(2*np.pi*t)
            
        imgzetacross[0].set_ydata(reshape(zeta1).mean(0))       
        imgzeta.set_array(reshape(zeta1))
        imgzeta.set_clim(zeta1.min(),zeta1.max())
        
        ax2.clear()
        ax2.quiver(X, Y, reshape(u1), reshape(v1),reshape(np.sqrt(u1**2+v1**2)),pivot='mid',units='dots',headwidth=0.1,linewidth=0.1,headlength=0.1)
        #ax2.quiverkey(q, 0.1, 0.1, 0.01)
            
        tlt.set_text('t = %3.3f' %(t))
      
                                                
        return imgzeta,imgzetacross,ax2
        
        # figure animation
        
    return animation.FuncAnimation(fig , animate  , interval=50 , repeat=True)
    
def Animation3():
    
    t = 0
    
    plt.ion()
        
    fig, ((ax1)) = plt.subplots(1)
    # Inital conditoin 

    # initialization of the movie figure
    
    
    Carr = np.reshape( C,[Ny-1,Nx-1])
    
    imgC = ax1.imshow(Carr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
    ax1.title.set_text('Concentration')
    plt.gca().invert_yaxis()
    
    plt.colorbar(imgC,orientation='horizontal',ax=ax1)
    

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
                        
        C_anim1=C
            
                
        imgC.set_array(np.reshape(C_anim1,[Ny-1,Nx-1]))
        imgC.set_clim(C_anim1.min(),C_anim1.max())
        
        
            
        tlt.set_text('t = %3.3f' %(t))
      
                                                
        return imgC
        
        # figure animation
        
    anim = animation.FuncAnimation(fig , animate  , interval=50 , repeat=True)

staticplots()

plt.figure()

Animation1()


Check_1=np.where(1-h+zetas<0)

Check_2=np.where(1-h+zetac<0)

if Check_1[0].size>0 or Check_2[0].size>0 :
    print('\n ------- \t  NON physical water model! \t ---------\n')

Check=np.where(C<0)
if Check[0].size>0 :
    print('\n ------- \t  NON physical Concentration! \t ---------\n')
    


