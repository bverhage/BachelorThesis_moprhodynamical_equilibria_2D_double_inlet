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
    Lx=1*59*10**3 #m
    Ly=1*10**3 #m
    H1=12#m
    H2=6
    
    A=0.74*1/H1
    
    Atilde=0.84*1/H1
    
    phi=1/2*np.pi#41/180*np.pi
    
    g=9.81 #m/s^2
    
    CD=0.0025
    sigma=1.4*10**-4 #/s
    
    lamnda=sigma*Lx/(np.sqrt(g*H1))
    Nx=100+1#25#100
    Ny=int((Nx-1)*1/2)+1
    
    dx=1/Nx
    
    #dy=1/Ny
    a=6.222*10**(-2)
    r=-(8*CD*A*Lx)/(3*np.pi*H1**2)/(H1*sigma)
    #r=-0.025
    # beta=0.08
    
    fhat=7.1*10**-1
    
    k=2.052*10**(-4)
    Mutilde=10**(-6)
    epsilon = A/H1
    
    BOOL_concentration = False
    BOOL_bedevolution  = False
    BOOL_two_open_ends = False
    
    BOOL_print_NR= False
    
    def deltafunc(x):
        if  np.abs(x)<=1/(2*dx): return 1
        else: return 0
    
    
    def bedprofile(x):
         # if x>0.1:
         #     if (y-0.5)**2<0.4**2: return 0
         #     else: return 0.7
         # else: return 0
         #return 0.95/(1+np.exp(-100*(x-0.2)))-bedfunction2(x,y)*(1-1/(1+np.exp(-100*(x-0.2))))
         #return 0.8/(1+np.exp(-100*(x-0.2)))-bedfunction2(x,y)*(1/(1+np.exp(-100*(x-0.2))))/(30)
         return 0*(1-H2/H1)*x
    
    def func0(x):
        return 0
    
    def westboundary(x):
        if x==0:
            return 1

        else: return 0
        
    def nearwestboundary(x):
        if x==dx:
            return 1

        else: return 0

    
    def eastboundary(x):
        if x==1: return 1
        return 0
    
    
    def neareastboundary(x):
        if x==1-dx: return 1
        return 0
    
        
    def create(Nx:int,func):
        Fvec=np.zeros((Nx+1))
       
        for i in range(Nx+1):
                x=dx*(i)
                
                Fvec[i]=func(x)
        return Fvec
    


    
    ICzeta0 = create(Nx,func0)
    ICu0    = create(Nx,func0)
    ICv0    = create(Nx,func0)
    ICh0    = create(Nx,bedprofile)


    WestBoundary = create(Nx,westboundary)
    NearWestBounary = create(Nx,nearwestboundary)
    EastBoundary = create(Nx,eastboundary)
    NearEastBounary = create(Nx,neareastboundary)
    
    Interior = np.ones(ICzeta0.shape)-WestBoundary-EastBoundary
    
    Check=np.where(ICh0>1)
    if Check[0].size>0 :
        print('\n ------- \t  BED violates the model! \t ---------\n')
    else:
        print('\n Boundary checked commensing Newton Rapsons algorithem')

if True:
    
   def FDLaplacian2D(Nx,dx):
        
        #diagonalsx = [-1*np.ones((1,Nx-1-1)),np.zeros((1,Nx-1)),np.ones((1,Nx-1-1))]
        #diagonalsy = [-1*np.ones((1,Ny-1-1)),np.zeros((1,Ny-1)),np.ones((1,Ny-1-1))]
        diagonalsx = [-1*np.ones((1,Nx+2)),np.ones((1,Nx+1))]
        
        #diagonalsx[1][0][0]=0;
        #diagonalsx[0][0][0]=0;
        
        #diagonalsx[1][0][-1]=0;
        #diagonalsx[0][0][-2]=0;
        
        Dx =1/(dx)*sp.diags(diagonalsx, [-1, 0], shape=(Nx+2,Nx+1))

        LXX=Dx.transpose().dot(Dx)
   

        Ans = LXX
        return Ans

   def LX(Nx,dx):
        
        diagonalsx = [-1*np.ones((1,Nx)),np.ones((1,Nx))]


        Dx =1/(2*dx)*sp.diags(diagonalsx, [-1 ,1], shape=(Nx+1,Nx+1))
        
        Ans =Dx
        #Ans = sp.kron(Dx,sp.eye(Ny-1))
        return Ans

       
    
    
        
        
 
# 2D FD Laplacian and identity matrices

LxD  =  LX(Nx,dx)
#LxD  =  LX(Nx,Ny,dx,dy,True,False) #WRONG
 

#LxN  =  LX(Nx,Ny,dx,dy,False,True) #WRONG



A_x = FDLaplacian2D(Nx,dx)


I = sp.eye((Nx+1))

# LxD = sp.bmat([
#     [0,np.zeros((1,LxD.shape[0])),0],
#     [np.zeros((LxD.shape[0],1)),LxD,np.zeros((LxD.shape[0],1))],
#     [0,np.zeros((1,LxD.shape[0])),0]],
#     format='csr')
#ICzetas,ICzetac,ICus,ICuc,ICvs,ICvc=np.array_split(np.loadtxt('test1.txt', dtype=int),6) 



Uinnitalguess=np.concatenate((ICzeta0,ICzeta0,ICu0,ICu0,ICv0,ICh0))

# #part 2 the Newton rapson method

# # first we need to define some functions:
    
def MaxNormOfU(U):
     ''' The max 2 norm of U=(u,v)^T w'''
     zetas,zetac,us,uc,C,h=np.array_split(U,6)
     return np.max([
         np.linalg.norm(zetas),
         np.linalg.norm(zetac),
         np.linalg.norm(us),
         np.linalg.norm(uc),
         np.linalg.norm(C),
         np.linalg.norm(h)
         ])
 
lamndad=1.8
delta_s=9.5*10**(-4)

def beta(h):
    return 1/(1-np.exp(-lamndad*(1-h)))

def Fzetas(zetas,zetac,us,uc,C,h):
    ''' interior '''
    
    fzetas =-zetas-(1-h)*(LxD*uc)+uc*(LxD*h)
    
    fzetas = Interior*fzetas
    
    
    
    ''' west boundary '''
    fzetas += WestBoundary*(
                            -zetas + 0
                            #-(1-0)*1/dx*(-uc.dot(WestBoundary)+uc.dot(NearWestBounary))
                            #+uc.dot(WestBoundary)*1/dx*(-0+h.dot(NearWestBounary))
                            )
    
    ''' east boundary '''
    if BOOL_two_open_ends:
        fzetas += EastBoundary*(
                                -zetas + Atilde*np.sin(phi)
                                #-(1-1+H2/H1)*1/dx*(-uc.dot(NearEastBounary)+uc.dot(EastBoundary))
                                #+uc.dot(EastBoundary)*1/dx*(-h.dot(NearEastBounary)+1-H2/H1)
                                )
    else:
        fzetas += EastBoundary*(
                                1/dx*(sp.kron(np.array([EastBoundary]).T,-NearEastBounary+EastBoundary)*zetas) 
                                -(1-1+H2/H1)*1/dx*(-uc.dot(NearEastBounary)+0 )
                                +   0*1/dx*(-h.dot(NearEastBounary)+1-H2/H1)
                                )
    return fzetas 

def Fzetac(zetas,zetac,us,uc,C,h):
    ''' interior '''
    fzetac =-zetac+(1-h)*(LxD*us)-us*(LxD*h)
    
    fzetac = Interior*fzetac
    
    ''' west boundary '''
    fzetac += WestBoundary*(
                            -zetac+A
                            #+(1-0)*1/dx*(-us.dot(WestBoundary)+us.dot(NearWestBounary))
                            #-us.dot(WestBoundary)*1/dx*(-0+h.dot(NearWestBounary))
                            )
    ''' east boundary ''' 
    if BOOL_two_open_ends:
        fzetac += EastBoundary*(
                                -zetac + Atilde*np.cos(phi)
                                #+(1-1+H2/H1)*1/dx*(-us.dot(NearEastBounary)+us.dot(EastBoundary))
                                #-us.dot(EastBoundary)*1/dx*(-h.dot(NearEastBounary)+1-H2/H1)
            )
    else:
        fzetac += EastBoundary*(
                                1/dx*(-zetac.dot(NearEastBounary)+zetac.dot(EastBoundary)) 
                                +(1-1+H2/H1)*1/dx*(-us.dot(NearEastBounary)+0)
                                -0*1/dx*(-h.dot(NearEastBounary)+1-H2/H1)
                                )
            
    return fzetac 


def Fus(zetas,zetac,us,uc,C,h):
    ''' interior '''
    fus   = -us -np.divide(r, 1-h)*uc-lamnda**(-2)*LxD*zetas
    
    fus = Interior*fus
    
    ''' west boundary '''
    ' KNOWN : zetas = 0 zetac = A  '
    ' UNKOWN : us, uc '
    fus   +=  WestBoundary*(
                            -us
                            -np.divide(r, 1-h)*uc
                            -lamnda**(-2)*1/(dx)*(-0 +zetas.dot(NearWestBounary))
                            )
    ''' east boundary ''' 
    if BOOL_two_open_ends:
         ' KNOWN : zetas = 0 zetac = A  '
         ' UNKOWN : us, uc '
         fus += EastBoundary*(
                             -us#-(Atilde*np.cos(phi)+uc.dot(NearEastBounary))/(1+h.dot(NearEastBounary)-1+H2/H1)
                             -np.divide(r, 1-h)*uc
                            
                             -lamnda**(-2)*1/dx*(-zetas.dot(NearEastBounary) +Atilde*np.sin(phi)) #wrong
                             )
    else:
         ' KNOWN :  d/dx zetas=0  d/dx zetac = 0 , us=0 , uc = 0 '
         ' UNKOWN : zetas, zetac '
         fus += EastBoundary*(
                             -us+0
                             -lamnda**(-2)*1/dx*(-zetas.dot(NearEastBounary)+zetas.dot(EastBoundary))
                             )
    
    return fus
         
def Fuc(zetas,zetac,us,uc,C,h):
    ''' interior '''
    fuc   = -uc +np.divide(r, 1-h)*us+lamnda**(-2)*LxD*zetac
    
    fuc = Interior*fuc 
    
    ''' west boundary '''
    ' KNOWN : zetas = 0 zetac = A  '
    ' UNKOWN : us, uc '
    fuc     +=  WestBoundary*(
                                -uc
                                +np.divide(r, 1-h)*us
                                
                                
                                +lamnda**(-2)*1/dx*( -A +zetac.dot(NearWestBounary))
                                )
                            
    ''' east boundary '''

    if BOOL_two_open_ends:
        ' KNOWN  : zetas = Atilde*sin zetac = Atilde*cos '
        ' UNKOWN : us , uc '
        fuc += EastBoundary*(
                                -uc#-(Atilde*np.sin(phi)*dx-uc.dot(NearEastBounary))/(-1+h.dot(NearEastBounary)-1+H2/H1)
                                +np.divide(r, 1-h)*us
                                +lamnda**(-2)*1/dx*(-zetac.dot(NearEastBounary)  + Atilde*np.cos(phi))
                                
                                )
    else:
        fuc += EastBoundary*(
                            -uc+0
                            +lamnda**(-2)*1/dx*(-zetac.dot(NearEastBounary)+zetac.dot(EastBoundary))
                            )
    return fuc


def FC(zetas,zetac,us,uc,C,h):
    ''' interior '''
    fC =-epsilon*beta(h)*C +epsilon*a*k*A_x*C+(us*us+uc*uc)+epsilon*a*k*lamndad*(beta(h)*LxD*h*LxD*C+C*(beta(h)*A_x*h+LxD*beta(h)*LxD*h))
    fC = Interior*fC
    
    ''' west boundary '''
    ' zetas = 0 zetac = A  us = ? ,  uc = ? '
    fC += WestBoundary*(-C+0 +0*epsilon*a*k*dx**(-2)*(-2*C.dot(WestBoundary)+C.dot(NearWestBounary))) 
    #fC += WestBoundary*(
    #      epsilon*a*k*lamndad*(beta(h)
    #                           *1/dx*(-h.dot(WestBoundary)+h.dot(NearWestBounary))
    #                           *1/dx*(-C.dot(WestBoundary)+C.dot(NearWestBounary))
    #                         +0*(
    #                             beta(h)*dx**(-2)*h
    #                             +1/dx*(-beta(h).dot(WestBoundary)+beta(h).dot(NearWestBounary))
    #                                  *1/dx*(-h.dot(WestBoundary)+h.dot(NearEastBounary))
    #                            )
    #                         ))
    ''' east boundary '''
    if BOOL_two_open_ends:
        
        fC += EastBoundary*(-C+0+0*epsilon*a*k*dx**(-2)*(C.dot(NearEastBounary)-2*C.dot(EastBoundary)))
    else:
        fC +=EastBoundary*1/dx*(-C.dot(NearEastBounary)+C.dot(EastBoundary))
     #   fC += EastBoundary*(
     #       epsilon*a*k*lamndad*(beta(h)
     #                          *1/dx*(-h.dot(NearEastBounary)+h.dot(EastBoundary))
     #                          *1/dx*(-C.dot(NearEastBounary)+C.dot(EastBoundary))
     #                        +0*(
     #                            
     #                            +1/dx*(-beta(h).dot(NearEastBounary)+beta(h).dot(EastBoundary))
     #                                 *1/dx*(-h.dot(NearEastBounary)+h.dot(EastBoundary))
     #                           )
     #                        ))
    return fC

def Fh(zetas,zetac,us,uc,C,h):
    
    ''' interior '''
    
    fh=Mutilde*A_x*h#-delta_s*(-epsilon*beta(h)*C+(us*us+uc*uc))#epsilon*a*k*A_x*C+epsilon*a*k*lamndad*(beta(h)*LxD*h*LxD*C+C*(beta(h)*A_x*h+LxD*beta(h)*LxD*h)))
    fh = Interior*fh
    
    ''' west boundary '''
    fh += WestBoundary*(-h+0)
    
    ''' east boundary '''
    fh += EastBoundary*(-h+(1-H2/H1))
    
    return fh
    
def F(U):
    zetas,zetac,us,uc,C,h=np.array_split(U,6)
    
    ans=np.concatenate((Fzetas(zetas,zetac,us,uc,C,h),
                        Fzetac(zetas,zetac,us,uc,C,h),
                        Fus(zetas,zetac,us,uc,C,h),
                        Fuc(zetas,zetac,us,uc,C,h),
                        FC(zetas,zetac,us,uc,C,h),
                        Fh(zetas,zetac,us,uc,C,h)))
    
    # h_east=h*EastBoundary
    # zeros=np.zeros(zetas.shape)
    
    # A_h=ADNNN
    
    # # -------------- zetas ------------------
    # fzetas =-zetas-(1-h)*(LxD*uc)+uc*(LxD*h)
    
    
    # # -------------- zetac ------------------
    # fzetac =-zetac+(1-h)*(LxD*us)-us*(LxD*h)
    
    
    # # -------------- us ------------------
    # fus    = -us -np.divide(r, 1-h)*uc-lamnda**(-2)*LxD*zetas
    
    # #fus    =+ -lamnda**(-2)*Atilde*np.cos(phi)*EastBoundary/(2*dx)
    
    # # -------------- uc ------------------
    # fuc    = -uc +np.divide(r, 1-h)*us+lamnda**(-2)*LxD*zetac
    

    # # -------------- C ------------------
    
    # fC     = zeros#-epsilon*beta(h)*C +epsilon*a*k*A_h*C+epsilon*a*k*lamndad*(beta(h)*Lx_zeta*h*Lx_zeta*C+C*(beta(h)*A_h*h+Lx_zeta*beta(h)*Lx_zeta*h))
    # if BOOL_concentration:
    #  fC = +(us*us+uc*uc)
    
    # # -------------- h ------------------
    # fh     =  zeros #Mutilde*A_h*h
    
    # if BOOL_bedevolution:
    #     #fh     =+ epsilon*delta_s*a*(k*A_h*C+lamndad*(beta(h)*Lx_zeta*h*Lx_zeta*C+C*(Lx_zeta*beta(h)*Lx_zeta*h+beta(h)*A_h*h)))
    #     fh     =+ +epsilon*beta(h)*C-(us*us+uc*uc)
       
    # ans=np.concatenate((fzetas,fzetac,fus,fuc,fC,fh))
    
    
    # ans=+ np.concatenate((
    #     WestBoundary*(  0 -1/dx*(-uc.dot(WestBoundary)+uc.dot(NearWestBounary))+uc.dot(WestBoundary)*1/dx*(0-h.dot(NearWestBounary))),
    #     WestBoundary*( -A +1/dx*(-us.dot(WestBoundary)+us.dot(NearWestBounary))-us.dot(WestBoundary)*1/dx*(0-h.dot(NearWestBounary))),
    #     WestBoundary*-lamnda**(-2)*1/(dx)*(  0 + zetac.dot(NearWestBounary)),
    #     WestBoundary* lamnda**(-2)*1/(dx)*( -A + zetac.dot(NearWestBounary)),
    #     zeros,
    #     zeros
    #     ))
    # if not BOOL_two_open_ends:
    #     ans=+ np.concatenate((
    #         +EastBoundary*zetas.dot(EastBoundary),
    #         +EastBoundary*zetac.dot(EastBoundary),
    #         EastBoundary*-lamnda**(-2)*1/dx*(-zetas.dot(NearEastBounary)+zetas.dot(EastBoundary)),
    #         EastBoundary*+lamnda**(-2)*1/dx*(-zetac.dot(NearEastBounary)+zetac.dot(EastBoundary)),
    #         zeros,
    #         zeros
    #         ))
    
    # if BOOL_two_open_ends:
    #     ans=+ np.concatenate((
    #         #zetas
    #         #+lamnda**(-2)*Atilde*np.cos(phi)*EastBoundary/(2*dx)*(1-H2/H1)*EastBoundary/(2*dx)
            
            
    #         #+uc*EastBoundary/(2*dx)*(1-H2/H1)
            
    #          (1-h)*EastBoundary*Atilde*np.sin(phi)*H1/(2*H2)+uc*EastBoundary/(2*dx)*(1-H2/H1-(1-h)*H1/H2*(1-H2/H1)),                      
    #         #+EastBoundary*-(1-h_east)*H1/H2*(-Atilde*np.sin(phi)+uc*(-h_east+1-H2/H1)/dx)/(1-H1/H2*dx*(-h_east+1-H2/H1)/dx)/2,
            
    #         #+EastBoundary*-(1-h_east)*lamnda**2*((ADNDN*zetas-Atilde*np.sin(phi))/(dx)-r/(1-1+H2/H1)*(zetac-Atilde*np.cos(phi)))/(1+(r/(1-1+H2/H1)**2)),
    #         #zetac
    #         #-us*(1-H2/H1)*EastBoundary*H/(2*dx)
            
    #         (1-h)*EastBoundary*Atilde*np.cos(phi)*H1/(2*H2)-us*EastBoundary/(2*dx)*(1-H2/H1-(1-h)*H1/H2*(1-H2/H1)),
    #         #EastBoundary*(1-h_east)*1/2*H1/H2*(Atilde*np.cos(phi)+us*(-h_east+1-H2/H1)/(dx))/(1-dx*H1/H2*(-h_east+1-H2/H1)/(dx))
    #         #+EastBoundary*(1-h_east)*-lamnda**2*((zetas-Atilde*np.sin(phi))/(dx)+r/(1-1+H2/H1)*(zetac-Atilde*np.cos(phi)))/(1+(r/(1-1+H2/H1))**2)
    #         #-us*EastBoundary*1/(2*dx)*(1-H2/H1),
            
    #         #us
    #         -lamnda**(-2)*Atilde*np.sin(phi)*EastBoundary/(2*dx),
    #         #uc
    #         +lamnda**(-2)*Atilde*np.cos(phi)*EastBoundary/(2*dx),
    #         #C
    #         EastBoundary*-epsilon*a*k*lamndad*( beta(h)*(1/(2*dx)*(1-H2/H1)*Lx_zeta*C +C*( beta(h)*1/(2*dx)*(1-H2/H1)+ beta(1-H2/H1)*(1-H2/H1)/(4*dx**2) ))),#a*k*lamndad*beta(h)*((1-H2/H1)/(2*dx)*C+(1-H2/H1)/(2*dx)*Lx_zeta*C)*EastBoundary,
    #         #h
    #         EastBoundary*(Mutilde*(1-H2/H1)/(2*dx))#+epsilon*delta_s*a*lamndad*C*(beta(h)*(1-H2/H1)/(2*dx)+beta(1-H2/H1)*(1-H2/H1)/(4*dx**2)))
    #         ))

    return ans




def NumericalJacobian(U):
    zetas,zetac,us,uc,C,h=np.array_split(U,6)
    
    J11=sp.csr_matrix(I.shape);J12=sp.csr_matrix(I.shape);J13=sp.csr_matrix(I.shape);J14=sp.csr_matrix(I.shape);J17=sp.csr_matrix(I.shape);J18=sp.csr_matrix(I.shape);
    J21=sp.csr_matrix(I.shape);J22=sp.csr_matrix(I.shape);J23=sp.csr_matrix(I.shape);J24=sp.csr_matrix(I.shape);J27=sp.csr_matrix(I.shape);J28=sp.csr_matrix(I.shape);
    J31=sp.csr_matrix(I.shape);J32=sp.csr_matrix(I.shape);J33=sp.csr_matrix(I.shape);J34=sp.csr_matrix(I.shape);J37=sp.csr_matrix(I.shape);J38=sp.csr_matrix(I.shape);
    J41=sp.csr_matrix(I.shape);J42=sp.csr_matrix(I.shape);J43=sp.csr_matrix(I.shape);J44=sp.csr_matrix(I.shape);J47=sp.csr_matrix(I.shape);J48=sp.csr_matrix(I.shape);
    J71=sp.csr_matrix(I.shape);J72=sp.csr_matrix(I.shape);J73=sp.csr_matrix(I.shape);J74=sp.csr_matrix(I.shape);J77=sp.csr_matrix(I.shape);J78=sp.csr_matrix(I.shape);
    J81=sp.csr_matrix(I.shape);J82=sp.csr_matrix(I.shape);J83=sp.csr_matrix(I.shape);J84=sp.csr_matrix(I.shape);J87=sp.csr_matrix(I.shape);J88=sp.csr_matrix(I.shape);
    
    for i in range(0,Nx+1):
        h_small=1e-8*I.toarray()[:,i]
        
        NJ_func=Fzetas
        J11[:,i] = np.array([(NJ_func(zetas+h_small,zetac,us,uc,C,h)-NJ_func(zetas-h_small,zetac,us,uc,C,h))/(2*np.linalg.norm(h_small))]).T
        J12[:,i] = np.array([(NJ_func(zetas,zetac+h_small,us,uc,C,h)-NJ_func(zetas,zetac-h_small,us,uc,C,h))/(2*np.linalg.norm(h_small))]).T
        J13[:,i] = np.array([(NJ_func(zetas,zetac,us+h_small,uc,C,h)-NJ_func(zetas,zetac,us-h_small,uc,C,h))/(2*np.linalg.norm(h_small))]).T
        J14[:,i] = np.array([(NJ_func(zetas,zetac,us,uc+h_small,C,h)-NJ_func(zetas,zetac,us,uc-h_small,C,h))/(2*np.linalg.norm(h_small))]).T
         
        J17[:,i] = np.array([(NJ_func(zetas,zetac,us,uc,C+h_small,h)-NJ_func(zetas,zetac,us,uc,C-h_small,h))/(2*np.linalg.norm(h_small))]).T
        J18[:,i] = np.array([(NJ_func(zetas,zetac,us,uc,C,h+h_small)-NJ_func(zetas,zetac,us,uc,C,h-h_small))/(2*np.linalg.norm(h_small))]).T
         
        NJ_func=Fzetac
        J21[:,i] = np.array([(NJ_func(zetas+h_small,zetac,us,uc,C,h)-NJ_func(zetas-h_small,zetac,us,uc,C,h))/(2*np.linalg.norm(h_small))]).T
        J22[:,i] = np.array([(NJ_func(zetas,zetac+h_small,us,uc,C,h)-NJ_func(zetas,zetac-h_small,us,uc,C,h))/(2*np.linalg.norm(h_small))]).T
        J23[:,i] = np.array([(NJ_func(zetas,zetac,us+h_small,uc,C,h)-NJ_func(zetas,zetac,us-h_small,uc,C,h))/(2*np.linalg.norm(h_small))]).T
        J24[:,i] = np.array([(NJ_func(zetas,zetac,us,uc+h_small,C,h)-NJ_func(zetas,zetac,us,uc-h_small,C,h))/(2*np.linalg.norm(h_small))]).T
         
        J27[:,i] = np.array([(NJ_func(zetas,zetac,us,uc,C+h_small,h)-NJ_func(zetas,zetac,us,uc,C-h_small,h))/(2*np.linalg.norm(h_small))]).T
        J28[:,i] = np.array([(NJ_func(zetas,zetac,us,uc,C,h+h_small)-NJ_func(zetas,zetac,us,uc,C,h-h_small))/(2*np.linalg.norm(h_small))]).T
    
        NJ_func=Fus
        J31[:,i] = np.array([(NJ_func(zetas+h_small,zetac,us,uc,C,h)-NJ_func(zetas-h_small,zetac,us,uc,C,h))/(2*np.linalg.norm(h_small))]).T
        J32[:,i] = np.array([(NJ_func(zetas,zetac+h_small,us,uc,C,h)-NJ_func(zetas,zetac-h_small,us,uc,C,h))/(2*np.linalg.norm(h_small))]).T
        J33[:,i] = np.array([(NJ_func(zetas,zetac,us+h_small,uc,C,h)-NJ_func(zetas,zetac,us-h_small,uc,C,h))/(2*np.linalg.norm(h_small))]).T
        J34[:,i] = np.array([(NJ_func(zetas,zetac,us,uc+h_small,C,h)-NJ_func(zetas,zetac,us,uc-h_small,C,h))/(2*np.linalg.norm(h_small))]).T
         
        J37[:,i] = np.array([(NJ_func(zetas,zetac,us,uc,C+h_small,h)-NJ_func(zetas,zetac,us,uc,C-h_small,h))/(2*np.linalg.norm(h_small))]).T
        J38[:,i] = np.array([(NJ_func(zetas,zetac,us,uc,C,h+h_small)-NJ_func(zetas,zetac,us,uc,C,h-h_small))/(2*np.linalg.norm(h_small))]).T
         
        NJ_func=Fuc
        J41[:,i] = np.array([(NJ_func(zetas+h_small,zetac,us,uc,C,h)-NJ_func(zetas-h_small,zetac,us,uc,C,h))/(2*np.linalg.norm(h_small))]).T
        J42[:,i] = np.array([(NJ_func(zetas,zetac+h_small,us,uc,C,h)-NJ_func(zetas,zetac-h_small,us,uc,C,h))/(2*np.linalg.norm(h_small))]).T
        J43[:,i] = np.array([(NJ_func(zetas,zetac,us+h_small,uc,C,h)-NJ_func(zetas,zetac,us-h_small,uc,C,h))/(2*np.linalg.norm(h_small))]).T
        J44[:,i] = np.array([(NJ_func(zetas,zetac,us,uc+h_small,C,h)-NJ_func(zetas,zetac,us,uc-h_small,C,h))/(2*np.linalg.norm(h_small))]).T
         
        J47[:,i] = np.array([(NJ_func(zetas,zetac,us,uc,C+h_small,h)-NJ_func(zetas,zetac,us,uc,C-h_small,h))/(2*np.linalg.norm(h_small))]).T
        J48[:,i] = np.array([(NJ_func(zetas,zetac,us,uc,C,h+h_small)-NJ_func(zetas,zetac,us,uc,C,h-h_small))/(2*np.linalg.norm(h_small))]).T
        
        NJ_func=FC
        J71[:,i] = np.array([(NJ_func(zetas+h_small,zetac,us,uc,C,h)-NJ_func(zetas-h_small,zetac,us,uc,C,h))/(2*np.linalg.norm(h_small))]).T
        J72[:,i] = np.array([(NJ_func(zetas,zetac+h_small,us,uc,C,h)-NJ_func(zetas,zetac-h_small,us,uc,C,h))/(2*np.linalg.norm(h_small))]).T
        J73[:,i] = np.array([(NJ_func(zetas,zetac,us+h_small,uc,C,h)-NJ_func(zetas,zetac,us-h_small,uc,C,h))/(2*np.linalg.norm(h_small))]).T
        J74[:,i] = np.array([(NJ_func(zetas,zetac,us,uc+h_small,C,h)-NJ_func(zetas,zetac,us,uc-h_small,C,h))/(2*np.linalg.norm(h_small))]).T
         
        J77[:,i] = np.array([(NJ_func(zetas,zetac,us,uc,C+h_small,h)-NJ_func(zetas,zetac,us,uc,C-h_small,h))/(2*np.linalg.norm(h_small))]).T
        J78[:,i] = np.array([(NJ_func(zetas,zetac,us,uc,C,h+h_small)-NJ_func(zetas,zetac,us,uc,C,h-h_small))/(2*np.linalg.norm(h_small))]).T
        
        NJ_func=Fh
        J81[:,i] = np.array([(NJ_func(zetas+h_small,zetac,us,uc,C,h)-NJ_func(zetas-h_small,zetac,us,uc,C,h))/(2*np.linalg.norm(h_small))]).T
        J82[:,i] = np.array([(NJ_func(zetas,zetac+h_small,us,uc,C,h)-NJ_func(zetas,zetac-h_small,us,uc,C,h))/(2*np.linalg.norm(h_small))]).T
        J83[:,i] = np.array([(NJ_func(zetas,zetac,us+h_small,uc,C,h)-NJ_func(zetas,zetac,us-h_small,uc,C,h))/(2*np.linalg.norm(h_small))]).T
        J84[:,i] = np.array([(NJ_func(zetas,zetac,us,uc+h_small,C,h)-NJ_func(zetas,zetac,us,uc-h_small,C,h))/(2*np.linalg.norm(h_small))]).T
         
        J87[:,i] = np.array([(NJ_func(zetas,zetac,us,uc,C+h_small,h)-NJ_func(zetas,zetac,us,uc,C-h_small,h))/(2*np.linalg.norm(h_small))]).T
        J88[:,i] = np.array([(NJ_func(zetas,zetac,us,uc,C,h+h_small)-NJ_func(zetas,zetac,us,uc,C,h-h_small))/(2*np.linalg.norm(h_small))]).T
    J=sp.bmat([
                [J11, J12, J13, J14, J17, J18],
                [J21, J22, J23, J24, J27, J28],
                [J31, J32, J33, J34, J37, J38],
                [J41, J42, J43, J44, J47, J48],
                [J71, J72, J73, J74, J77, J78],
                [J81, J82, J83, J84, J87, J88]
                ],format='csr')
    return J




def Jacobian(U):
     zetas,zetac,us,uc,C,h=np.array_split(U,6)
     zeros=sp.csr_matrix(LxD.shape)
     h_east=h*EastBoundary
     
     A_h=ADNNN
     
     J11 = -I                                      # zetas,zetas
     J12 = zeros                                   # zetas,zetac
     J13 = zeros                                   # zetas, us
     J14 = -LxD.multiply(1-h.T)+sp.diags(LxD*h)     # zetas, uc
     J15 = zeros                                   # zetas, vs
     
     J17 = zeros                                    # zetas, C
     J18 = sp.diags(LxD*uc)+LxD.multiply(uc.T)
     
     J21 = zeros
     J22 = -I
     J23 =  LxD.multiply(1-h.T)-sp.diags(LxD*h)
     J24 = zeros
    
     J26 = zeros
     J27 = zeros
     J28 = -sp.diags(LxD*us)-LxD.multiply(us.T)
     
     J31 = -lamnda**(-2)*LxD
     J32 = zeros
     J33 = -I
     J34 = -sp.diags(np.divide(r,1-h))
     J35 = zeros
     J36 = I*fhat
     J37 = zeros
     J38 = -sp.diags(r*(1-h)**(-2)*uc)
     
     J41= zeros
     J42= lamnda**(-2)*LxD
     J43= sp.diags(np.divide(r,1-h))
     J44= -I
     J45= -I*fhat
     J46=  zeros
     J47 = zeros
     J48 = sp.diags(r*(1-h)**(-2)*us)
          
     
     
     
     J71 = zeros
     J72 = zeros
     J73 = 2*sp.diags(us)
     J74 = 2*sp.diags(uc)
     J77 = epsilon*a*k*A_h#0*sp.diags(-epsilon*beta(h))+epsilon*a*k*lamndad*(Lx_zeta.multiply(beta(h)*Lx_zeta*h)+sp.diags(beta(h)*A_h*h+Lx_zeta*beta(h)*Lx_zeta*h))
     J78 = zeros
     
     J81 = zeros
     J82 = zeros
     J83 = zeros
     J84 = zeros
     J85 = zeros
     J86 = zeros
     J87 = sp.diags(epsilon*beta(h))#+epsilon*delta_s*a*k*A_h
     J88 = -Mutilde*A_h# + sp.diags(epsilon*C*lamndad*np.exp(-lamndad*(1-h))/(1-np.exp(-lamndad*(1-h)))**2)
     
     # BOUNDARY
     J11 = J11
     J12 = J12 
     J14 = J14 + sp.kron(np.array([WestBoundary]).T,-1/dx*(-WestBoundary+NearWestBounary)+1/dx*WestBoundary*-h.dot(NearWestBounary)) 
     J23 = J23 + sp.kron(np.array([WestBoundary]).T,+1/dx*(-WestBoundary+NearWestBounary)-1/dx*WestBoundary*-h.dot(NearWestBounary)) 

    
     if not BOOL_two_open_ends:
        
        J11= J11 + sp.kron(np.array([EastBoundary]).T,(-NearEastBounary+2*EastBoundary))
        J22= J22 + sp.kron(np.array([EastBoundary]).T,(-NearEastBounary+2*EastBoundary))
        J31= J31 + sp.kron(np.array([EastBoundary]).T,-lamnda**(-2)*1/dx*(-NearEastBounary+EastBoundary))
        J42= J42 + sp.kron(np.array([EastBoundary]).T,+lamnda**(-2)*1/dx*(-NearEastBounary+EastBoundary))
        

    
     if  not BOOL_bedevolution: 
         J87= zeros
         J78= zeros
         J68= zeros
         J48= zeros
         J38= zeros
         J28= zeros
         J18= zeros
         
     
     
     if not BOOL_concentration:

         J73 = zeros
         J74 = zeros
         J75 = zeros
         J76 = zeros
         
    
         
     if BOOL_two_open_ends:
         
         J11 = J11  #+ Lx_zeta.multiply(-(1-h_east)*1/(2*dx)*lamnda**2/(1+1/(1-1+H2/H1)**2))                                            #zetas, zetas
         J12 = J12  #+ Lx_zeta.multiply(-(1-h_east)*1/(2*dx)*-r/(1-1+H2/H1)*lamnda**2/(1+1/(1-1+H2/H1)**2))                                          #zetas, zetac
         J13 = J13                                              #zetas, us
         J14 = J14 +sp.kron(np.array([EastBoundary]).T,(EastBoundary/(2*dx)*(1-H2/H1)))      #zetas, uc
         J15 = J15                                              #zetas, vs
        
         
       
         J21 = J21  #+ Lx_zeta.multiply((1-h_east)*1/(2*dx)*-lamnda**2*r/(1-1+H2/H1)/(1+1/(1-1+H2/H1)**2))                                                     #zetac, zetas
         J22 = J22  #+ Lx_zeta.multiply((1-h_east)*1/(2*dx)*-lamnda**2/(1+1/(1-1+H2/H1)**2))                                                        #zetac, zetac
         J23 = J23 - sp.diags(EastBoundary/(2*dx)*(1-H2/H1))               #zetac, us
         J24 = J24                                                          #zetac, uc
         

         J77 = J77 + -epsilon*k*lamndad*(Lx_zeta.multiply(EastBoundary*beta(h)*1/(2*dx)*(1-H2/H1))+sp.diags(EastBoundary*beta(h)*1/(2*dx)*(1-H2/H1)+beta(1-H2/H1)*(1-H2/H1)/(4*dx**2)))                                          #zetac, vu
         
    
         
     
     J=sp.bmat([
                [J11, J12, J13, J14, J17, J18],
                [J21, J22, J23, J24, J27, J28],
                [J31, J32, J33, J34, J37, J38],
                [J41, J42, J43, J44, J47, J48],
                [J71, J72, J73, J74, J77, J78],
                [J81, J82, J83, J84, J87, J88]
                ],format='csr')
     return J
 
#J=Jacobian(Uinnitalguess) 

NJ=NumericalJacobian(Uinnitalguess)

# def residual(U,U1,ht):
#     return U-U1+ht*F(U1)


# def NewtonRapsonInnerloop(Uinnitalguess:'np.ndarray'):
#     Uiend=np.copy(Uinnitalguess)
    
#     i=0
    
# #     if Boolean_print: print('\t Newton Rapson Inner loop \n i=0 \t ||U-U_i+h_t*F(U_i)|| = %f < %f ' %(MaxNormOfU(residual(Uinnitalguess,Uiend,ht)),epsilon))
    
# #     if MaxNormOfU(residual(Uinnitalguess,Uiend,ht))>epsilon:
# #         Stopcondition=1
# #     else:
# #         Stopcondition=0
# #         #Uiend=Uiend+la.spsolve(I_U-ht*Jacobian(Uiend),residual(Uinnitalguess,Uiend,ht))
    
# #     while Stopcondition==1:
#     for i in tqdm(range(1,20)):
        
#          Uiend=Uiend-la.spsolve(J,F(Uiend))
#          #i+=1
        
# #         if MaxNormOfU(residual(Uinnitalguess,Uiend,ht))>epsilon:
# #             Stopcondition=1
# #             if Boolean_print:print('\t Newton Rapson Inner loop \n i=%i \t ||U-U_i+h_t*F(U_i)|| = %f < %f' %(i,MaxNormOfU(residual(Uinnitalguess,Uiend,ht)),epsilon))
# #         if MaxNormOfU(residual(Uinnitalguess,Uiend,ht))>1000000000*epsilon: # this is the fail save if the explodes.
# #             if Boolean_print:print('\n \t -----Divergence----- ')
# #             break 
# #         if MaxNormOfU(residual(Uinnitalguess,Uiend,ht))<=epsilon:
# #             Stopcondition=0
# #             if Boolean_print:print('\t Newton Rapson Inner loop \n i=%i \t ||U-U_i+h_t*F(U_i)|| = %f < %f' %(i,MaxNormOfU(residual(Uinnitalguess,Uiend,ht)),epsilon))
    
#     return Uiend

def NewtonRapsonInnerloop(Uinnitalguess:'np.ndarray'):
    epsilon=10**(-6)
    
    Uiend=np.copy(Uinnitalguess)
    
    i=0
    
    DeltaU=la.spsolve(NumericalJacobian(Uiend),F(Uiend))
    Uiend=Uiend-DeltaU

    print('\t Newton Rapson loop \n i=0 \t ||F(U)|| = %f < %f \n ' %(MaxNormOfU(F(Uiend)),epsilon))
    
    if MaxNormOfU(F(Uiend))>epsilon:
        Stopcondition=1
        
    else:
        Stopcondition=0
        
    
    while Stopcondition==1:
        
         if BOOL_print_NR :
             fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
             deltazetas,deltazetac,deltaus,deltauc,deltaC,deltah=np.array_split(DeltaU,6)
             ax1.plot(deltazetas)
             ax1.title.set_text('$\delta \zeta-sin$')
             ax2.plot(deltazetac)
             ax2.title.set_text('$\delta \zeta-cos$')
             ax3.plot(deltaus)
             ax3.title.set_text('$\delta u-sin$')
             ax4.plot(deltauc)
             ax4.title.set_text('$\delta u-cos$')
             ax5.plot(deltaC)
             ax5.title.set_text('$\delta C$')
             ax6.plot(deltah)
             ax6.title.set_text('$\delta h$')
             
         
             plt.suptitle('itteration i=%i' %(i))
         
         
         
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
             
             
         if i>20:
            break
         # zetas,zetac,us,uc,vs,vc,C,h=np.array_split(DeltaU,8)
         # Check=np.where(1-h+zetas<0)
         # if Check[0].size>0 :
         #        print('\t non physical water  \t \n')
            
         # Check=np.where(C<0)
         # if Check[0].size>0 :
         #        print('\n ------- \t  NON physical Concentration! \t ---------\n')
    return Uiend    

# the run
    
Ufinal=NewtonRapsonInnerloop(Uinnitalguess)
zetas,zetac,us,uc,C,h=np.array_split(Ufinal,6)

Check=np.where(F(Ufinal)!=0)
if Check[0].size>0 :
    print('\t ------- F=!0 ------- \t \n')
    F_zetas,F_zetac,F_us,F_uc,F_C,F_h=np.array_split(F(Ufinal),6)
    fig, ((ax1,ax3,ax5),(ax2,ax4,ax6)) = plt.subplots(2,3)
    ax1.plot(F_zetas,'k.',markersize=1)
    ax1.title.set_text('$F(\zeta^{s})$')
    ax2.plot(F_zetac,'k.',markersize=1)
    ax2.title.set_text('$F(\zeta^{c})$')
    ax3.plot(F_us,'k.',markersize=1)
    ax3.title.set_text('$F(u^{s})$')
    ax4.plot(F_uc,'k.',markersize=1)
    ax4.title.set_text('$F(u^{c})$')
    ax5.plot(F_C,'k.',markersize=1)
    ax5.title.set_text('$F(C)$')
    ax6.plot(F_h,'k.',markersize=1)
    ax6.title.set_text('$F(h)$')

#checks

#C0,C1=NRConcentration(Ufinal,np.concatenate((ICzeta0,ICzeta0)),np.concatenate((ICzeta0,ICzeta0)))

#Cs,Cc=np.array_split(C0,2)

#C2s,C2c=np.array_split(C1,2)

#the plots

def staticplots():
        plt.ion()
        
        fig, ((ax1, ax2),(ax4, ax5)) = plt.subplots(2, 2)
        

        # initialization of the movie figure
        zetasarr = zetas
        zetacarr =zetac
        usarr = us
        ucarr = uc

        
        
    
        
        imgzetas = ax1.plot(zetasarr)
        
        ax1.title.set_text('zeta-sin')
        
        
        imgzetac = ax4.plot(zetacarr)
        
        ax4.title.set_text('zeta-cos')
        
    
     
        imgus = ax2.plot(usarr)
        
        
        imguc = ax5.plot(ucarr)
        


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
    zeta0arr = np.reshape(zeta0,[Ny-1,Nx-1])
    u0arr = np.reshape(u0,[Ny-1,Nx-1])
    v0arr = np.reshape(v0,[Ny-1,Nx-1])
        
    X = np.linspace(0, 1, Nx-1)
    Y = np.linspace(0, 1, Ny-1)
    U, V =v0arr, u0arr
    
    imgzeta = ax1.imshow(zeta0arr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
    ax1.title.set_text('zeta')
    plt.gca().invert_yaxis()
    
    plt.colorbar(imgzeta,orientation='horizontal',ax=ax1)
    

    imgzetacross = ax12.plot(zeta0arr.mean(0),linewidth=1,color='k')
    ax12.set_ylim([-5*(Atilde+A),5*(Atilde+A)])  
    
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
            
        imgzetacross[0].set_ydata(np.reshape(zeta1,[Ny-1,Nx-1]).mean(0))       
        imgzeta.set_array(np.reshape(zeta1,[Ny-1,Nx-1]))
        imgzeta.set_clim(zeta1.min(),zeta1.max())
        
        ax2.clear()
        ax2.quiver(X, Y, np.reshape(u1,[Ny-1,Nx-1]), np.reshape(v1,[Ny-1,Nx-1]),np.reshape(np.sqrt(u1**2+v1**2),[Ny-1,Nx-1]),pivot='mid',units='dots',headwidth=0.1,linewidth=0.1,headlength=0.1)
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
# plt.figure()
# plt.imshow(np.reshape(h,[Ny-1,Nx-1]))
# plt.figure()
# plt.imshow(np.reshape(C,[Ny-1,Nx-1]))
# plt.show()

Check_1=np.where(1-h+zetas<0)
Check_2=np.where(1-h+zetac<0)
if Check_1[0].size>0 or Check_2[0].size>0 :
    print('\n ------- \t  NON physical water model! \t ---------\n')

Check=np.where(C<0)
if Check[0].size>0 :
    print('\n ------- \t  NON physical Concentration! \t ---------\n')
    


