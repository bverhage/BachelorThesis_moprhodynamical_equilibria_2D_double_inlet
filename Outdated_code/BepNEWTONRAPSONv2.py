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
    H2=6#6
    
    A=0.74*1/H1
    
    Atilde=0.84*1/H1
    
    phi=41/180*np.pi
    
    g=9.81 #m/s^2
    
    CD=0.0025
    sigma=1.4*10**-4 #/s
    
    lamnda=sigma*Lx/(np.sqrt(g*H1))
    Nx=100+1#25#100
    Ny=int((Nx-1)*1/2)+1
    
    dx=1/Nx
    
    dy=1/Ny
    a=6.222*10**(-2)
    r=-(8*CD*A*Lx)/(3*np.pi*H1**2)/(H1*sigma)
    #r=-0.025
    # beta=0.08
    
    fhat=0#7.1*10**-1
    
    k=2.052*10**(-4)
    Mutilde=10**(-6)
    epsilon = A/H1
    
    BOOL_concentration= True
    BOOL_bedevolution= False
    BOOL_two_open_ends= True
    
    BOOL_print_NR= True
    
    def deltafunc(x):
        if  np.abs(x)<=1/(2*dx): return 1
        else: return 0
            
    def bedfunction1(x,y,alpha,Q):
        return np.sum(
            list(
                [deltafunc(x-Q[i][0])*np.exp(alpha*np.linalg.norm(np.array([x,y])-Q[i])**2) for i in range(Q.shape[0])]
                )
            )

        
    def bedfunction2(x,y):
        
        Q=np.array([np.array([x,0.5+0.3*np.sin(2*np.pi*x)]) for x in np.linspace(0,1,Nx)])
        F=bedfunction1(x,y,-20,Q)
        return F
    
    
    def bedprofile(x,y):
         # if x>0.1:
         #     if (y-0.5)**2<0.4**2: return 0
         #     else: return 0.7
         # else: return 0
         #return 0.95/(1+np.exp(-100*(x-0.2)))-bedfunction2(x,y)*(1-1/(1+np.exp(-100*(x-0.2))))
         #return 0.8/(1+np.exp(-100*(x-0.2)))-bedfunction2(x,y)*(1/(1+np.exp(-100*(x-0.2))))/(30)
         return (1-H2/H1)*x
    
    def func0(x,y):
        return 0
    
    def westboundary(x,y):
        if x==1*dx:
            return 1

        else: return 0
        
    def northboundary(x,y):
        if y==1-dy: return 1

        else: return 0
        
    
    def eastboundary(x,y):
        if x==1-dx: return 1
        return 0
    
    def southboundary(x,y):
        if y==dy: return 1
        else: return 0
        
    def create(Nx:int,Ny:int,func):
        Fvec=np.zeros((Nx-1)*(Ny-1))
        for j in range(Ny-1):
            for i in range(Nx-1):
                x=dx*(i+1)
                y=dy*(j+1)
                Fvec[i+j*(Nx-1)]=func(x,y)
        return Fvec
    
    WestBoundary = create(Nx,Ny,westboundary)
    EastBoundary = create(Nx,Ny,eastboundary)
    NorthBoundary = create(Nx,Ny,northboundary)
    SouthBoundary = create(Nx,Ny,southboundary)
    
    
    ICzeta0 = create(Nx,Ny,func0)
    ICu0    = create(Nx,Ny,func0)
    ICv0    = create(Nx,Ny,func0)
    ICh0    = create(Nx,Ny,bedprofile)
    
    Check=np.where(ICh0>1)
    if Check[0].size>0 :
        print('\n ------- \t  BED violates the model! \t ---------\n')
    else:
        print('\n Boundary checked commensing Newton Rapsons algorithem')

if True:
    
   def FDLaplacian2D(Nx,Ny,dx,dy,WestNeuman=False,NorthNeuman=False,EastNeuman=False,SouthNeuman=False):
        
        #diagonalsx = [-1*np.ones((1,Nx-1-1)),np.zeros((1,Nx-1)),np.ones((1,Nx-1-1))]
        #diagonalsy = [-1*np.ones((1,Ny-1-1)),np.zeros((1,Ny-1)),np.ones((1,Ny-1-1))]
        diagonalsx = [-1*np.ones((1,Nx)),np.ones((1,Nx-1))]
        diagonalsy = [-1*np.ones((1,Ny)),np.ones((1,Ny-1))]
        
        if WestNeuman==True:
            #diagonalsx[1][0][0]=-1;
            diagonalsx[1][0][0]=0;
        if EastNeuman==True:
            #diagonalsx[1][0][-1]=1;
            diagonalsx[0][0][-2]=0;
        if NorthNeuman==True:
            #diagonalsy[1][0][0]=-1;
            diagonalsy[1][0][0]=0;
        if SouthNeuman==True:
            #diagonalsy[1][0][-1]=1;
            diagonalsy[0][0][-2]=0;
            
        
        Dx =1/(dx)*sp.diags(diagonalsx, [-1, 0], shape=(Nx,Nx-1))
        Dy =1/(dy)*sp.diags(diagonalsy, [-1, 0], shape=(Ny,Ny-1))
        
        LXX=Dx.transpose().dot(Dx)
        LYY=Dy.transpose().dot(Dy)

        Ans = sp.kron(LYY,sp.eye(Nx-1))+sp.kron(sp.eye(Ny-1),LXX)    
        return Ans

   def LX(Nx,Ny,dx,dy,LeftNeuman=False,RightNeuman=False):
        
        diagonalsx = [-1*np.ones((1,Nx-1-1)),np.zeros((1,Nx-1)),np.ones((1,Nx-1-1))]
        
        if LeftNeuman==True:
            diagonalsx[1][0][0]=-1;
        if RightNeuman==True:
            diagonalsx[1][0][-1]=1;
    
        Dx =1/(2*dx)*sp.diags(diagonalsx, [-1,0 ,1], shape=(Nx-1,Nx-1))
        
        Ans =sp.kron(sp.eye(Ny-1),Dx)
        #Ans = sp.kron(Dx,sp.eye(Ny-1))
        return Ans

       
    
   def LY(Nx,Ny,dx,dy,LeftNeuman=False,RightNeuman=False):
        
        diagonalsy = [-1*np.ones((1,Ny-1-1)),np.zeros((1,Ny-1)),np.ones((1,Ny-1-1))]
        
        if LeftNeuman==True:
            diagonalsy[1][0][0]=-1;
        if RightNeuman==True:
            diagonalsy[1][0][-1]=1;


        Dy =1/(2*dy)*sp.diags(diagonalsy, [-1,0, 1], shape=(Ny-1,Ny-1))
       
        
        

        Ans =sp.kron(Dy,sp.eye(Nx-1))
        #Ans= sp.kron(sp.eye(Nx-1),Dy)
        return Ans
    
    
        
        
 
# 2D FD Laplacian and identity matrices

LxD  =  LX(Nx,Ny,dx,dy)
#LxD  =  LX(Nx,Ny,dx,dy,True,False) #WRONG
 
LxND =  LX(Nx,Ny,dx,dy,True)
LxDN =  LX(Nx,Ny,dx,dy,False,True)
LxN  =  LX(Nx,Ny,dx,dy,True,True)

#LxN  =  LX(Nx,Ny,dx,dy,False,True) #WRONG

LyD  = LY(Nx,Ny,dx,dy)
LyND = LY(Nx,Ny,dx,dy,True)
LyDN = LY(Nx,Ny,dx,dy,False,True)
LyN  = LY(Nx,Ny,dx,dy,True,True)

ADDDD = FDLaplacian2D(Nx,Ny,dx,dy)
ADNNN = FDLaplacian2D(Nx,Ny,dx,dy,False,True,True,True)
ADNDN = FDLaplacian2D(Nx,Ny,dx,dy,False,True,False,True)
#ADNDN = FDLaplacian2D(Nx,Ny,dx,dy,True,True,False,True) #wrong

ANDDD = FDLaplacian2D(Nx,Ny,dx,dy,True,False,False,False)


I = sp.eye((Nx-1)*(Ny-1))

ONES= np.ones((Nx-1)*(Ny-1))

#ICzetas,ICzetac,ICus,ICuc,ICvs,ICvc=np.array_split(np.loadtxt('test1.txt', dtype=int),6) 




if False:
    if Nx==150+1 and Ny==150+1:
            Uinnitalguess=np.loadtxt('data.csv',delimiter=',')

else:
    Uinnitalguess=np.concatenate((ICzeta0,ICzeta0,ICu0,ICu0,ICv0,ICv0,ICv0,ICh0))

# #part 2 the Newton rapson method

# # first we need to define some functions:
    
def MaxNormOfU(U):
     ''' The max 2 norm of U=(u,v)^T w'''
     zetas,zetac,us,uc,vs,vc,C,h=np.array_split(U,8)
     return np.max([np.linalg.norm(zetas),np.linalg.norm(zetac),np.linalg.norm(us),np.linalg.norm(uc),np.linalg.norm(vs),np.linalg.norm(vc)])
 
lamndad=1.8
   
def beta(h):
    return 1/(1-np.exp(-lamndad*(1-h)))
def F(U):
    zetas,zetac,us,uc,vs,vc,C,h=np.array_split(U,8)
    
    zeros=np.zeros(zetas.shape)
    
    if BOOL_two_open_ends:
     Lx_zeta= LxD
     Lx_u   = LxN
     A_h    = ADNDN
    else:
     Lx_zeta= LxDN
     Lx_u   = LxND
     A_h    = ADNDN
    
    # -------------- zetas ------------------
    fzetas =-zetas-(1-h)*(Lx_u*uc+LyD*vc)+uc*(Lx_zeta*h)+vc*(LyN*h)
    
    #fzetas =+ uc*(1-H2/H1)*EastBoundary/(2*dx)
    
    #fzetas =+ -(1-h)*EastBoundary*H1/H2*Atilde*np.sin(phi)/2
    
    # -------------- zetac ------------------
    fzetac =-zetac+(1-h)*(Lx_u*us+LyD*vs)-us*(Lx_zeta*h)-vs*(LyN*h)
    
    #fzetac =+ -us*(1-H2/H1)*EastBoundary/(2*dx)
    
    #fzetac =+ (1-h)*WestBoundary*A/2
    
    #fzetac =+ (1-h)*EastBoundary*H1/H2*Atilde*np.cos(phi)/2
    
    # -------------- us ------------------
    fus    = -us +fhat*vc-np.divide(r, 1-h)*uc-lamnda**(-2)*Lx_zeta*zetas
    
    #fus    =+ -lamnda**(-2)*Atilde*np.cos(phi)*EastBoundary/(2*dx)
    
    # -------------- uc ------------------
    fuc    = -uc -fhat*vs+np.divide(r, 1-h)*us+lamnda**(-2)*Lx_zeta*zetac
    
    #fuc    =+  -lamnda**(-2)*A*WestBoundary/(2*dx)
    
    #fuc    =+  lamnda**(-2)*Atilde*np.sin(phi)*EastBoundary/(2*dx)
    
    # -------------- vs ------------------
    fvs    =-vs-fhat*uc-np.divide(r, 1-h)*vc-lamnda**(-2)*LyN*zetas
    
    # -------------- vc ------------------
    fvc    =-vc+fhat*us   +np.divide(r, 1-h)*vs+lamnda**(-2)*LyN*zetac
    
    # -------------- C ------------------
    
    fC     = -epsilon*beta(h)*C +epsilon*a*k*A_h*C+epsilon*a*k*lamndad*(beta(h)*Lx_zeta*h*Lx_zeta*C+C*(beta(h)*A_h*h+Lx_zeta*beta(h)*Lx_zeta*h))
    if BOOL_concentration:
     fC = +(us*us+vs*vs+uc*uc+vc*vc)
    
    # -------------- h ------------------
    fh     =  -Mutilde*A_h*h
    #fh     =+ Mutilde*EastBoundary*(1-H2/H1)/(2*dx)
    if BOOL_bedevolution:
        fh     += -k*A_h*C
       
    ans=np.concatenate((fzetas,fzetac,fus,fuc,fvs,fvc,fC,fh))
    
    
    ans+= np.concatenate((
        zeros,
        +(1-h)*WestBoundary*(A/2),
        zeros,
        -lamnda**(-2)*A*WestBoundary/(2*dx),
        zeros,
        zeros,
        zeros,
        zeros
        ))
    
    if BOOL_two_open_ends:
        ans += np.concatenate((
            #zetas
            #+lamnda**(-2)*Atilde*np.cos(phi)*EastBoundary/(2*dx)*(1-H2/H1)*EastBoundary/(2*dx)
            
            (1-h)*EastBoundary*Atilde*np.sin(phi)*H1/(2*H2),#+uc*EastBoundary/(2*dx)*(1-H2/H1-(1-h)*H1/H2*(1-H2/H1)),
            
            #zetac
            #-us*(1-H2/H1)*EastBoundary*H/(2*dx)
            
            (1-h)*EastBoundary*Atilde*np.cos(phi)*H1/(2*H2),#-us*EastBoundary/(2*dx)*(1-H2/H1-(1-h)*H1/H2*(1-H2/H1)),
            #us
            -lamnda**(-2)*Atilde*np.sin(phi)*EastBoundary/(2*dx),
            #uc
            +lamnda**(-2)*Atilde*np.cos(phi)*EastBoundary/(2*dx),
            #vc
            zeros,
            #vs
            zeros,
            #C
            EastBoundary*-epsilon*a*k*lamndad*( beta(h)*(1/(2*dx)*(1-H2/H1)*Lx_zeta*C +C*( beta(h)*1/(2*dx)*(1-H2/H1)+ beta(1-H2/H1)*(1-H2/H1)/(4*dx**2) ))),#a*k*lamndad*beta(h)*((1-H2/H1)/(2*dx)*C+(1-H2/H1)/(2*dx)*Lx_zeta*C)*EastBoundary,
            #h
            Mutilde*EastBoundary*(1-H2/H1)/(2*dx)
            ))

    return ans




def Jacobian(U):
     zetas,zetac,us,uc,vs,vc,C,h=np.array_split(U,8)
     zeros=sp.csr_matrix(LxD.shape)
     
     if BOOL_two_open_ends:
      Lx_zeta= LxD
      Lx_u   = LxN
      A_h    = ADNDN
     else:
      Lx_zeta= LxDN
      Lx_u   = LxND
      A_h    = ADNNN
     
     J11 = -I                                      # zetas,zetas
     J12 = zeros                                   # zetas,zetac
     J13 = zeros                                   # zetas, us
     J14 = -Lx_u.multiply(1-h.T)+sp.diags(Lx_zeta*h)     # zetas, uc
     J15 = zeros                                   # zetas, vs
     J16 = -LyD.multiply(1-h.T)+sp.diags(LyN*h)      # zetas, vc
     J17 = zeros                                    # zetas, C
     J18 = sp.diags(Lx_u*uc+LyD*vc)+Lx_zeta.multiply(uc.T)+LyN.multiply(vc.T) #zetas, h
     
     J21 = zeros
     J22 = -I
     J23 =  Lx_u.multiply(1-h.T)-sp.diags(Lx_zeta*h)
     J24 = zeros
     J25 =  LyD.multiply(1-h.T)-sp.diags(LyN*h)
     J26 = zeros
     J27 = zeros
     J28 = -sp.diags(Lx_u*us+LyD*vs)-Lx_zeta.multiply(us.T)-LyN.multiply(vs.T)
     
     J31 = -lamnda**(-2)*Lx_zeta
     J32 = zeros
     J33 = -I
     J34 = -sp.diags(np.divide(r,1-h))
     J35 = zeros
     J36 = I*fhat
     J37 = zeros
     J38 = -sp.diags(r*(1-h)**(-2)*uc)
     
     J41= zeros
     J42= lamnda**(-2)*Lx_zeta
     J43= sp.diags(np.divide(r,1-h))
     J44= -I
     J45= -I*fhat
     J46=  zeros
     J47 = zeros
     J48 = sp.diags(r*(1-h)**(-2)*us)
          
     J51 = -lamnda**(-2)*LyN
     J52 = zeros
     J53 = zeros
     J54 = -I*fhat
     J55 = -I
     J56 = -sp.diags(np.divide(r,1-h))
     J57 = zeros
     J58 = -sp.diags(r*(1-h)**(-2)*vc)
     
     J61= zeros
     J62= lamnda**(-2)*LyN
     J63= I*fhat
     J64= zeros
     J65= sp.diags(np.divide(r,1-h))
     J66= -I
     J67 = zeros
     J68 = sp.diags(r*(1-h)**(-2)*vs)
     
     J71 = zeros
     J72 = zeros
     J73 = 2*sp.diags(us)
     J74 = 2*sp.diags(uc)
     J75 = 2*sp.diags(vs)
     J76 = 2*sp.diags(vc)
     J77 = sp.diags(-epsilon*beta(h))+epsilon*a*k*A_h+epsilon*a*k*lamndad*(Lx_zeta.multiply(beta(h)*Lx_zeta*h)+sp.diags(beta(h)*A_h*h+Lx_zeta*beta(h)*Lx_zeta*h))
     J78 = zeros
     
     J81 = zeros
     J82 = zeros
     J83 = zeros
     J84 = zeros
     J85 = zeros
     J86 = zeros
     J87 =-k*A_h
     J88 =-Mutilde*A_h
     
     
     
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
         
         J11 = J11                                              #zetas, zetas
         J12 = J12                                              #zetas, zetac
         J13 = J13                                              #zetas, us
         J14 = J14 + sp.diags((1-H2/H1)*EastBoundary/(2*dx))        #zetas, uc
         J15 = J15                                              #zetas, vs
         J16 = J16 #-0*LyD.multiply(-(1-h)*EastBoundary)+ 0*sp.diags(-(1-h)*EastBoundary*LyN*h*H1/H2) #zetas, vc
         
       
         J21 = J21                                                          #zetac, zetas
         J22 = J22                                                          #zetac, zetac
         J23 = J23 - sp.diags((1-H2/H1)*EastBoundary/(2*dx))               #zetac, us
         J24 = J24                                                          #zetac, uc
         J25 = J25 #- 0*LyD.multiply((1-h)*EastBoundary/2)+ 0*sp.diags((1-h)*EastBoundary/2*LyN*h*H1/H2)             #zetac, vs
         J26 = J26    

         J77 = J77 + -epsilon*k*lamndad*(Lx_zeta.multiply(EastBoundary*beta(h)*1/(2*dx)*(1-H2/H1))+sp.diags(EastBoundary*beta(h)*1/(2*dx)*(1-H2/H1)+beta(1-H2/H1)*(1-H2/H1)/(4*dx**2)))                                          #zetac, vu
         
        
         
     
     J=sp.bmat([
                [J11, J12, J13, J14, J15, J16, J17, J18],
                [J21, J22, J23, J24, J25, J26, J27, J28],
                [J31, J32, J33, J34, J35, J36, J37, J38],
                [J41, J42, J43, J44, J45, J46, J47, J48],
                [J51, J52, J53, J54, J55, J56, J57, J58],
                [J61, J62, J63, J64, J65, J66, J67, J68],
                [J71, J72, J73, J74, J75, J76, J77, J78],
                [J81, J82, J83, J84, J85, J86, J87, J88]
                ],format='csr')
     return J
 
J=Jacobian(Uinnitalguess) 

def F2(U):
    zetas,zetac,us,uc,vs,vc,C,h=np.array_split(U,8)
    if BOOL_two_open_ends:
      Lx_zeta= LxD
      Lx_u   = LxN
      A_h    = ADNNN
    else:
      Lx_zeta= LxDN
      Lx_u   = LxND
      A_h    = ADNDN
     
    zeros=np.zeros(zetas.shape)
    
    ans=Jacobian(Uinnitalguess)*U

    ans+= np.concatenate((
        zeros,
        +(1-h)*WestBoundary*(A/2),
        zeros,
        -lamnda**(-2)*A*WestBoundary/(2*dx),
        zeros,
        zeros,
        zeros,
        zeros
        ))
    
    if BOOL_two_open_ends:
        ans+= np.concatenate((
            #zetas
            #+lamnda**(-2)*Atilde*np.cos(phi)*EastBoundary/(2*dx)*(1-H2/H1)*EastBoundary/(2*dx)
            
            (1-h)*EastBoundary*Atilde*np.sin(phi)*H1/(2*H2)+uc*EastBoundary/(2*dx)*(1-H2/H1-(1-h)*H1/H2*(1-H2/H1)),
            
            #zetac
            #-us*(1-H2/H1)*EastBoundary*H/(2*dx)
            
            (1-h)*EastBoundary*Atilde*np.cos(phi)*H1/(2*H2)-us*EastBoundary/(2*dx)*(1-H2/H1-(1-h)*H1/H2*(1-H2/H1)),
            #us
            -lamnda**(-2)*Atilde*np.sin(phi)*EastBoundary/(2*dx),
            #uc
            +lamnda**(-2)*Atilde*np.cos(phi)*EastBoundary/(2*dx),
            #vc
            zeros,
            #vs
            zeros,
            #C
            EastBoundary*-epsilon*a*k*lamndad*( beta(h)*(1/(2*dx)*(1-H2/H1)*Lx_zeta*C +C*( beta(h)*1/(2*dx)*(1-H2/H1)+ beta(1-H2/H1)*(1-H2/H1)/(4*dx**2) ))),#a*k*lamndad*beta(h)*((1-H2/H1)/(2*dx)*C+(1-H2/H1)/(2*dx)*Lx_zeta*C)*EastBoundary,
            #h
            Mutilde*EastBoundary*(1-H2/H1)/(2*dx)
            ))

    if BOOL_bedevolution and BOOL_two_open_ends: 
        ans += np.concatenate((zeros,zeros,zeros,zeros,zeros,zeros,zeros,Mutilde*EastBoundary*(1-H2/H1)/(2*dx)))

    return ans




def FConcentration0(C0):
    Cs,Cc=np.array_split(C0,2)
    

    
    fCs = -Cs+k*ADNDN*Cc-Cc
    fCc = -Cc-k*ADNDN*Cs+Cs
    
    return np.concatenate((fCs,fCc))


def JacobianConcentration0():
    J11=-I
    J12=k*ADNDN-I
    
    J21= -k*ADNDN+I
    J22= -I
    J=sp.bmat([
            [J11,J12],
            [J21,J22] 
        ],format='csr')
    return J

def FConcentration1(C1,C0,U0):
    C2s,C2c = np.array_split(C1,2)
    Cs,Cc   =  np.array_split(C0,2)
    zetas,zetac,us,uc,vs,vc=np.array_split(U0,6)
    
    
    fC2s = -2*C2s
    fC2s += 1/2*uc*LxDN*Cs+1/2*Cs*LxND*uc+1/2*us*LxDN*Cc+1/2*Cc*LxND*us
    fC2s += 1/2*vc*LyN*Cs +1/2*Cs*LyD*vc +1/2*vs*LyN*Cc +1/2*Cc*LyD*vs
    fC2s += k*ADNDN*C2c
    fC2s += 1/epsilon*us*uc-C2c
    
    fC2c = -2*C2c
    fC2c += 1/2*uc*LxDN*Cc+1/2*Cc*LxND*uc+1/2*us*LxDN*Cs+1/2*Cs*LxND*us
    fC2c += 1/2*vc*LyN*Cc +1/2*Cc*LyD*vc +1/2*vs*LyN*Cs +1/2*Cs*LyD*vs
    fC2c += k*ADNDN*C2s
    fC2s += 1/epsilon*(1/2*us*us+1/2*uc*uc)-C2s
    
    fC2c=-fC2c
    return np.concatenate((fC2s,fC2c))


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
    epsilon=.1
    
    Uiend=np.copy(Uinnitalguess)
    
    i=0
    
    DeltaU=la.spsolve(Jacobian(Uiend),F(Uiend))
    Uiend=Uiend-DeltaU

    print('\t Newton Rapson loop \n i=0 \t ||delta U|| = %f < %f \n ' %(MaxNormOfU(F(Uiend)),epsilon))
    
    if MaxNormOfU(DeltaU)>epsilon:
        Stopcondition=1
        
    else:
        Stopcondition=0
        
    
    while Stopcondition==1:
         if BOOL_print_NR :
             fig, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8)) = plt.subplots(2,4)
             deltazetas,deltazetac,deltaus,deltauc,deltavs,deltavc,deltaC,deltah=np.array_split(DeltaU,8)
             ax1.imshow(np.reshape(deltazetas,[Ny-1,Nx-1]))
             ax1.title.set_text('$\delta \zeta-sin$')
             ax2.imshow(np.reshape(deltazetac,[Ny-1,Nx-1]))
             ax2.title.set_text('$\delta \zeta-cos$')
             ax3.imshow(np.reshape(deltaus,[Ny-1,Nx-1]))
             ax3.title.set_text('$\delta u-sin$')
             ax4.imshow(np.reshape(deltauc,[Ny-1,Nx-1]))
             ax4.title.set_text('$\delta u-cos$')
             ax5.imshow(np.reshape(deltavs,[Ny-1,Nx-1]))
             ax5.title.set_text('$\delta v-sin$')
             ax6.imshow(np.reshape(deltavc,[Ny-1,Nx-1]))
             ax6.title.set_text('$\delta v-cos$')
             ax7.imshow(np.reshape(deltaC,[Ny-1,Nx-1]))
             ax7.title.set_text('$\delta C$')
             ax8.imshow(np.reshape(deltah,[Ny-1,Nx-1]))
             ax8.title.set_text('$\delta h$')
             
             DeltaU=la.spsolve(Jacobian(Uiend),F(Uiend))
         
             plt.suptitle('itteration i=%i' %(i))
         
         
         
         i+=1
         if MaxNormOfU(DeltaU)>10**(6)*epsilon: # this is the fail save if the explodes.
             print('\n \t -----Divergence----- \n')
             break 
         if MaxNormOfU(DeltaU)>epsilon:
             Uiend=Uiend-DeltaU
             Stopcondition=1
             print('\t Newton Rapson loop \n i=%i \t ||delta U|| = %f < %f\n' %(i,MaxNormOfU(F(Uiend)),epsilon))

         if MaxNormOfU(DeltaU)<=epsilon:
             Stopcondition=0
             print('\t Newton Rapson loop \n i=%i \t ||delta U|| = %f < %f\n' %(i,MaxNormOfU(F(Uiend)),epsilon))    
         if i>4:
            break
         # zetas,zetac,us,uc,vs,vc,C,h=np.array_split(DeltaU,8)
         # Check=np.where(1-h+zetas<0)
         # if Check[0].size>0 :
         #        print('\t non physical water  \t \n')
            
         # Check=np.where(C<0)
         # if Check[0].size>0 :
         #        print('\n ------- \t  NON physical Concentration! \t ---------\n')
    return Uiend    

def NRConcentration(Ufinal:'np.ndarray',C0innitialguess:'np.ndarray',C1innitialguess:'np.ndarray'):
    epsilon_1=1#10**(-10)
    C0iend=np.copy(C0innitialguess)
    C1iend=np.copy(C1innitialguess)
    
    i=0
    
    DeltaC0=la.spsolve(JacobianConcentration0(),FConcentration0(C0iend))
    DeltaC1=la.spsolve(JacobianConcentration0(),FConcentration1(C1iend,C0iend,Ufinal))
    
    print('\t Newton Rapson loop \n i=0 \t ||delta C^0|| = %f , ||delta C^1|| = %f ' %(MaxNormOfU(DeltaC0),MaxNormOfU(DeltaC1)))
    
    if max(MaxNormOfU(DeltaC0),MaxNormOfU(DeltaC1))>epsilon_1:
        Stopcondition=1
    else:
        Stopcondition=0
        C0iend=C0iend-DeltaC0
        C1iend=C1iend-DeltaC1
        
    while Stopcondition==1:
        
         DeltaC0=la.spsolve(JacobianConcentration0(),FConcentration0(C0iend))
         DeltaC1=la.spsolve(JacobianConcentration0(),FConcentration1(C1iend,C0iend,Ufinal))
         C0iend=C0iend-DeltaC0
         C1iend=C1iend-DeltaC1
         
         i+=1
        
         if max(MaxNormOfU(DeltaC0),MaxNormOfU(DeltaC1))>epsilon_1:
             Stopcondition=1
             print('\t Newton Rapson loop \n i=0 \t ||delta C^0|| = %f , ||delta C^1|| = %f  ' %(MaxNormOfU(DeltaC0),MaxNormOfU(DeltaC1)))
         if max(MaxNormOfU(DeltaC0),MaxNormOfU(DeltaC1))>10**(20)*epsilon_1: # this is the fail save if the explodes.
             print('\n \t -----Divergence----- ')
             break 
         if max(MaxNormOfU(DeltaC0),MaxNormOfU(DeltaC1))<=epsilon_1:
             Stopcondition=0
             print('\t Newton Rapson loop \n i=0 \t ||delta C^0|| = %f , ||delta C^1|| = %f  ' %(MaxNormOfU(DeltaC0),MaxNormOfU(DeltaC1)))
         if i>1:
            break
    
    return C0iend,C1iend
     

# the run
    
Ufinal=NewtonRapsonInnerloop(Uinnitalguess)
zetas,zetac,us,uc,vs,vc,C,h=np.array_split(Ufinal,8)

#checks

#C0,C1=NRConcentration(Ufinal,np.concatenate((ICzeta0,ICzeta0)),np.concatenate((ICzeta0,ICzeta0)))

#Cs,Cc=np.array_split(C0,2)

#C2s,C2c=np.array_split(C1,2)

#the plots

def staticplots():
        plt.ion()
        
        fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3)
        

        # initialization of the movie figure
        zetasarr = np.reshape(zetas,[Ny-1,Nx-1])
        zetacarr = np.reshape(zetac,[Ny-1,Nx-1])
        usarr = np.reshape(us,[Ny-1,Nx-1])
        ucarr = np.reshape(uc,[Ny-1,Nx-1])
        vsarr = np.reshape(vs,[Ny-1,Nx-1])
        vcarr = np.reshape(vc,[Ny-1,Nx-1])
        
        
    
        
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
        
        fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2,3)
        
        # Inital conditoin 
        zeta0=zetas*np.sin(t)+zetac*np.cos(t)
        u0=uc
        v0=vc

        # initialization of the movie figure
        zeta0arr = np.reshape(zeta0,[Ny-1,Nx-1])
        u0arr = np.reshape(u0,[Ny-1,Nx-1])
        v0arr = np.reshape(v0,[Ny-1,Nx-1])

        
        
        imgzetacross = ax1.plot(zeta0arr.mean(0),'k.',markersize=1)
        ax1.title.set_text('zeta cross avg')
        ax1.set_ylim([-A,A])
        
        imgucross = ax2.plot(u0arr.mean(0),'k.',markersize=1)
        ax2.title.set_text('u cross avg')
        #ax2.set_ylim([-0.2,0.2])
        
        imgvcross = ax3.plot(v0arr.mean(0),'k.',markersize=1)
        ax3.title.set_text('v cross avg')
        
        
      
        
        imgzeta = ax4.imshow(zeta0arr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        ax4.title.set_text('zeta')
        plt.gca().invert_yaxis()
        
        plt.colorbar(imgzeta,orientation='horizontal',ax=ax4)
     
        imgu=ax5.imshow(u0arr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        plt.gca().invert_yaxis()
        ax5.title.set_text('u')
        plt.colorbar(imgu,orientation='horizontal',ax=ax5)
        
        imgv=ax6.imshow(v0arr,extent=[dx/2,Lx-dx/2,Ly-dy/2,dy/2],interpolation='none',aspect='auto')
        
        plt.gca().invert_yaxis()
        ax6.title.set_text('v')
        plt.colorbar(imgu,orientation='horizontal',ax=ax6)
        
        
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
            imgucross[0].set_ydata(np.reshape(u1,[Ny-1,Nx-1]).mean(0))
            imgvcross[0].set_ydata(np.reshape(v1,[Ny-1,Nx-1]).mean(0))

            
            imgzeta.set_array(np.reshape(zeta1,[Ny-1,Nx-1]))
            imgu.set_array(np.reshape(u1,[Ny-1,Nx-1]))
            
            imgv.set_array(np.reshape(v1,[Ny-1,Nx-1]))


            
            
            tlt.set_text('t = %3.3f' %(t))
            imgzeta.set_clim(zeta1.min(),zeta1.max())
            imgu.set_clim(u1.min(),u1.max())
            imgv.set_clim(v1.min(),v1.max())            
                                                
            return imgzetacross,imgucross,imgvcross,imgzeta,imgu,imgv
        
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
Animation2()
plt.figure()
plt.imshow(np.reshape(h,[Ny-1,Nx-1]))
plt.figure()
plt.imshow(np.reshape(C,[Ny-1,Nx-1]))
plt.show()

Check=np.where(1-h+zetas<0)
if Check[0].size>0 :
    print('\n ------- \t  NON physical water model! \t ---------\n')

Check=np.where(C<0)
if Check[0].size>0 :
    print('\n ------- \t  NON physical Concentration! \t ---------\n')