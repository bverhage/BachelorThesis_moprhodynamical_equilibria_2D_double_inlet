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
    Lx=10*10**3 #m
    Ly=1*10**3 #m
    H=10#m
    
    A=1/H 
    
    g=9.81 #m/s^2
    
    CD=0.001
    sigma=1.4*10**-4 #/s
    
    lamnda=sigma*Lx/(np.sqrt(g*H))
    Nx=100+1#25#100
    Ny=int((Nx-1)*1/2)+1
    
    dx=1/Nx
    
    dy=1/Ny
    
    r=(8*CD*A*Lx)/(3*np.pi*H**2)
    #r=-0.025
    # beta=0.08
    
    fhat=7.1*10**-1
    
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
         return 0.8/(1+np.exp(-100*(x-0.2)))-bedfunction2(x,y)*(1/(1+np.exp(-100*(x-0.2))))/(30)
         #return #1*x+1*(y-1)*y*x 
    
    def func0(x,y):
        return 0
    
    def westboundary(x,y):
        if x==1*dx:
            return 1

        else: return 0
        
    def northboundary(x,y):
        if y==Ly-dy: return 1

        else: return 0
        
    
    def eastboundary(x,y):
        if x==Lx-dx: return 1
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
LxND =  LX(Nx,Ny,dx,dy,True)
LxDN =  LX(Nx,Ny,dx,dy,False,True)
LxN  =  LX(Nx,Ny,dx,dy,True,True)

LyD  = LY(Nx,Ny,dx,dy)
LyND = LY(Nx,Ny,dx,dy,True)
LyDN = LY(Nx,Ny,dx,dy,False,True)
LyN  = LY(Nx,Ny,dx,dy,True,True)

ADDDD = FDLaplacian2D(Nx,Ny,dx,dy)
ADNNN = FDLaplacian2D(Nx,Ny,dx,dy,False,True,True,True)

I = sp.eye((Nx-1)*(Ny-1))

ONES= np.ones((Nx-1)*(Ny-1))

h=ICh0

#ICzetas,ICzetac,ICus,ICuc,ICvs,ICvc=np.array_split(np.loadtxt('test1.txt', dtype=int),6) 




if False:
    if Nx==150+1 and Ny==150+1:
            Uinnitalguess=np.loadtxt('data.csv',delimiter=',')

else:
    Uinnitalguess=np.concatenate((ICzeta0,ICzeta0,ICu0,ICu0,ICv0,ICv0))

# #part 2 the Newton rapson method

# # first we need to define some functions:
    
def MaxNormOfU(U):
     ''' The max 2 norm of U=(u,v)^T w'''
     zetas,zetac,us,uc,vs,vc=np.array_split(U,6)
     return np.max([np.linalg.norm(zetas),np.linalg.norm(zetac),np.linalg.norm(us),np.linalg.norm(uc),np.linalg.norm(vs),np.linalg.norm(vc)])
 
def F(U):
    zetas,zetac,us,uc,vs,vc=np.array_split(U,6)
    
    fzetas =-zetas-(1-h)*(LxND*uc+LyD*vc)+uc*(LxDN*h)+vc*(LyN*h)

    
    fzetac =-zetac+(1-h)*(LxND*us+LyD*vs)-us*(LxDN*h)-vs*(LyN*h)
    fzetac =+ (1-h)*WestBoundary*A/2
  
    fus    =-us+fhat*vc+np.divide(r, 1-h)*uc-lamnda**(-2)*LxDN*zetas
    
    fuc    =-uc-fhat*vs+np.divide(r, 1-h)*us-lamnda**(-2)*LxDN*zetac
    fuc    =+ lamnda**(-2)*A*WestBoundary/(2*dx)
    
    fvs    =-vs-fhat*uc+np.divide(r, 1-h)*vc-lamnda**(-2)*LyN*zetas
    fvc    =-vc+fhat*us+np.divide(r, 1-h)*vs-lamnda**(-2)*LyN*zetac
    
    return np.concatenate((fzetas,fzetac,fus,fuc,fvs,fvc))






def Jacobian(U):
     zetas,zetac,us,uc,vs,vc=np.array_split(U,6)
     zeros=sp.csr_matrix(LxD.shape)
     
     J11= -I                                      # zetas,zetas
     J12= zeros                                   # zetas,zetac
     J13= zeros                                   # zetas, us
     J14= -LxND.multiply(1-h.T)+sp.diags(LxDN*h)#+sp.diags(-1/(2*dx)*WestBoundary*-(1-h)/(1+h))     # zetas, uc
     J15= zeros                                   # zetas, vs
     J16= -LyD.multiply(1-h.T)+sp.diags(LyN*h)      # zetas, vc
     
     J21= zeros
     J22= -I
     J23=  LxND.multiply(1-h.T)-sp.diags(LxDN*h)#-sp.diags(-1/(2*dx)*WestBoundary*-(1-h)/(1+h))
     J24= zeros
     J25=  LyD.multiply(1-h.T)-sp.diags(LyN*h)
     J26= zeros
     
     J31= -lamnda**(-2)*LxDN
     J32= zeros
     J33= -I
     J34= sp.diags(np.divide(r,1-h))
     J35= zeros
     J36= I*fhat
     
     J41= zeros
     J42= -lamnda**(-2)*LxDN#+lamnda**(-2)*A*(sp.diags(WestBoundary))/(2*dx)
     J43= sp.diags(np.divide(r,1-h))
     J44= -I
     J45= -I*fhat
     J46= zeros
          
     J51= -lamnda**(-2)*LyN
     J52= zeros
     J53= zeros
     J54= I*fhat
     J55= -I
     J56= sp.diags(np.divide(r,1-h))
     
     J61= zeros
     J62= -lamnda**(-2)*LyN
     J63= -I*fhat
     J64= zeros
     J65= sp.diags(np.divide(r,1-h))
     J66= -I
     
     J=sp.bmat([
                [J11, J12, J13, J14, J15, J16],
                [J21, J22, J23, J24, J25, J26],
                [J31, J32, J33, J34, J35, J36],
                [J41, J42, J43, J44, J45, J46],
                [J51, J52, J53, J54, J55, J56],
                [J61, J62, J63, J64, J65, J66]
                ],format='csr')
     return J
 
J=Jacobian(Uinnitalguess) 

def F2(U):
    zetas,zetac,us,uc,vs,vc=np.array_split(U,6)

    zeros=np.zeros(zetas.shape)
    
    ans=J*U+np.concatenate((zeros,(1-h)*WestBoundary*A/2,zeros,lamnda**(-2)*A*WestBoundary/(2*dx),zeros,zeros))
    return ans



Mu=1.8*10**(-3)
epsilon = 0.15
def FConcentration0(C0):
    Cs,Cc=np.array_split(C0,2)
    

    
    fCs = -Cs+Mu*ADNNN*Cc-Cc
    fCc = -Cc-Mu*ADNNN*Cs+Cs
    
    return np.concatenate((fCs,fCc))


def JacobianConcentration0():
    J11=-I
    J12=Mu*ADNNN-I
    
    J21= -Mu*ADNNN+I
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
    fC2s =+ 1/2*uc*LxDN*Cs+1/2*Cs*LxND*uc+1/2*us*LxDN*Cc+1/2*Cc*LxND*us
    fC2s =+ 1/2*vc*LyN*Cs +1/2*Cs*LyD*vc +1/2*vs*LyN*Cc +1/2*Cc*LyD*vs
    fC2s =+ Mu*ADNNN*C2c
    fC2s =+ 1/epsilon*us*uc-C2c
    
    fC2c = -2*C2c
    fC2c =+ 1/2*uc*LxDN*Cc+1/2*Cc*LxND*uc+1/2*us*LxDN*Cs+1/2*Cs*LxND*us
    fC2c =+ 1/2*vc*LyN*Cc +1/2*Cc*LyD*vc +1/2*vs*LyN*Cs +1/2*Cs*LyD*vs
    fC2c =+ Mu*ADNNN*C2s
    fC2s =+ 1/epsilon*(1/2*us*us+1/2*uc*uc)-C2s
    
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
    epsilon=1
    
    Uiend=np.copy(Uinnitalguess)
    
    i=0
    
    DeltaU=la.spsolve(J,F2(Uiend))
    
    print('\t Newton Rapson loop \n i=0 \t ||delta U|| = %f < %f ' %(MaxNormOfU(DeltaU),epsilon))
    
    if MaxNormOfU(DeltaU)>epsilon:
        Stopcondition=1
    else:
        Stopcondition=0
        Uiend=Uiend-DeltaU
    
    while Stopcondition==1:
        
         DeltaU=la.spsolve(J,F2(Uiend))
         Uiend=Uiend-DeltaU
         i+=1
        
         if MaxNormOfU(DeltaU)>epsilon:
             Stopcondition=1
             print('\t Newton Rapson loop \n i=%i \t ||delta U|| = %f < %f' %(i,MaxNormOfU(DeltaU),epsilon))
         if MaxNormOfU(DeltaU)>10**(20)*epsilon: # this is the fail save if the explodes.
             print('\n \t -----Divergence----- ')
             break 
         if MaxNormOfU(DeltaU)<=epsilon:
             Stopcondition=0
             print('\t Newton Rapson loop \n i=%i \t ||delta U|| = %f < %f' %(i,MaxNormOfU(DeltaU),epsilon))    
         if i>10:
            break
    
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
         if i>10:
            break
    
    return C0iend,C1iend
     

# the run
    
Ufinal=NewtonRapsonInnerloop(Uinnitalguess)
zetas,zetac,us,uc,vs,vc=np.array_split(Ufinal,6)

#checks

C0,C1=NRConcentration(Ufinal,np.concatenate((ICzeta0,ICzeta0)),np.concatenate((ICzeta0,ICzeta0)))

Cs,Cc=np.array_split(C0,2)

C2s,C2c=np.array_split(C1,2)

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
        
        anim = animation.FuncAnimation(fig , animate  , interval=50 , repeat=False)
  
    
def Animation2():
    t = 0
    
    plt.ion()
        
    fig, ((ax1, ax2)) = plt.subplots(2)
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
            
                
        imgzeta.set_array(np.reshape(zeta1,[Ny-1,Nx-1]))
        imgzeta.set_clim(zeta1.min(),zeta1.max())
        
        ax2.clear()
        ax2.quiver(X, Y, np.reshape(u1,[Ny-1,Nx-1]), np.reshape(v1,[Ny-1,Nx-1]),np.reshape(np.sqrt(u1**2+v1**2),[Ny-1,Nx-1]),pivot='mid',units='dots',headwidth=0.1,linewidth=0.1,headlength=0.1)
        #ax2.quiverkey(q, 0.1, 0.1, 0.01)
            
        tlt.set_text('t = %3.3f' %(t))
      
                                                
        return imgzeta,ax2
        
        # figure animation
        
    anim = animation.FuncAnimation(fig , animate  , interval=50 , repeat=True)
    
def Animation3():
    t = 0
    
    plt.ion()
        
    fig, ((ax1)) = plt.subplots(1)
    # Inital conditoin 

    # initialization of the movie figure
    C_anim=0*Cs+1*Cc+0*C2s+1*C2c
    
    Carr = np.reshape( C_anim,[Ny-1,Nx-1])
    
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
                        
        C_anim1=np.sin(t)*Cs+np.cos(t)*Cc+np.sin(2*t)*C2s+np.cos(2*t)*C2c
            
                
        imgC.set_array(np.reshape(C_anim1,[Ny-1,Nx-1]))
        imgC.set_clim(C_anim1.min(),C_anim1.max())
        
        
            
        tlt.set_text('t = %3.3f' %(t))
      
                                                
        return imgC
        
        # figure animation
        
    anim = animation.FuncAnimation(fig , animate  , interval=50 , repeat=True)

Animation2()