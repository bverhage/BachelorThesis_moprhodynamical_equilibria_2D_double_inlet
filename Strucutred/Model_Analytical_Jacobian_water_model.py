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

ones=np.ones((I.shape[0],1))

interior = sp.kron(P.Interior[:,np.newaxis].T,ones).T#+sp.kron(P.Interior[:,np.newaxis].T,ones).T-sp.kron(P.Interior[:,np.newaxis].T,P.Interior[:,np.newaxis])
WestBoudnary = sp.kron(P.WestBoundary[:,np.newaxis].T,ones).T#+sp.kron(P.WestBoundary[:,np.newaxis].T,ones).T-sp.kron(P.WestBoundary[:,np.newaxis].T,P.WestBoundary[:,np.newaxis])
EastBoundary = sp.kron(P.EastBoundary[:,np.newaxis].T,ones).T#+sp.kron(P.EastBoundary[:,np.newaxis].T,ones).T-sp.kron(P.EastBoundary[:,np.newaxis].T,P.EastBoundary[:,np.newaxis])
SouthBoundary = sp.kron(P.SouthBoundary[:,np.newaxis].T,ones).T#+sp.kron(P.SouthBoundary[:,np.newaxis].T,ones).T-sp.kron(P.SouthBoundary[:,np.newaxis].T,P.SouthBoundary[:,np.newaxis])
NorthBoundary = sp.kron(P.NorthBoundary[:,np.newaxis].T,ones).T#+sp.kron(P.NorthBoundary[:,np.newaxis].T,ones).T-sp.kron(P.NorthBoundary[:,np.newaxis].T,P.NorthBoundary[:,np.newaxis])
NWCorner = sp.kron(P.NWCorner[:,np.newaxis].T,ones).T#+sp.kron(P.NWCorner[:,np.newaxis].T,ones).T-sp.kron(P.NWCorner[:,np.newaxis].T,P.NWCorner[:,np.newaxis])
NECorner = sp.kron(P.NECorner[:,np.newaxis].T,ones).T#+sp.kron(P.NECorner[:,np.newaxis].T,ones).T-sp.kron(P.NECorner[:,np.newaxis].T,P.NECorner[:,np.newaxis])
SWCorner = sp.kron(P.SWCorner[:,np.newaxis].T,ones).T#+sp.kron(P.SWCorner[:,np.newaxis].T,ones).T-sp.kron(P.SWCorner[:,np.newaxis].T,P.SWCorner[:,np.newaxis])
SECorner = sp.kron(P.SECorner[:,np.newaxis].T,ones).T#+sp.kron(P.SECorner[:,np.newaxis].T,ones).T-sp.kron(P.SECorner[:,np.newaxis].T,P.SECorner[:,np.newaxis])
zeros=sp.csr_matrix(func.LxD.shape)





def AnalyticalJacobian_zetas(U):
    zetas,zetac,us,uc,vs,vc,C,h=split_animation(U)

    
    ''' zetas '''
    ''' Interior    -zetas-(1-h)*(LxD*uc+LyD*vc)+uc*(LxD*h)+vc*(LyD*h)  '''
    J11 = interior.multiply(    -I      )                                                # zetas,zetas
    J12 = zeros                                                         # zetas,zetac
    J13 = zeros                                                       # zetas, us
    J14 = interior.multiply(    -func.LxD.multiply(1-h)   + sp.diags(func.LxD*h)   )    # zetas, uc
    J15 = zeros                                                         # zetas, vs
    J16 = interior.multiply(    -func.LyD.multiply(1-h)   + sp.diags(func.LyD*h)   )    # zetas, vc
    J17 = zeros                                                         # zetas, C
    J18 = interior.multiply(    sp.diags(func.LxD*uc+func.LyD*vc)    + func.LxD.multiply(uc.T)    + func.LyD.multiply(vc.T)    ) #zetas, h
    ''' West boundary    -zetas + 0 '''
    J11 += WestBoudnary.multiply(    -I   ) # zetas,zetas
    #J12 += zeros       # zetas,zetac
    #J13 += zeros       # zetas, us
    #J14 += zeros       # zetas, uc
    #J15 += zeros       # zetas, vs
    #J16 += zeros       # zetas, vc
    #J17 += WestBoudnary*zeros       # zetas, C
    #J18 += WestBoudnary*zeros       # zetas, h
    ''' East boundary -zetas + P.Atilde*np.sin(P.phi) '''
    J11 += EastBoundary.multiply(    -I      )                                                # zetas,zetas
    #J12 += EastBoundary*zeros                                                         # zetas,zetac
    #J13 += EastBoundary*zeros                                                         # zetas, us
    #J14 += EastBoundary*zeros                                                         # zetas, uc
    #J15 += EastBoundary*zeros                                                         # zetas, vs
    #J16 += EastBoundary*zeros                                                         # zetas, vc
    #J17 += EastBoundary*zeros                                                         # zetas, C
    #J18 += EastBoundary*zeros                                                         # zetas, h
    ''' South boundary -zetas-(1-h)*(LxD*uc+LyD_f*vc*0)+uc*(LxD*h)+0*vc*(LyD_f*h) '''
    J11 += SouthBoundary.multiply(    -I      )                                                # zetas,zetas
    #J12 += SouthBoundary*zeros                                                         # zetas,zetac
    #J13 += SouthBoundary*zeros                                                         # zetas, us
    J14 += SouthBoundary.multiply(    -func.LxD.multiply(1-h)   + sp.diags(func.LxD*h)   )    # zetas, uc
    #J15 += SouthBoundary*zeros                                                         # zetas, vs
    #J16 += SouthBoundary*zeros    # zetas, vc
    #J17 += SouthBoundary*zeros                                                         # zetas, C
    J18 += SouthBoundary.multiply(    sp.diags(func.LxD*uc)    + func.LxD.multiply(uc)        ) #zetas, h
    ''' North boundary -zetas-(1-h)*(LxD*uc+LyD_b*vc*0)+uc*(LxD*h)+0*vc*(LyD_b*h) '''
    J11 += NorthBoundary.multiply(    -I      )                                                # zetas,zetas
    #J12 += NorthBoundary*zeros                                                         # zetas,zetac
    #J13 += NorthBoundary*zeros                                                         # zetas, us
    J14 += NorthBoundary.multiply(    -func.LxD.multiply(1-h)   + sp.diags(func.LxD*h)   )    # zetas, uc
    #J15 += NorthBoundary*zeros                                                         # zetas, vs
    #J16 += NorthBoundary*zeros   # zetas, vc
    #J17 += NorthBoundary*zeros                                                         # zetas, C
    J18 += NorthBoundary.multiply(    sp.diags(func.LxD*uc)    + func.LxD.multiply(uc.T) ) #zetas, h
    ''' Corner points  '''
    J11 += NWCorner.multiply(    -I      )                                                # zetas,zetas
    #J12 += NWCorner*zeros                                                         # zetas,zetac
    #J13 += NWCorner*zeros                                                         # zetas, us
    #J14 += NWCorner*zeros    # zetas, uc
    #J15 += NWCorner*zeros                                                         # zetas, vs
    #J16 += NWCorner*zeros    # zetas, vc
    #J17 += NWCorner*zeros                                                         # zetas, C
    #J18 += NWCorner*zeros #zetas, h
    
    J11 += SWCorner.multiply(    -I      )                                                # zetas,zetas
    #J12 += SWCorner*zeros                                                         # zetas,zetac
    #J13 += SWCorner*zeros                                                         # zetas, us
    #J14 += SWCorner*zeros    # zetas, uc
    #J15 += SWCorner*zeros                                                         # zetas, vs
    #J16 += SWCorner*zeros #(    -func.LyD_f.multiply(1-h.T)   + sp.diags(func.Ly_f*h)   )    # zetas, vc
    #J17 += SWCorner*zeros                                                         # zetas, C
    #J18 += SWCorner*zeros #zetas, h
    
    
    J11 += NECorner.multiply(    -I      )                                                # zetas,zetas
    #J12 += NECorner*zeros                                                         # zetas,zetac
    #J13 += NECorner*zeros                                                         # zetas, us
    # J14 += NECorner*zeros    # zetas, uc
    # J15 += NECorner*zeros                                                         # zetas, vs
    # J16 += NECorner*zeros #(    -func.LyD_f.multiply(1-h.T)   + sp.diags(func.Ly_f*h)   )    # zetas, vc
    # J17 += NECorner*zeros                                                         # zetas, C
    # J18 += NECorner*zeros #zetas, h
    
        
    J11 += SECorner.multiply(    -I      )                                                # zetas,zetas
    # J12 += SECorner*zeros                                                         # zetas,zetac
    # J13 += SECorner*zeros                                                         # zetas, us
    # J14 += SECorner*zeros    # zetas, uc
    # J15 += SECorner*zeros                                                         # zetas, vs
    # J16 += SECorner*zeros #(    -func.LyD_f.multiply(1-h.T)   + sp.diags(func.Ly_f*h)   )    # zetas, vc
    # J17 += SECorner*zeros                                                         # zetas, C
    # J18 += SECorner*zeros #zetas, h
    
    return J11,J12,J13,J14,J15,J16

def AnalyticalJacobian_zetac(U):
    zetas,zetac,us,uc,vs,vc,C,h=split_animation(U)
    
    ''' zetac '''
    ''' Interior     -zetac+(1-h)*(LxD*us+LyD*vs)-us*(LxD*h)-vs*(LyD*h) '''  
    J21 = interior*zeros
    J22 = interior.multiply(    -I      )
    J23 = interior.multiply(     func.LxD.multiply(1-h)    -  sp.diags(func.LxD*h)  )
    J24 = interior*zeros
    J25 = interior.multiply(     func.LyD.multiply(1-h)   -  sp.diags(func.LyD*h)  )
    J26 = interior*zeros
    J27 = interior*zeros
    J28 = interior.multiply(    -sp.diags(func.LxD*us+func.LyD*vs)   -func.LxD.multiply(us) -func.LyD.multiply(vs)     )
    ''' West boundary  -zetac + 1 '''
    # J21 += WestBoudnary*zeros
    J22 += WestBoudnary.multiply(    -I      )
    # J23 += WestBoudnary*zeros
    # J24 += WestBoudnary*zeros
    # J25 += WestBoudnary*zeros
    # J26 += WestBoudnary*zeros
    # J27 += WestBoudnary*zeros
    # J28 += WestBoudnary*zeros
    ''' East boundary    -zetac + P.Atilde*np.cos(P.phi) '''
    # J21 += EastBoundary*zeros
    J22 += EastBoundary.multiply(    -I      )
    # J23 += EastBoundary*zeros
    # J24 += EastBoundary*zeros
    # J25 += EastBoundary*zeros
    # J26 += EastBoundary*zeros
    # J27 += EastBoundary*zeros
    # J28 += EastBoundary*zeros
    ''' South boundary  -zetac+(1-h)*(LxD*us+LyD_f*vs*0)-us*(LxD*h)-0*vs*(LyD_f*h) '''
    J21 += SouthBoundary*zeros
    J22 += SouthBoundary.multiply(    -I      )
    J23 += SouthBoundary.multiply(     func.LxD.multiply(1-h)    -  sp.diags(func.LxD*h)  )
    # J24 += SouthBoundary*zeros
    # J25 += SouthBoundary*zeros #(     func.LyD.multiply(1-h.T)   -  sp.diags(func.Ly*h)  )
    # J26 += SouthBoundary*zeros
    # J27 += SouthBoundary*zeros
    J28 += SouthBoundary.multiply(    -sp.diags(func.LxD*us)   -func.LxD.multiply(us.T)   )
    ''' North boundary    -zetac+(1-h)*(LxD*us+LyD_b*vs*0)-us*(LxD*h)-0*vs*(LyD_b*h) '''
    J21 += NorthBoundary*zeros
    J22 += NorthBoundary.multiply(    -I      )
    J23 += NorthBoundary.multiply(     func.LxD.multiply(1-h)    -  sp.diags(func.LxD*h)  )
    J24 += NorthBoundary*zeros
    J25 += NorthBoundary*zeros #(     func.LyD.multiply(1-h.T)   -  sp.diags(func.Ly*h)  )
    J26 += NorthBoundary*zeros
    J27 += NorthBoundary*zeros
    J28 += NorthBoundary.multiply(    -sp.diags(func.LxD*us)   -func.LxD.multiply(us)    )
    ''' Corner points  '''
    J21 += NWCorner*zeros                                           # zetas,zetas
    J22 += NWCorner.multiply(    -I      )                                                         # zetas,zetac
    J23 += NWCorner*zeros                                                         # zetas, us
    J24 += NWCorner*zeros    # zetas, uc
    J25 += NWCorner*zeros                                                         # zetas, vs
    J26 += NWCorner*zeros    # zetas, vc
    J27 += NWCorner*zeros                                                         # zetas, C
    J28 += NWCorner*zeros #zetas, h
    
    J21 += SWCorner*zeros                                                # zetas,zetas
    J22 += SWCorner.multiply(    -I      )                                                         # zetas,zetac
    J23 += SWCorner*zeros                                                         # zetas, us
    J24 += SWCorner*zeros    # zetas, uc
    J25 += SWCorner*zeros                                                         # zetas, vs
    J26 += SWCorner*zeros #(    -func.LyD_f.multiply(1-h.T)   + sp.diags(func.Ly_f*h)   )    # zetas, vc
    J27 += SWCorner*zeros                                                         # zetas, C
    J28 += SWCorner*zeros #zetas, h
    
    
    J21 += NECorner*zeros                                               # zetas,zetas
    J22 += NECorner.multiply(    -I      )                                                         # zetas,zetac
    J23 += NECorner*zeros                                                         # zetas, us
    J24 += NECorner*zeros    # zetas, uc
    J25 += NECorner*zeros                                                         # zetas, vs
    J26 += NECorner*zeros #(    -func.LyD_f.multiply(1-h.T)   + sp.diags(func.Ly_f*h)   )    # zetas, vc
    J27 += NECorner*zeros                                                         # zetas, C
    J28 += NECorner*zeros #zetas, h
    
        
    J21 += SECorner*zeros                                               # zetas,zetas
    J22 += SECorner.multiply(    -I      )                                                         # zetas,zetac
    J23 += SECorner*zeros                                                         # zetas, us
    J24 += SECorner*zeros    # zetas, uc
    J25 += SECorner*zeros                                                         # zetas, vs
    J26 += SECorner*zeros #(    -func.LyD_f.multiply(1-h.T)   + sp.diags(func.Ly_f*h)   )    # zetas, vc
    J27 += SECorner*zeros                                                         # zetas, C
    J28 += SECorner*zeros #zetas, h
    return J21,J22,J23,J24,J25,J26


def AnalyticalJacobian_us(U):
    zetas,zetac,us,uc,vs,vc,C,h=split_animation(U)
    ''' us '''
    ''' Interior -us -np.divide(P.r, 1-h)*uc-P.lambda_L**(2)*LxD*zetas '''  
    J31 = interior.multiply(    -P.lambda_L**(2)*func.LxD       )
    J32 = interior*zeros
    J33 = interior.multiply(    -I      )
    J34 = interior.multiply(     -sp.diags(np.divide(P.r,1-h))      )
    J35 = interior*zeros
    J36 = interior*zeros
    J37 = interior*zeros
    J38 = interior.multiply(    -sp.diags(P.r*(1-h)**(-2)*uc)       )
    ''' West boundary   -us-np.divide(P.r, 1-h)*uc-P.lambda_L**(2)*(LxD_f*zetas)  '''
    J31 += WestBoudnary.multiply(    -P.lambda_L**(2)*func.LxD_f       )
    J32 += WestBoudnary*zeros
    J33 += WestBoudnary.multiply(    -I      )
    J34 += WestBoudnary.multiply(     -sp.diags(np.divide(P.r,1-h))      )
    J35 += WestBoudnary*zeros
    J36 += WestBoudnary*zeros
    J37 += WestBoudnary*zeros
    J38 += WestBoudnary.multiply(    -sp.diags(P.r*(1-h)**(-2)*uc)       )
    ''' East boundary -us-np.divide(P.r, 1-h)*uc-P.lambda_L**(2)*(LxD_b*zetas)  '''
    J31 += EastBoundary.multiply(    -P.lambda_L**(2)*func.LxD_b       )
    J32 += EastBoundary*zeros
    J33 += EastBoundary.multiply(    -I      )
    J34 += EastBoundary.multiply(     -sp.diags(np.divide(P.r,1-h))      )
    J35 += EastBoundary*zeros
    J36 += EastBoundary*zeros
    J37 += EastBoundary*zeros
    J38 += EastBoundary.multiply(    -sp.diags(P.r*(1-h)**(-2)*uc)       )
    ''' South boundary -us-np.divide(P.r, 1-h)*uc-P.lambda_L**(2)*LxD*zetas '''
    J31 += SouthBoundary.multiply(    -P.lambda_L**(2)*func.LxD       )
    J32 += SouthBoundary*zeros
    J33 += SouthBoundary.multiply(    -I      )
    J34 += SouthBoundary.multiply(     -sp.diags(np.divide(P.r,1-h))      )
    J35 += SouthBoundary*zeros
    J36 += SouthBoundary*zeros
    J37 += SouthBoundary*zeros
    J38 += SouthBoundary.multiply(    -sp.diags(P.r*(1-h)**(-2)*uc)       )
    ''' North boundary -us-np.divide(P.r, 1-h)*uc-P.lambda_L**(2)*LxD*zetas '''
    J31 += NorthBoundary.multiply(    -P.lambda_L**(2)*func.LxD       )
    J32 += NorthBoundary*zeros
    J33 += NorthBoundary.multiply(    -I      )
    J34 += NorthBoundary.multiply(     -sp.diags(np.divide(P.r,1-h))      )
    J35 += NorthBoundary*zeros
    J36 += NorthBoundary*zeros
    J37 += NorthBoundary*zeros
    J38 += NorthBoundary.multiply(    -sp.diags(P.r*(1-h)**(-2)*uc)       )
    ''' Corner points  '''
    J31 += NWCorner.multiply(    -P.lambda_L**(2)*func.LxD_f      )
    J32 += NWCorner*zeros
    J33 += NWCorner.multiply(    -I      )
    J34 += NWCorner.multiply(     -sp.diags(np.divide(P.r,1-h))      )
    J35 += NWCorner*zeros
    J36 += NWCorner*zeros
    J37 += NWCorner*zeros
    J38 += NWCorner.multiply(    -sp.diags(P.r*(1-h)**(-2)*uc)       )
    J31 += SWCorner.multiply(    -P.lambda_L**(2)*func.LxD_f      )
    J32 += SWCorner*zeros
    J33 += SWCorner.multiply(    -I      )
    J34 += SWCorner.multiply(     -sp.diags(np.divide(P.r,1-h))      )
    J35 += SWCorner*zeros
    J36 += SWCorner*zeros
    J37 += SWCorner*zeros
    J38 += SWCorner.multiply(    -sp.diags(P.r*(1-h)**(-2)*uc)       )
    
    J31 += NECorner.multiply(    -P.lambda_L**(2)*func.LxD_b      )
    J32 += NECorner*zeros
    J33 += NECorner.multiply(    -I      )
    J34 += NECorner.multiply(     -sp.diags(np.divide(P.r,1-h))      )
    J35 += NECorner*zeros
    J36 += NECorner*zeros
    J37 += NECorner*zeros
    J38 += NECorner.multiply(    -sp.diags(P.r*(1-h)**(-2)*uc)       )
    J31 += SECorner.multiply(    -P.lambda_L**(2)*func.LxD_b     )
    J32 += SECorner*zeros
    J33 += SECorner.multiply(    -I      )
    J34 += SECorner.multiply(     -sp.diags(np.divide(P.r,1-h))      )
    J35 += SECorner*zeros
    J36 += SECorner*zeros
    J37 += SECorner*zeros
    J38 += SECorner.multiply(    -sp.diags(P.r*(1-h)**(-2)*uc)       )
    return J31,J32,J33,J34,J35,J36

def AnalyticalJacobian_uc(U):
    zetas,zetac,us,uc,vs,vc,C,h=split_animation(U)
    ''' uc '''
    ''' Interior -uc  +np.divide(P.r, 1-h)*us +P.lambda_L**(2)*LxD*zetac ''' 
    J41 = interior*zeros
    J42 = interior.multiply(        P.lambda_L**(2)*func.LxD       )
    J43 = interior.multiply(        sp.diags(np.divide(P.r,1-h))    )
    J44 = interior.multiply(        -I      )
    J45 = interior*zeros
    J46 = interior*zeros
    J47 = interior*zeros
    J48 = interior.multiply(        sp.diags(P.r*(1-h)**(-2)*us)     )
    ''' West boundary -uc+np.divide(P.r, 1-h)*us+P.lambda_L**(2)*LxD_f*zetac '''
    J41 += WestBoudnary*zeros
    J42 += WestBoudnary.multiply(        P.lambda_L**(2)*func.LxD_f       )
    J43 += WestBoudnary.multiply(        sp.diags(np.divide(P.r,1-h))    )
    J44 += WestBoudnary.multiply(        -I      )
    J45 += WestBoudnary*zeros
    J46 += WestBoudnary*zeros
    J47 += WestBoudnary*zeros
    J48 += WestBoudnary.multiply(        sp.diags(P.r*(1-h)**(-2)*us)     )
    ''' East boundary  -uc+np.divide(P.r, 1-h)*us+P.lambda_L**(2)*LxD_b*zetac  '''
    J41 += EastBoundary*zeros
    J42 += EastBoundary.multiply(        P.lambda_L**(2)*func.LxD_b       )
    J43 += EastBoundary.multiply(        sp.diags(np.divide(P.r,1-h))    )
    J44 += EastBoundary.multiply(        -I      )
    J45 += EastBoundary*zeros
    J46 += EastBoundary*zeros
    J47 += EastBoundary*zeros
    J48 += EastBoundary.multiply(        sp.diags(P.r*(1-h)**(-2)*us)     )
    ''' South boundary   -uc+np.divide(P.r, 1-h)*us+P.lambda_L**(2)*LxD*zetac  '''
    J41 += SouthBoundary*zeros
    J42 += SouthBoundary.multiply(        P.lambda_L**(2)*func.LxD       )
    J43 += SouthBoundary.multiply(        sp.diags(np.divide(P.r,1-h))    )
    J44 += SouthBoundary.multiply(        -I      )
    J45 += SouthBoundary*zeros
    J46 += SouthBoundary*zeros
    J47 += SouthBoundary*zeros
    J48 += SouthBoundary.multiply(        sp.diags(P.r*(1-h)**(-2)*us)     )
    ''' North boundary -uc+np.divide(P.r, 1-h)*us+P.lambda_L**(2)*LxD*zetac  '''
    J41 += NorthBoundary*zeros
    J42 += NorthBoundary.multiply(        P.lambda_L**(2)*func.LxD       )
    J43 += NorthBoundary.multiply(        sp.diags(np.divide(P.r,1-h))    )
    J44 += NorthBoundary.multiply(        -I      )
    J45 += NorthBoundary*zeros
    J46 += NorthBoundary*zeros
    J47 += NorthBoundary*zeros
    J48 += NorthBoundary.multiply(        sp.diags(P.r*(1-h)**(-2)*us)     )
    ''' Corner points  '''
    J41 += NWCorner*zeros
    J42 += NWCorner.multiply(        P.lambda_L**(2)*func.LxD_f       )
    J43 += NWCorner.multiply(        sp.diags(np.divide(P.r,1-h))    )
    J44 += NWCorner.multiply(        -I      )
    J45 += NWCorner*zeros
    J46 += NWCorner*zeros
    J47 += NWCorner*zeros
    J48 += NWCorner.multiply(        sp.diags(P.r*(1-h)**(-2)*us)     )
    J41 += SWCorner*zeros
    J42 += SWCorner.multiply(        P.lambda_L**(2)*func.LxD_f       )
    J43 += SWCorner.multiply(        sp.diags(np.divide(P.r,1-h))    )
    J44 += SWCorner.multiply(        -I      )
    J45 += SWCorner*zeros
    J46 += SWCorner*zeros
    J47 += SWCorner*zeros
    J48 += SWCorner.multiply(        sp.diags(P.r*(1-h)**(-2)*us)     )
    
    J41 += NECorner*zeros
    J42 += NECorner.multiply(        P.lambda_L**(2)*func.LxD_b       )
    J43 += NECorner.multiply(        sp.diags(np.divide(P.r,1-h))    )
    J44 += NECorner.multiply(        -I      )
    J45 += NECorner*zeros
    J46 += NECorner*zeros
    J47 += NECorner*zeros
    J48 += NECorner.multiply(        sp.diags(P.r*(1-h)**(-2)*us)     )
    J41 += SECorner*zeros
    J42 += SECorner.multiply(        P.lambda_L**(2)*func.LxD_b       )
    J43 += SECorner.multiply(        sp.diags(np.divide(P.r,1-h))    )
    J44 += SECorner.multiply(        -I      )
    J45 += SECorner*zeros
    J46 += SECorner*zeros
    J47 += SECorner*zeros
    J48 += SECorner.multiply(        sp.diags(P.r*(1-h)**(-2)*us)     )
    return J41,J42,J43,J44,J45,J46

def AnalyticalJacobian_vs(U):
    zetas,zetac,us,uc,vs,vc,C,h=split_animation(U)
    ''' vs '''
    ''' Interior -vs  -np.divide(P.r, 1-h)*vc -P.lambda_L**(2)*LyD*zetas ''' 
    J51 = interior.multiply(    -P.lambda_L**(2)*func.LyD      )
    J52 = interior*zeros
    J53 = interior*zeros
    J54 = interior*zeros
    J55 = interior.multiply(    -I      )
    J56 = interior.multiply(    -sp.diags(np.divide(P.r,1-h))       )
    J57 = interior*zeros
    J58 = interior.multiply(    -sp.diags(P.r*(1-h)**(-2)*vc)       )
    ''' West boundary -vc+np.divide(P.r,1-h)*vs+P.lambda_L**(2)*LyD*zetac '''
    J51 += WestBoudnary.multiply(    -P.lambda_L**(2)*func.LyD      )
    J52 += WestBoudnary*zeros
    J53 += WestBoudnary*zeros
    J54 += WestBoudnary*zeros
    J55 += WestBoudnary.multiply(    -I      )
    J56 += WestBoudnary.multiply(    -sp.diags(np.divide(P.r,1-h))       )
    J57 += WestBoudnary*zeros
    J58 += WestBoudnary.multiply(    -sp.diags(P.r*(1-h)**(-2)*vc)       )
    ''' East boundary  '''
    J51 += EastBoundary.multiply(    -P.lambda_L**(2)*func.LyD      )
    J52 += EastBoundary*zeros
    J53 += EastBoundary*zeros
    J54 += EastBoundary*zeros
    J55 += EastBoundary.multiply(    -I      )
    J56 += EastBoundary.multiply(    -sp.diags(np.divide(P.r,1-h))       )
    J57 += EastBoundary*zeros
    J58 += EastBoundary.multiply(    -sp.diags(P.r*(1-h)**(-2)*vc)       )
    ''' South boundary  -vc+0 '''
    J51 += SouthBoundary*zeros
    J52 += SouthBoundary*zeros
    J53 += SouthBoundary*zeros
    J54 += SouthBoundary*zeros
    J55 += SouthBoundary.multiply(    -I      )
    J56 += SouthBoundary*zeros
    J57 += SouthBoundary*zeros
    J58 += SouthBoundary*zeros
    ''' North boundary -vc+0 '''
    J51 += NorthBoundary*zeros
    J52 += NorthBoundary*zeros
    J53 += NorthBoundary*zeros
    J54 += NorthBoundary*zeros
    J55 += NorthBoundary.multiply(    -I      )
    J56 += NorthBoundary*zeros
    J57 += NorthBoundary*zeros
    J58 += NorthBoundary*zeros
    ''' Corner points North -vc+P.fhat*us+np.divide(P.r,1-h)*vs+P.lambda_L**(2)*LyD_b*zetac '''
    J51 += NWCorner.multiply(    -P.lambda_L**(2)*func.LyD_b      )
    J52 += NWCorner*zeros
    J53 += NWCorner*zeros
    J54 += NWCorner*zeros
    J55 += NWCorner.multiply(    -I      )
    J56 += NWCorner.multiply(    -sp.diags(np.divide(P.r,1-h))       )
    J57 += NWCorner*zeros
    J58 += NWCorner.multiply(    -sp.diags(P.r*(1-h)**(-2)*vc)       )
    J51 += SWCorner.multiply(    -P.lambda_L**(2)*func.LyD_f      )
    J52 += SWCorner*zeros
    J53 += SWCorner*zeros
    J54 += SWCorner*zeros
    J55 += SWCorner.multiply(    -I      )
    J56 += SWCorner.multiply(    -sp.diags(np.divide(P.r,1-h))       )
    J57 += SWCorner*zeros
    J58 += SWCorner.multiply(    -sp.diags(P.r*(1-h)**(-2)*vc)       )
    
    J51 += NECorner.multiply(    -P.lambda_L**(2)*func.LyD_b      )
    J52 += NECorner*zeros
    J53 += NECorner*zeros
    J54 += NECorner*zeros
    J55 += NECorner.multiply(    -I      )
    J56 += NECorner.multiply(    -sp.diags(np.divide(P.r,1-h))       )
    J57 += NECorner*zeros
    J58 += NECorner.multiply(    -sp.diags(P.r*(1-h)**(-2)*vc)       )
    J51 += SECorner.multiply(    -P.lambda_L**(2)*func.LyD_f      )
    J52 += SECorner*zeros
    J53 += SECorner*zeros
    J54 += SECorner*zeros
    J55 += SECorner.multiply(    -I      )
    J56 += SECorner.multiply(    -sp.diags(np.divide(P.r,1-h))       )
    J57 += SECorner*zeros
    J58 += SECorner.multiply(    -sp.diags(P.r*(1-h)**(-2)*vc)       )
    return J51,J52,J53,J54,J55,J56


def AnalyticalJacobian_vc(U):
    zetas,zetac,us,uc,vs,vc,C,h=split_animation(U)
    ''' vc '''
    ''' Interior -vc +P.fhat*us   +np.divide(P.r, 1-h)*vs +P.lambda_L**(2)*LyD*zetac ''' 
    J61 = interior*zeros
    J62 = interior.multiply(        P.lambda_L**(2)*func.LyD       )
    J63 = interior*zeros
    J64 = interior*zeros
    J65 = interior.multiply(        sp.diags(np.divide(P.r,1-h))    )
    J66 = interior.multiply(        -I      )
    J67 = interior*zeros
    J68 = interior.multiply(        sp.diags(P.r*(1-h)**(-2)*vs)    )
    ''' West boundary  '''
    J61 += WestBoudnary*zeros
    J62 += WestBoudnary.multiply(        P.lambda_L**(2)*func.LyD       )
    J63 += WestBoudnary*zeros
    J64 += WestBoudnary*zeros
    J65 += WestBoudnary.multiply(        sp.diags(np.divide(P.r,1-h))    )
    J66 += WestBoudnary.multiply(        -I      )
    J67 += WestBoudnary*zeros
    J68 += WestBoudnary.multiply(        sp.diags(P.r*(1-h)**(-2)*vs)    )
    ''' East boundary  '''
    J61 += EastBoundary*zeros
    J62 += EastBoundary.multiply(        P.lambda_L**(2)*func.LyD       )
    J63 += EastBoundary*zeros
    J64 += EastBoundary*zeros
    J65 += EastBoundary.multiply(        sp.diags(np.divide(P.r,1-h))    )
    J66 += EastBoundary.multiply(        -I      )
    J67 += EastBoundary*zeros
    J68 += EastBoundary.multiply(        sp.diags(P.r*(1-h)**(-2)*vs)    )
    ''' South boundary '''
    J61 += SouthBoundary*zeros
    J62 += SouthBoundary*zeros
    J63 += SouthBoundary*zeros
    J64 += SouthBoundary*zeros
    J65 += SouthBoundary*zeros
    J66 += SouthBoundary.multiply(        -I      )
    J67 += SouthBoundary*zeros
    J68 += SouthBoundary*zeros
    ''' North boundary '''
    J61 += NorthBoundary*zeros
    J62 += NorthBoundary*zeros
    J63 += NorthBoundary*zeros
    J64 += NorthBoundary*zeros
    J65 += NorthBoundary*zeros
    J66 += NorthBoundary.multiply(        -I      )
    J67 += NorthBoundary*zeros
    J68 += NorthBoundary*zeros
    ''' Corner points  '''
    J61 += (NWCorner+NECorner)*zeros
    J62 += (NWCorner+NECorner).multiply(        P.lambda_L**(2)*func.LyD_b       )
    J63 += (NWCorner+NECorner)*zeros
    J64 += (NWCorner+NECorner)*zeros
    J65 += (NWCorner+NECorner).multiply(        sp.diags(np.divide(P.r,1-h))    )
    J66 += (NWCorner+NECorner).multiply(        -I      )
    J67 += (NWCorner+NECorner)*zeros
    J68 += (NWCorner+NECorner).multiply(        sp.diags(P.r*(1-h)**(-2)*vs)    )
    
    J61 += (SWCorner+SECorner)*zeros
    J62 += (SWCorner+SECorner).multiply(        P.lambda_L**(2)*func.LyD_f       )
    J63 += (SWCorner+SECorner)*zeros
    J64 += (SWCorner+SECorner)*zeros
    J65 += (SWCorner+SECorner).multiply(        sp.diags(np.divide(P.r,1-h))    )
    J66 += (SWCorner+SECorner).multiply(        -I      )
    J67 += (SWCorner+SECorner)*zeros
    J68 += (SWCorner+SECorner).multiply(        sp.diags(P.r*(1-h)**(-2)*vs)    )
    return J61,J62,J63,J64,J65,J66




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


