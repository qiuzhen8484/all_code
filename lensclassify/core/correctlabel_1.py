#coding:UTF-8
'''
Created on 2018年10月22日
@author: qiuzhen
'''

import numpy as np
import scipy.io as scio
import os

def Matrix_to_array(data_path):
    data=scio.loadmat(data_path)
    left=data.get('left')[0][0]
    right=data.get('right')[0][0]
    top=data.get('top')[0][0]
    NucleusBoundary=np.array(data.get('NucleusBoundary'))
    LensBoundary2=np.array(data.get('LensBoundary2'))
    LensBoundary=np.array(data.get('LensBoundary'))

    #get lens
    row,col=LensBoundary2.shape[:]
    l_x=np.arange(left+1,right+1,1)
    l_f=np.zeros(len(l_x))
    l_b=np.zeros(len(l_x))
    for i in range(col):
        A1=LensBoundary2[:,i]
        lo_A1=np.where(A1==1)[0]
        if len(lo_A1)==0:
            break
        if len(lo_A1)>2:
            if i==0:
                l_f[i]=lo_A1[0]+top
                l_b[i]=lo_A1[-1]+top
            else:
                disl_f=np.zeros(len(lo_A1))
                disl_b=np.zeros(len(lo_A1))
                for j in range(len(lo_A1)):
                    disl_f[j]=abs(lo_A1[j]-l_f[i-1]+top)
                    disl_b[j]=abs(lo_A1[j]-l_b[i-1]+top)
                locl_f=np.where(disl_f==np.min(disl_f))[0][0]
                locl_b=np.where(disl_b==np.min(disl_b))[0][0]
                l_f[i]=lo_A1[locl_f]+top
                l_b[i]=lo_A1[locl_b]+top
        else:
            l_f[i]=lo_A1[0]+top
            l_b[i]=lo_A1[-1]+top

     #get cortex
    row,col=LensBoundary.shape[:]
    c_x=np.arange(left+1,right+1,1)
    c_f=np.zeros(len(c_x))
    c_b=np.zeros(len(c_x))
    for i in range(col):
        A1=LensBoundary[:,i]
        lo_A1=np.where(A1==1)[0]
        if len(lo_A1)==0:
            break
        if len(lo_A1)>2:
            if i==0:
                c_f[i]=lo_A1[0]+top
                c_b[i]=lo_A1[-1]+top
            else:
                disc_f=np.zeros(len(lo_A1))
                disc_b=np.zeros(len(lo_A1))
                for j in range(len(lo_A1)):
                    disc_f[j]=abs(lo_A1[j]-c_f[i-1]+top)
                    disc_b[j]=abs(lo_A1[j]-c_b[i-1]+top)
                locc_f=np.where(disc_f==np.min(disc_f))[0][0]
                locc_b=np.where(disc_b==np.min(disc_b))[0][0]
                c_f[i]=lo_A1[locc_f]+top
                c_b[i]=lo_A1[locc_b]+top
        else:
            c_f[i]=lo_A1[0]+top
            c_b[i]=lo_A1[-1]+top


    #get nucleus
    row,col=NucleusBoundary.shape[:]
    for i in range(col):
        A1=NucleusBoundary[:,i]
        lo_A1=np.where(A1==1)[0]
        if len(lo_A1)==0:
            continue
        else:
            A1=NucleusBoundary[:,i+1]
            lo_A1=np.where(A1==1)[0]
            if len(lo_A1)==0:
                continue
            else:
                nleft=left+i
                break

    for i in range(nleft-left,col):
        A1=NucleusBoundary[:,i]
        lo_A1=np.where(A1==1)[0]
        if len(lo_A1)==0:
            A1=NucleusBoundary[:,i+1]
            lo_A1=np.where(A1==1)[0]
            if len(lo_A1):
                nright=left+i+1
                break
        if i==col-1:
            nright=left+col

    n_x=np.arange(nleft+1,nright+1,1)
    n_f=np.zeros(len(n_x))
    n_b=np.zeros(len(n_x))
    for i in range(nleft-left,nright-left):
        A1=NucleusBoundary[:,i]
        lo_A1=np.where(A1==1)[0]
        if len(lo_A1)==0:
            continue
        if len(lo_A1)>2:
            if (i-nleft+left)==0:
                n_f[i-nleft+left]=lo_A1[0]+top
                n_b[i-nleft+left]=lo_A1[-1]+top
            else:
                disn_f=np.zeros(len(lo_A1))
                disn_b=np.zeros(len(lo_A1))
                for j in range(len(lo_A1)):
                    disn_f[j]=abs(lo_A1[j]-n_f[i-nleft+left-1]+top)
                    disn_b[j]=abs(lo_A1[j]-n_b[i-nleft+left-1]+top)
                locn_f=np.where(disn_f==np.min(disn_f))[0][0]
                locn_b=np.where(disn_b==np.min(disn_b))[0][0]
                n_f[i-nleft+left]=lo_A1[locn_f]+top
                n_b[i-nleft+left]=lo_A1[locn_b]+top
        else:
            n_f[i-nleft+left]=lo_A1[0]+top
            n_b[i-nleft+left]=lo_A1[-1]+top


    labeldictionary={'l_x':l_x,'l_f':l_f,'l_b':l_b,'c_x':c_x,'c_f':c_f,'c_b':c_b,'n_x':n_x,'n_f':n_f,'n_b':n_b,'datapath':data_path}
    return labeldictionary


def Delete_mistake_point(labeldictionary):
    l_x=labeldictionary['l_x']
    l_f=labeldictionary['l_f']
    l_b=labeldictionary['l_b']
    c_x=labeldictionary['c_x']
    c_f=labeldictionary['c_f']
    c_b=labeldictionary['c_b']
    n_x=labeldictionary['n_x']
    n_f=labeldictionary['n_f']
    n_b=labeldictionary['n_b']
    l_unit=len(l_x)//5
    n_unit=len(n_x)//5
    random_int1=np.random.randint(0,l_unit)
    random_int2=np.random.randint(l_unit,2*l_unit)
    random_int3=np.random.randint(2*l_unit,3*l_unit)
    random_int4=np.random.randint(3*l_unit,4*l_unit)
    random_int5=np.random.randint(4*l_unit,5*l_unit)
    random_int=np.array([random_int1,random_int2,random_int3,random_int4,random_int5])
    samplel_x=l_x[random_int]
    samplec_x=c_x[random_int]
    samplel_f=l_f[random_int]
    samplel_b=l_b[random_int]
    samplec_f=c_f[random_int]
    samplec_b=c_b[random_int]

    random_int1=np.random.randint(0,n_unit)
    random_int2=np.random.randint(n_unit,2*n_unit)
    random_int3=np.random.randint(2*n_unit,3*n_unit)
    random_int4=np.random.randint(3*n_unit,4*n_unit)
    random_int5=np.random.randint(4*n_unit,5*n_unit)
    random_int=np.array([random_int1,random_int2,random_int3,random_int4,random_int5])
    samplen_x=n_x[random_int]
    samplen_f=n_f[random_int]
    samplen_b=n_b[random_int]

    #fitting
    r1=np.polyfit(samplel_x,samplel_f,2)
    l_f1=np.polyval(r1,l_x)
    r2=np.polyfit(samplel_x,samplel_b,2)
    l_b1=np.polyval(r2,l_x)
    r3=np.polyfit(samplec_x,samplec_f,2)
    c_f1=np.polyval(r3,c_x)
    r4=np.polyfit(samplec_x,samplec_b,2)
    c_b1=np.polyval(r4,c_x)

    for i in range(len(l_x)):
        if abs(l_f1[i]-l_f[i])>4:
            l_f[i]=0
        if abs(l_b1[i]-l_b[i])>4:
            l_b[i]=0
        if abs(c_f1[i]-c_f[i])>4:
            c_f[i]=0
        if abs(c_b1[i]-c_b[i])>4:
            c_b[i]=0


    r5=np.polyfit(samplen_x,samplen_f,2)
    n_f1=np.polyval(r5,n_x)
    r6=np.polyfit(samplen_x,samplen_b,2)
    n_b1=np.polyval(r6,n_x)
    for i in range(len(n_x)):
        if abs(n_f1[i]-n_f[i])>4:
            n_f[i]=0
        if abs(n_b1[i]-n_b[i])>4:
            n_b[i]=0

    labeldictionary={'l_x':l_x,'l_f':l_f,'l_b':l_b,'c_x':c_x,'c_f':c_f,'c_b':c_b,'n_x':n_x,'n_f':n_f,'n_b':n_b,'datapath':labeldictionary['datapath']}
    return labeldictionary


def Anew_fitting(labeldictionary):
    l_x=labeldictionary['l_x']
    l_f=labeldictionary['l_f']
    l_b=labeldictionary['l_b']
    c_x=labeldictionary['c_x']
    c_f=labeldictionary['c_f']
    c_b=labeldictionary['c_b']
    n_x=labeldictionary['n_x']
    n_f=labeldictionary['n_f']
    n_b=labeldictionary['n_b']
    l_fx=l_x
    l_bx=l_x
    c_fx=c_x
    c_bx=c_x
    n_fx=n_x
    n_bx=n_x

    #lens
    A_lf_x=l_x
    A_lb_x=l_x
    bdel=np.where(l_b==0)[0]
    fdel=np.where(l_f==0)[0]
    l_f=np.delete(l_f,fdel,0)
    l_b=np.delete(l_b,bdel,0)
    l_bx=np.delete(l_bx,bdel,0)
    l_fx=np.delete(l_fx,fdel,0)
    r1=np.polyfit(l_fx,l_f,4)
    A_lf_y=np.polyval(r1,A_lf_x)
    r2=np.polyfit(l_bx,l_b,4)
    A_lb_y=np.polyval(r2,A_lb_x)


    #cortex
    A_cf_x=c_x
    A_cb_x=c_x
    bdel=np.where(c_b==0)[0]
    fdel=np.where(c_f==0)[0]
    c_f=np.delete(c_f,fdel,0)
    c_b=np.delete(c_b,bdel,0)
    c_bx=np.delete(c_bx,bdel,0)
    c_fx=np.delete(c_fx,fdel,0)
    r3=np.polyfit(c_fx,c_f,4)
    A_cf_y=np.polyval(r3,A_cf_x)
    r4=np.polyfit(c_bx,c_b,4)
    A_cb_y=np.polyval(r4,A_cb_x)


    #nucleus
    A_nf_x=n_x
    A_nb_x=n_x
    bdel=np.where(n_b==0)[0]
    fdel=np.where(n_f==0)[0]
    n_f=np.delete(n_f,fdel,0)
    n_b=np.delete(n_b,bdel,0)
    n_bx=np.delete(n_bx,bdel,0)
    n_fx=np.delete(n_fx,fdel,0)
    r5=np.polyfit(n_fx,n_f,3)
    A_nf_y=np.polyval(r5,A_nf_x)
    r6=np.polyfit(n_bx,n_b,3)
    A_nb_y=np.polyval(r6,A_nb_x)

    data_path=labeldictionary['datapath'][:-4]+'_1.mat'
    label={'A_lb_x':A_lb_x,'A_lb_y':A_lb_y,'A_lf_x':A_lf_x,'A_lf_y':A_lf_y,'A_cb_x':A_cb_x,'A_cb_y':A_cb_y,'A_cf_x':A_cf_x,'A_cf_y':A_cf_y,
           'A_nb_x':A_nb_x,'A_nb_y':A_nb_y,'A_nf_x':A_nf_x,'A_nf_y':A_nf_y}
    scio.savemat(data_path,label)

Anew_fitting(Delete_mistake_point(Matrix_to_array('/home/intern1/qiuzhen/Works/test/1056_20180215_091107_R_CASIA2_001_000.mat')))