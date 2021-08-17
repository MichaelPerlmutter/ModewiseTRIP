# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 08:30:48 2021

@author: micha
"""

#########################
# IMPORTS
#########################

import torch
import torch.nn.functional as F

import tensorly as tl
from tensorly import decomposition

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from PIL import Image

from scipy.linalg import dft

import timeit
import pandas as pd
from datetime import datetime



##########################
# AUXILIARY FUCNTIONS
##########################

def random_low_rank(n,r):
    #torch.manual_seed(0)
    #np.random.seed(0)
    C=np.random.normal(0,1,size=r)
    C=tl.tensor(C)
    C.shape
    X=C

    U=[]
    for i in range(len(n)):
        M=np.random.normal(0,1,size=(n[i],n[i]))
        u,sigma,v=np.linalg.svd(M)
        U.append(u[:,0:r[i]])

    for i in range(len(n)):
        X=tl.tenalg.mode_dot(X,U[i],i)
    return X

def low_rank_approx(tensor,r):
    #torch.manual_seed(0)

    core, factors = tl.decomposition.tucker(tensor.numpy(), r)
    answer = torch.tensor(tl.tucker_to_tensor([core, factors]))
    return answer
    
def vectorize(X):
    x=X.numpy()
    x=x.reshape(-1)
    return x

def tensorize(x,n):
    return torch.tensor(x.reshape(n))

def modewise_measurements(cur_tensor, measurements):
    cur_tensor_array = cur_tensor.numpy()
    cur_tensor_array = tl.tenalg.multi_mode_dot(cur_tensor_array, measurements)
    return torch.tensor(cur_tensor_array)

def matrix_modewise_measurements(cur_tensor, measurements):
    cur_tensor_array = cur_tensor.numpy()
    int_array = np.matmul(measurements[0], cur_tensor_array)
    cur_tensor_array = np.matmul(int_array, measurements[1].T)
    return torch.tensor(cur_tensor_array)

def two_step_measurements_original(X,A,Afinal):
    return np.matmul(Afinal,vectorize(matrix_modewise_measurements(X, A)))

def two_step_lift(y, A, AT, Afinal, mintermediate, n):
    ybig=np.matmul(Afinal.conj().T, y)
    Ybig=tensorize(ybig, mintermediate)
    Xpullback= matrix_modewise_measurements(Ybig, AT)
    return torch.reshape(Xpullback, n)

def relative_error(true,guess,first_loss):
    return (np.linalg.norm(true - guess)/first_loss)

def create_kfjl_meas(dim, k):
    if dim<k:
        raise ValueError("dim is less than k, matrix needs to be tall and skinny")
        
    #np.random.seed(0)
    m=dft(dim)/np.sqrt(dim)
    vec=np.random.choice([-1,1],dim)
    m = np.matmul(m, np.diag(vec))
    m = np.sqrt(dim/k)*m[:int(k), :]
    return m

def create_gaussian_meas(dim, k):
    np.random.seed(0)
    return np.sqrt(1/k)*np.random.normal(0.0, 1.0, [k, dim])

def reshaped_dimension(n1,d):
    n=n1*np.ones(d,dtype=int)
    nn=((n[:d//2]).prod(),(n[d//2:]).prod())
    n = tuple(n)
    return n,nn

def reshaped_rank(r1,d):
    r=r1*np.ones(d,dtype=int)
    rp=((r[:d//2]).prod(),(r[d//2:]).prod())
    return r,rp

####################################################
# Generate Random Tensors
####################################################

def generate_tensors(n1,d,r1,num_samples):

    #ensure same data every time
    #torch.manual_seed(0)
    
    #reshaped dimensions
    n,nn=reshaped_dimension(n1,d)
    r,rp=reshaped_rank(r1,d)

    #Create random low rank tensor and reshape it
    XX = []
    reshaped_XX = []
    for j in range(num_samples):
        X=torch.tensor(random_low_rank(n, r), dtype=torch.float64)
        reshaped_X=torch.reshape(X, nn)
        XX.append(X)
        reshaped_XX.append(reshaped_X)
        
    return XX, reshaped_XX

##################################################
#Modewise Fourier
##################################################
def mw_measurements(reshaped_XX,m1,num_samples,meas="Fourier"):

    ## Re-running this window with different compression ratios resamples measurement matrices only
    #torch.manual_seed(0)
  
    nn=tuple(reshaped_XX[0].shape)
    
    # Compute and print out intermediate dimensions
    mi=np.array([m1 for i in range(len(nn))])
    m_first=mi.prod()
    m_second = m_first

    start = timeit.default_timer()
    #Compute Fourier operators
    if meas=="Fourier":
        A = [create_kfjl_meas(nn[i], mi[i]) for i in range(len(nn))]
    elif meas=="Gaussian":
        A = [create_gaussian_meas(nn[i], mi[i]) for i in range(len(nn))]
    else:
        raise ValueError("Set meas to either 'Fourier' or 'Gaussian'")
    
    AT=[A[i].conj().T for i in range(len(A))]

    yy = []
    # Compute measurements
    for j in range(num_samples):
        y = matrix_modewise_measurements(reshaped_XX[j], A)
        yy.append(y)
    stop = timeit.default_timer()
    
    #record time
    avg_meas_time=(stop - start)/num_samples 
    
    return A,AT,yy, avg_meas_time  

def MWTIHT(n1,d,r1,m1,num_samples=100,meas="Fourier",mu=.1,N_iter=1000,accuracy=.001):
    

    #parameters
    #mu, N_iter =.1, 1000
    #accuracy = 0.001
    good_runs = 0
    total_time = 0
    total_iters = 0
    r,rp=reshaped_rank(r1,d)
    n,nn=reshaped_dimension(n1,d)
    
    #generate tensors and measurements
    torch.manual_seed(0)
    X0 = torch.randn(n)
    XX, reshaped_XX = generate_tensors(n1,d,r1,num_samples)
    A,AT,yy,time=mw_measurements(reshaped_XX,m1,num_samples,meas)
    
    Losses=[[1] for _ in range(num_samples)]
    # Run recovery algorithm
    for j in range(num_samples):
        #print(j)
        start = timeit.default_timer()
        X_iter=torch.clone(X0)
        first_loss = np.linalg.norm(XX[j] - X_iter)
        i = 0
        while Losses[j][-1] > accuracy and i < N_iter:
            i += 1 
            Losses[j].append(relative_error(true=XX[j], guess=X_iter, first_loss=first_loss))
            X_iter_reshaped=torch.reshape(X_iter, nn)
            first_step = matrix_modewise_measurements(X_iter_reshaped, A)
            Z = yy[j] - first_step

            Z = torch.reshape(matrix_modewise_measurements(Z, AT), n)    
            Y_iter=X_iter+mu*Z
    
            X_iter=low_rank_approx(Y_iter, r)
        stop = timeit.default_timer()
        #plt.plot(range(len(Losses[j])), Losses[j])   
        #if i == N_iter:
        #    print("Not converged")
        if i<N_iter:
            good_runs += 1
            total_time += stop - start
            total_iters += i
            #print("Converged!")
            #print('Number of iterations: ', i)
            #print('Final loss: ', Losses[j][-1])
    if good_runs != 0:
        Convergence_percent=100*good_runs/num_samples
        #print('\n')
        #print('Percentage of converged runs:', 100*good_runs/num_samples)
        #print('Average recovery time: ', total_time/good_runs)
        Average_recovery_time= total_time/good_runs
        #print('Average number of iterations: ', total_iters/good_runs)
        Average_number_of_iterations= total_iters/good_runs 
   
    else:
        Convergence_percent=0
        #print('\n')
        #print('Percentage of converged runs:', 100*good_runs/num_samples)
        #print('Average recovery time: ', total_time/good_runs)
        Average_recovery_time= np.inf
        #print('Average number of iterations: ', total_iters/good_runs)
        Average_number_of_iterations= N_iter
   
    return Convergence_percent, Average_recovery_time, Average_number_of_iterations
    #plt.show()

####################################################
# VECTORIZED MEASUREMENTS
####################################################


def vectorized_meas_measurements(XX,m1,num_samples,meas):

    #torch.manual_seed(0)

    n=tuple(XX[0].shape)
    #n=np.ones(d)*n1
    #r=np.ones(d)*r1

    
    mi=np.array([int(n[0]) for i in range(len(n))])
    m_first=mi.prod()
    m_second = m1**2
    #print("dim initial ", np.array(n).prod())
    #print("dim after comp ", m_second)

    yy = []
    start = timeit.default_timer()
    if meas=="Fourier":
        Afinal=create_kfjl_meas(m_first, m_second)
    elif meas=="Gaussian":
        Afinal=create_gaussian_meas(m_first, m_second)
    else:
        raise ValueError("Set meas to either 'Fourier' or 'Gaussian'")
    
    Afconj = Afinal.conj().T
    for j in range(num_samples):
        y =  np.matmul(Afinal,vectorize(XX[j]))
        yy.append(y)
    stop = timeit.default_timer()
    #print(y.shape)
    #print('Measurement time: ', (stop - start)/num_samples)
    avg_meas_time=(stop - start)/num_samples
    return Afinal,Afconj,yy, avg_meas_time
    
            
def VECTIHT(n1,d,r1,m1,num_samples=100,meas="Fourier",mu=.1,N_iter=1000,accuracy=.001):

    #torch.manual_seed(0)

    XX, reshaped_XX = generate_tensors(n1,d,r1,num_samples)
    Afinal,Afconj,yy, avg_meas_time=vectorized_meas_measurements(XX,m1,num_samples,meas)

    Losses3 = [[1] for _ in range(num_samples)]
    #mu, N_iter =.1, 1000
    #accuracy = 0.001

    good_runs = 0
    total_time = 0
    total_iters = 0

    n=tuple(XX[0].shape)
    r,rp=reshaped_rank(r1,d)

    X0 = torch.randn(n)

    # Run recovery algorithm
    for j in range(num_samples):
        #print(j)
        start = timeit.default_timer()
        X_iter=torch.clone(X0)
        first_loss = np.linalg.norm(XX[j] - X_iter)
        i = 0
        while Losses3[j][-1] > accuracy and i < N_iter:
            i += 1 
            Losses3[j].append(relative_error(true=XX[j] ,guess=X_iter, first_loss=first_loss))
            measX = np.matmul(Afinal,vectorize(X_iter))
            Z = yy[j] - measX
     
            Z = torch.reshape(torch.tensor(np.matmul(Afconj, Z)), n)
            Y_iter=X_iter+mu*Z
    
            X_iter=low_rank_approx(Y_iter, r)
        stop = timeit.default_timer()
    
        #plt.plot(range(len(Losses3[j])), Losses3[j])   
        if i < N_iter:
            good_runs += 1
            total_time += stop - start
            total_iters += i
            #print("Converged!")
            #print('Number of iterations: ', i)
    '''if good_runs != 0:
        print('\n')
        print('Percentage of converged runs:', 100*good_runs/num_samples)
        print('Average recovery time: ', total_time/good_runs) 
        print('Average number of iterations: ', total_iters/good_runs) 
    else:
        print("Never converged :(")'''
        
    if good_runs != 0:
        Convergence_percent=100*good_runs/num_samples
        #print('\n')
        #print('Percentage of converged runs:', 100*good_runs/num_samples)
        #print('Average recovery time: ', total_time/good_runs)
        Average_recovery_time= total_time/good_runs
        #print('Average number of iterations: ', total_iters/good_runs)
        Average_number_of_iterations= total_iters/good_runs 
   
    else:
        Convergence_percent=0
        #print('\n')
        #print('Percentage of converged runs:', 100*good_runs/num_samples)
        #print('Average recovery time: ', total_time/good_runs)
        Average_recovery_time= np.inf
        #print('Average number of iterations: ', total_iters/good_runs)
        Average_number_of_iterations= N_iter
   
    return Convergence_percent, Average_recovery_time, Average_number_of_iterations
    
    
    #plt.show()

####################################################
# TWOSTEP
####################################################



def two_step_measurements(XX,reshaped_XX,m1_intermediate,m2,num_samples,meas):


    #torch.manual_seed(0)

    n=tuple(XX[0].shape)
    nn=tuple(reshaped_XX[0].shape)

    # Compute and print out intermediate dimensions
    mi=np.array([m1_intermediate for i in range(len(nn))])
    m_first=mi.prod()
    m_second = m2

    #print("dim initial ", np.array(nn).prod())
    #print("dim after 1 comp ", m_first)
    #print("dim after 2 comp ", m_second)

    start = timeit.default_timer()

    #Compute Fourier intermediate operators
    if meas=="Fourier":
        A = [create_kfjl_meas(nn[i], mi[i]) for i in range(len(nn))]
        Afinal=create_kfjl_meas(m_first, m_second)
    elif meas=="Gaussian":
        A = [create_gaussian_meas(nn[i], mi[i]) for i in range(len(nn))]
        Afinal=create_gaussian_meas(m_first, m_second)
    else:
        raise ValueError("Set meas to either 'Fourier' or 'Gaussian'")
    AT=[A[i].conj().T for i in range(len(A))]
    Afconj = Afinal.conj().T

    yy = []
    # Compute measurements
    for j in range(num_samples):
        y1=vectorize(matrix_modewise_measurements(reshaped_XX[j], A))
        y = np.matmul(Afinal,y1)
        
        yy.append(y)
    #print('target dimension: ', y.shape) 
    stop = timeit.default_timer()
    #print('Measurement time: ', (stop - start)/num_samples) 
    average_time=(stop - start)/num_samples
    return A,AT,Afinal,Afconj,yy,average_time

def TWOSTEPTIHT(n1,d,r1,m1_intermediate,m2,num_samples=100,meas="Fourier",mu=.1,N_iter=1000,accuracy=.001):

    XX, reshaped_XX = generate_tensors(n1,d,r1,num_samples)
    n=tuple(XX[0].shape)
    nn=tuple(reshaped_XX[0].shape)
    mi=np.array([m1_intermediate for i in range(len(nn))])


    A,AT,Afinal,Afconj,yy,average_time=two_step_measurements(XX,reshaped_XX,m1_intermediate,m2,num_samples,meas)


    Losses2=[[1] for _ in range(num_samples)]
    #mu, N_iter =.1, 1000
    #accuracy = 0.001
    good_runs = 0
    total_time = 0
    total_iters = 0

    r,rp=reshaped_rank(r1,d)

    X0 = torch.randn(n)

    # Run recovery algorithm
    for j in range(num_samples):
        #print(j)
        start = timeit.default_timer()
        X_iter=torch.clone(X0)
        first_loss = np.linalg.norm(XX[j] - X_iter)
        i = 0
        while Losses2[j][-1] > 0.001 and i < N_iter:
            i += 1 
            Losses2[j].append(relative_error(true=XX[j],guess=X_iter, first_loss=first_loss))
            X_iter_reshaped=torch.reshape(X_iter, nn)
            first_step = matrix_modewise_measurements(X_iter_reshaped, A)

            measX = np.matmul(Afinal,vectorize(first_step))
            Z = yy[j] - measX
            Z = tensorize(np.matmul(Afinal.conj().T, Z), mi)

            Z = torch.reshape(matrix_modewise_measurements(Z, AT), n)    
            Y_iter=X_iter+mu*Z
            X_iter=low_rank_approx(Y_iter, r)
        
        stop = timeit.default_timer()
        #plt.plot(range(len(Losses2[j])), Losses2[j])   
        if i < N_iter:
            good_runs += 1
            total_time += stop - start
            total_iters += i
            #print("Converged!")
            #print('Number of iterations: ', i)
            #print('Final loss: ', Losses[j][-1])
    '''if good_runs != 0:
        print('\n')
        print('Percentage of converged runs:', 100*good_runs/num_samples)
        print('Average recovery time: ', total_time/good_runs) 
        print('Average number of iterations: ', total_iters/good_runs) 
    else:
        print("Never converged :(")'''
    
    if good_runs != 0:
        Convergence_percent=100*good_runs/num_samples
        #print('\n')
        #print('Percentage of converged runs:', 100*good_runs/num_samples)
        #print('Average recovery time: ', total_time/good_runs)
        Average_recovery_time= total_time/good_runs
        #print('Average number of iterations: ', total_iters/good_runs)
        Average_number_of_iterations= total_iters/good_runs 
   
    else:
        Convergence_percent=0
        #print('\n')
        #print('Percentage of converged runs:', 100*good_runs/num_samples)
        #print('Average recovery time: ', total_time/good_runs)
        Average_recovery_time= np.inf
        #print('Average number of iterations: ', total_iters/good_runs)
        Average_number_of_iterations= N_iter
   
    return Convergence_percent, Average_recovery_time, Average_number_of_iterations
    
############################################################
# RUN EXPERIMENTS
############################################################

def run_trial(n1,d,r1,m1,m1_intermediate,num_samples,mode,meas):
    
    #seeding
    torch.manual_seed(0)
    np.random.seed(0)
    
    #hyperparameters
    mu, N_iter =.1, 1000
    accuracy = 0.001

    #fix random seed here 
    if mode=="MW":
        return MWTIHT(n1,d,r1,m1,num_samples,meas,mu,N_iter,accuracy)
    elif mode=="VEC":
        return  VECTIHT(n1,d,r1,m1,num_samples,meas,mu,N_iter,accuracy)
    elif mode =="TWOSTEP":
        #m2 in TWOSTEP corresponds to m1^2
        return TWOSTEPTIHT(n1,d,r1,m1_intermediate,m1*m1,num_samples,meas,mu,N_iter,accuracy)
    else:
        raise ValueError("Invalid Mode: Please select MW, VEC, or TWOSTEP")

###########################################################
# STORE RESULTS IN CSV
##########################################################

#store parameters in a list
ns=[5]
rs=[2]
target_dims=[int(x**2) for x in [16]]
ds=[4]
m1smallss=[22]
modes=["VEC"]
num_samples=1
meases=["Gaussian"]
params=[(n1,d,r1,t,mode,meas) for n1 in ns for d in ds for t in target_dims for r1 in rs for mode in modes for meas in meases]


#store results in list
#empty lists 
resultns=[] #
resultrs=[]#
resulttars=[]#
resultpercents=[]#
resultds=[]#
resultiters=[]#
resulttimes=[]#
resultintermediates=[]
resultmodes=[]#
resultmeases=[]

for p in params:
    print(p)
    m1=(int(np.sqrt(p[3])))
    if p[4] == "TWOSTEP":
        
        #This is a cheap hack
        if p[0]==5:
            m1_intermediate=22
        else:
            m1_intermediate=22

        Convergence_percent, Average_recovery_time, Average_number_of_iterations=run_trial(p[0],p[1],p[2],m1,m1_intermediate,num_samples,p[4],p[5])
        resultintermediates.append(m1_intermediate**2)
    else:
        Convergence_percent, Average_recovery_time, Average_number_of_iterations=run_trial(p[0],p[1],p[2],m1,1,num_samples,p[4],p[5])
        resultintermediates.append("NA")
    
    #add parameter settings to a list
    resultns.append(p[0])
    resultds.append(p[1])
    resultrs.append(p[2])
    resulttars.append(p[3])
    resultmodes.append(p[4])
    resultmeases.append(p[5])
    resultiters.append(Average_number_of_iterations)
    resulttimes.append(Average_recovery_time)
    resultpercents.append(Convergence_percent)
    
#Put results in a dict
result_dict={
    "n":resultns,
    "r":resultrs,
    "target_dim":resulttars,
    "percent_recovered":resultpercents,
    "avg # iters": resultiters,
    "avg time": resulttimes,
    "intermediate dimension":resultintermediates,
    "mode":resultmodes,
    "meas":resultmeases
}
results=pd.DataFrame(result_dict)

#reorder columns
cols=["n","r","mode","target_dim","percent_recovered","avg # iters","avg time","intermediate dimension"]
results=results[cols]

#Save to CSV
now = datetime.now()
dt_string = now.strftime("%m%d%H%M")

name="results"+str(meases[0])+str(modes[0])+dt_string+".csv"

results.to_csv(name)


