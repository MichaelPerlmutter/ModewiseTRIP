
"""
07 JUL 2022

This is a module that contains all the necessary functions to conduct the numeric experiments in the (revised) https://export.arxiv.org/abs/2109.10454

adapted from Michael Perlmutter's notebooks and scripts, this is a refactor by Cullen Haselby to allow for ease of use at MSU HPCC at facilitate larger tensor sizes and more humane runtimes.

Main improvements:

Reorganized main loop to better use and manage memory

Moved the most computational demanding linear algebra parts to GPU. 

Implemented FFT to allow for larger dimensions, note this is done only for A_final, not the modewise maps, which are still formed explicitly
Initialization is now a (random) low rank tensor (had been fully random)

Scale thresholding step off of the target accuracy, this prevents wasteful time spent fitting a low rank tensor at each step

Implemented a function which allows for multiple GPUs to complete vectorized TIHT. This is done only for Gaussian maps

Simplified or removed many of the tensor manipulations auxiliary functions, preferring tensorly calls since that allows for swapping out backends, though that wasn't always practical, and so torch calls appear throughout as well.
"""

#########################
# IMPORTS
#########################

import torch
import tensorly as tl
from tensorly import decomposition
from tensorly.random import random_tucker

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from datetime import datetime

##########################
# AUXILIARY FUCNTIONS
##########################

def create_kfjl_meas(dim, k):
    """Explicitly compute and return the RFD matrix of size k x dim. Hard coded to GPU

    Parameters
    ----------
    dim : int, required
        size of the vector or matrix you are planning to reduce with the map
    k : int, required
        sketching dimension. Should be smaller than dim
    
    Raises
    ------
    ValueError
        If dim<k then this is probably not being used right
    """
    
    #Appropriate root of unity needed to compute rows of the DFT
    w = torch.exp(torch.tensor([-2j*np.pi / dim], device="cuda"))
    #this is where the map is going to be constructed
    m = torch.empty((k,dim), dtype=torch.cfloat, device="cuda")
    
    if dim<k:
        raise ValueError("dim is less than k, matrix needs to be tall and skinny")
    
    #the R part of RFD- we figure out which rows of the DFT we actually plan on computing
    #Not in love with this approach but works fine
    rows = torch.randperm(dim)
    rows = rows[:k]
    rows = rows.sort()[0]
    rows = rows.to('cuda')
    
    #the enumeration of the columns, useful in doing this a row at a time
    cols = torch.arange(dim).to('cuda')

    #the D part of RFD. These are the sign changes that can be imagined as flipping sign of the columns in RF part
    vec=torch.randn(dim, device='cuda')
    vec[vec <= 0] = -1
    vec[vec > 0] = 1

    #Given a row we actually care about, compute that row of the DFT applying the signs as appropriate. Note the normalizing constant
    #Note I think broadcasting with the numpy call is adding an unnecessary memory write but didn't look into fixing it
    for i,r in enumerate(rows):
        m[i] =  np.sqrt(1/k)*(w**(r*cols))*vec

    return m

def create_kfjl_meas_slim(dim, k):
    """Returns the rows samped and the sign changes (the R and D of RFD) as vectors for use with FFT in order to compute the RFD at a smaller memory footprint than create_kfjl

    Parameters
    ----------
    dim : int, required
        size of the vector or matrix you are planning to reduce with the map
    k : int, required
        sketching dimension. Should be smaller than dim

    """
    rows = torch.randperm(dim)
    rows = rows[:k]
    rows = rows.sort()[0]
    
    vec=torch.randn(dim, device='cuda')
    vec[vec <= 0] = -1
    vec[vec > 0] = 1
    
    return rows, vec   

def create_gaussian_meas(dim, k):
    """Explicitly return the dense gaussian matrix of size k x dim. Hard coded to GPU and with normalizing constant sqrt(1/k)

    Parameters
    ----------
    dim : int, required
        size of the vector or matrix you are planning to reduce with the map
    k : int, required
        sketching dimension. Should be smaller than dim
    """    
    return torch.cuda.FloatTensor(k, dim).normal_(std=np.sqrt(1/k))

    
            
def VECTIHT(n1,d,r1,m1,num_samples=100,meas="Fourier",mu=.1,N_iter=1000,accuracy=.001):
    """Run vectorized TIHT on a problem of the given size and return the return performance information. For now, all dimensions have the same length and rank.
    
    Initialization is random using tensorly random tucker call

    Parameters
    ----------
    n1 : int, required
        length of a single mode
    d : int, required
        number of modes
    r1 : int, required
        rank of the tensor 
    m1 : int, required
        sketching dimension, should be smaller than n1^d 
    num_samples : int, default=100
        how many trials to run, each trial will generate a new target tensor and reform the sketching matrix
    meas : str, default = "Fourier" 
        "Gaussian" or "Fourier" are implemented currently. Note Gaussian will try and construct sketching matrix in GPU memory. Fourier uses RFD sketching matrix and will have significantly smaller memory foot print
    mu :  float, default=0.1
        hyper paramter in TIHT used in the thresholding step
    N_iter: int, default=1000
        max number of iterations of TIHT to perform per trial
    accuracy: float, default 0.001
        relative error tolerance, declare convergence if relative error goes below within N_iter
    
    Returns
    ----------
    Convergence_percent : float
        what percentage of trials converged given N_iter and accuracy
    Average_recovery_time : float
        elapsed time by wall clock, averaged over good runs
    Average_number_of_iterations : float
        average number of iterations, averaged over good runs
    reconst_rel_error: list
        per trial what the final relative error was before halting
    """   
    
    # for now, pytorch is really the only backend that will work
    tl.set_backend('pytorch')
    
    #tuples for convience
    n_tuple = tuple(n1 for _ in range(d))
    r_tuple = tuple(r1 for _ in range(d))

    #counters to keep track of performance
    reconst_rel_error = []
    good_runs = 0
    total_time = 0
    total_iters = 0
    
    #We should be able to throw away nearly everything between trials
    for trial in range(num_samples):
        torch.cuda.empty_cache() 
        
        #In the SORS case, we're going to use FFT and iFFT. Also data type is going to be torch.cfloat so can't share the same calls as Gaussian
        if meas=="Fourier":
            #Just need the vectors that pick the rows and flip the signs. 
            R,D = create_kfjl_meas_slim(n1**d, m1)
            
            #random tucker is a wrapper for the backend. 
            T_true = random_tucker(n_tuple,r_tuple,orthogonal=True,full=True) 
            T_true = T_true.detach().requires_grad_(False).type(torch.cfloat).to('cuda')
            
            T_init = random_tucker(n_tuple,r_tuple,orthogonal=True,full=True) 
            T_init = T_init.detach().requires_grad_(False).type(torch.cfloat).to('cuda')
            
            #Apply D to the true tensor, then F using FFT, then sample the rows using R
            y = np.sqrt(1/m1)*torch.fft.fft(D*T_true.reshape(-1), norm="backward")[R]
            
            #We will need this for performing iFFT in the loop - for padding with zeros at the right locations
            RZ = torch.zeros(n1**4, dtype=torch.cfloat, device='cuda')
            
        elif meas=="Gaussian":
            #Note that as implemented, the entire Afinal has to fit on the GPU memory, which restricts sizes to what can fit on the hardware
            Afinal = create_gaussian_meas(n1**d, m1)
            T_true = random_tucker(n_tuple,r_tuple,orthogonal=True,full=True) 
            T_true = T_true.detach().requires_grad_(False).type(torch.float).to('cuda')
            T_init = random_tucker(n_tuple,r_tuple,orthogonal=True,full=True) 
            T_init = T_init.detach().requires_grad_(False).type(torch.float).to('cuda')

            y = Afinal @ T_true.reshape(-1)            
        else:
            raise ValueError("Set meas to either 'Fourier' or 'Gaussian'")
        
        #Compute our first loss, random initialization vs target
        T_true_norm = tl.norm(T_true)
        reconst_rel_error.append([tl.norm(T_init - T_true) / T_true_norm])
        
        start = timeit.default_timer()    
        i = 0
        while reconst_rel_error[trial][-1] > accuracy and i < N_iter:
            i += 1
            #TIHT for vectors
            if meas=="Gaussian":
                Z = y -  Afinal @ T_init.reshape(-1)
                Z = Afinal.T @ Z
                
            if meas=="Fourier":
                #Apply RFD to T_init then take difference, then apply R^H to Z 
                RZ[R] = y - np.sqrt(1/m1)*(torch.fft.fft(D*(T_init.reshape(-1)), norm="backward")[R])
                #Now apply D^H F^H to the result 
                Z = D*np.sqrt(1/m1)*torch.fft.ifft(RZ,norm="forward")
            
            #Now perform the low rank thresholding step
            #Note the tolerance is scaled of the final accuracy
            #Occassionally for poorly chosen sketching dimension, the eigen value problem, which will involve calls to ARPACK or whatever the backend is using will fail to converge and NaNs will enter the stream which will then cause subsequent iterations to throw unexpected excpetions. If this happens, best bet is to reinitalize and try again.
            try:
                T_init=tl.tucker_to_tensor(tl.decomposition.tucker(T_init+mu*Z.reshape(n_tuple), r_tuple,tol=accuracy*0.1, init="random"))
            except:
                T_init = random_tucker(n_tuple,r_tuple,orthogonal=True,full=True) 
                if meas=="Fourier":
                    T_init= T_init.detach().requires_grad_(False).type(torch.cfloat).to('cuda')
                elif meas=="Gaussian":
                    T_init = T_init.detach().requires_grad_(False).to('cuda')
                else:
                    print("measure type", meas, "not implemented")
            #How'd we do so far?        
            reconst_rel_error[trial].append(tl.norm(T_init - T_true) / T_true_norm)
            
        #runs over, how long did it take?
        stop = timeit.default_timer()

        #clean up on the heavy hitters
        del T_true,T_init,Z,y
        
        if meas=="Gaussian":
            del Afinal
        
        #update counters if we broke out of the loop before maxing iterations
        if i < N_iter:
            good_runs += 1
            total_time += stop - start
            total_iters += i
    #compute the performance stats we want to track   
    if good_runs != 0:
        Convergence_percent=100*good_runs/num_samples
        Average_recovery_time= total_time/good_runs
        Average_number_of_iterations= total_iters/good_runs 
    
    #Avoid diving by zero if everything failed
    else:
        Convergence_percent=0
        Average_recovery_time= np.inf
        Average_number_of_iterations= N_iter
   
    return Convergence_percent, Average_recovery_time, Average_number_of_iterations,reconst_rel_error
    
def VECTIHT_DISTRIBUTED(n1,d,r1,m1,num_samples=100,meas="Gaussian",mu=.1,N_iter=1000,accuracy=.001):
    """Run vectorized TIHT on a problem of the given size and return the return performance information. For now, all dimensions have the same length and rank. This function will use all available CUDA devices to apply the measurement matrices. Only implemented for Guassian
    
    Initialization is random using tensorly random tucker call

    Parameters
    ----------
    n1 : int, required
        length of a single mode
    d : int, required
        number of modes
    r1 : int, required
        rank of the tensor 
    m1 : int, required
        sketching dimension, should be smaller than n1^d 
    num_samples : int, default=100
        how many trials to run, each trial will generate a new target tensor and reform the sketching matrix
    meas : str, default = "Guassian"
    mu :  float, default=0.1
        hyper paramter in TIHT used in the thresholding step
    N_iter: int, default=1000
        max number of iterations of TIHT to perform per trial
    accuracy: float, default 0.001
        relative error tolerance, declare convergence if relative error goes below within N_iter
    
    Returns
    ----------
    Convergence_percent : float
        what percentage of trials converged given N_iter and accuracy
    Average_recovery_time : float
        elapsed time by wall clock, averaged over good runs
    Average_number_of_iterations : float
        average number of iterations, averaged over good runs
    reconst_rel_error: list
        per trial what the final relative error was before halting
    """   
    tl.set_backend('pytorch')
    
    n_tuple = tuple(n1 for _ in range(d))
    r_tuple = tuple(r1 for _ in range(d))

    reconst_rel_error = []
    good_runs = 0
    total_time = 0
    total_iters = 0
        
    for trial in range(num_samples):
        torch.cuda.empty_cache() 
        if meas=="Gaussian":
            #how many GPUs we have available?
            ngpu = torch.cuda.device_count()
            
            #Each GPU gets an equal slice of the map. Doesn't handle m1 not divisible my number of devices elegantly
            Afinal = []
            Afinal.append(torch.cuda.FloatTensor(m1 // ngpu, n1**d).normal_(std=np.sqrt(1/m1)) ) 
            for i in range(1,ngpu):
                Afinal.append(torch.cuda.FloatTensor(m1 // ngpu, n1**d, device='cuda:' + str(i)).normal_(std=np.sqrt(1/m1))) 
            
            #True signal and initilization, note these will end up on device 0 usually, so you don't have the whole device available for Afinal
            T_true = random_tucker(n_tuple,r_tuple,orthogonal=True,full=True) 
            T_true = T_true.detach().requires_grad_(False).type(torch.float).to('cuda')
            T_init = random_tucker(n_tuple,r_tuple,orthogonal=True,full=True) 
            T_init = T_init.detach().requires_grad_(False).type(torch.float).to('cuda')
            
            #Give each device a copy of the vector
            B = T_true.reshape(-1)
            B_ = [T_true.reshape(-1)]
            
            for i in range(ngpu):
                if i != 0:
                    B_.append(B.to('cuda:' + str(i)))

            #perform the matrix vector multiply on each device        
            C_ = []
            for i in range(ngpu):
                C_.append(torch.matmul(Afinal[i], B_[i]))
            
            #Gather the results on the default first device
            y = torch.empty(m1)
            for i in range(ngpu):
                start_index = i * (m1//ngpu)
                y[start_index:start_index+(m1//ngpu)].copy_(C_[i])
            
            #clean up and sync threads
            torch.cuda.synchronize()
            del B, B_, C_
            
        else:
            raise ValueError("Set meas 'Gaussian', distrubted setup for gaussian to accomodate multiple GPUs worth of memory")
        

        T_true_norm = tl.norm(T_true)
        reconst_rel_error.append([tl.norm(T_init - T_true) / T_true_norm])

        start = timeit.default_timer()    
        iter_count = 0
        while reconst_rel_error[trial][-1] > accuracy and iter_count < N_iter:
            iter_count += 1

            if meas=="Gaussian":
                
                #Afinal @ Z step
                B = T_init.reshape(-1)

                B_ = [T_init.reshape(-1)]
                for i in range(ngpu):
                    if i != 0:
                        B_.append(B.to('cuda:' + str(i)))

                C_ = []
                for i in range(ngpu):
                    C_.append(torch.matmul(Afinal[i], B_[i]))

                Z = torch.empty(m1)
                for i in range(ngpu):
                    start_index = i * (m1//ngpu)
                    Z[start_index:start_index+(m1//ngpu)].copy_(C_[i]) 
                
                    
                Z = y - Z
                torch.cuda.synchronize()
                
                #Afinal.T @ (y-Z) step
                B = Z.reshape(-1)
                B_ = []
                for i in range(ngpu):
                    start_index = i * (m1//ngpu)
                    B_.append(B[start_index:start_index+(m1//ngpu)].to('cuda:' + str(i)))
    
                C_ = []
                for i in range(ngpu):
                    C_.append(torch.matmul(Afinal[i].T, B_[i]))
                
                Z = torch.zeros(n1**d, device='cuda:0')
                for i in range(ngpu):
                    Z += C_[i].to('cuda:0')
                
                torch.cuda.synchronize()
                
            #Note the thresholding is done on one device
            try:
                T_init=tl.tucker_to_tensor(tl.decomposition.tucker(T_init+mu*Z.reshape(n_tuple), r_tuple,tol=accuracy*0.1, init="random"))
            except:
                print("restarting initalization for failure to converge threhold step, iter ", iter_count)
                T_init = random_tucker(n_tuple,r_tuple,orthogonal=True,full=True) 
                T_init = T_init.detach().requires_grad_(False).to('cuda')
            #How did we do so far? 
            reconst_rel_error[trial].append(tl.norm(T_init - T_true) / T_true_norm)
        
        #How long did the run take?
        stop = timeit.default_timer()

       
        if meas=="Gaussian":
            del Afinal,T_init,T_true, B, B_, C_, Z
        
        
        if iter_count < N_iter:
            good_runs += 1
            total_time += stop - start
            total_iters += iter_count

        
    if good_runs != 0:
        Convergence_percent=100*good_runs/num_samples
        Average_recovery_time= total_time/good_runs

        Average_number_of_iterations= total_iters/good_runs 
   
    else:
        Convergence_percent=0
        Average_recovery_time= np.inf
        Average_number_of_iterations= N_iter
   
    return Convergence_percent, Average_recovery_time, Average_number_of_iterations,reconst_rel_error


def TWOSTEPTIHT(n1,d,r1,m1_intermediate,m2,num_samples=100,meas="Fourier",mu=.1,N_iter=1000,accuracy=.01):
    """Run two-step TIHT on a problem of the given size and return the return performance information. For now, all dimensions have the same length and rank. Reshaping will always be done according to d // 2, won't handle odds elegantly
    
    Initialization is random using tensorly random tucker call

    Parameters
    ----------
    n1 : int, required
        length of a single mode
    d : int, required
        number of modes
    r1 : int, required
        rank of the tensor
    m1_intermediate : int, required
        the sketching dimension to use once the tensor is reshaped to n1^(d//2) x n1^(d//2) ...
    m1 : int, required
        sketching dimension, should be smaller than n1^d 
    num_samples : int, default=100
        how many trials to run, each trial will generate a new target tensor and reform the sketching matrix
    meas : str, default = "Fourier" 
        "Gaussian" or "Fourier" are implemented currently. Note Gaussian will try and construct sketching matrix in GPU memory. Fourier uses RFD sketching matrix and will have significantly smaller memory foot print
    mu :  float, default=0.1
        hyper paramter in TIHT used in the thresholding step
    N_iter: int, default=1000
        max number of iterations of TIHT to perform per trial
    accuracy: float, default 0.001
        relative error tolerance, declare convergence if relative error goes below within N_iter
    
    Returns
    ----------
    Convergence_percent : float
        what percentage of trials converged given N_iter and accuracy
    Average_recovery_time : float
        elapsed time by wall clock, averaged over good runs
    Average_number_of_iterations : float
        average number of iterations, averaged over good runs
    reconst_rel_error: list
        per trial what the final relative error was before halting
    """   
    tl.set_backend('pytorch')

    #convenience tuples for various reshapes
    n_tuple = tuple(n1 for _ in range(d))
    r_tuple = tuple(r1 for _ in range(d))
    n_squared_tuple = tuple(n1**2 for _ in range(d//2)) 
    m1_tuple = tuple(m1_intermediate for _ in range(d//2))

    #initialize counters for tracking performance
    reconst_rel_error = []
    good_runs = 0
    total_time = 0
    total_iters = 0
    
    for trial in range(num_samples):
        torch.cuda.empty_cache() 

        if meas=="Fourier":
            #slim call only returns rows to sample and sign changes in RFD. Will use FFT to apply F part
            Rfinal, Dfinal = create_kfjl_meas_slim(m1_intermediate**(d//2),m2)   
            #Need a zero padded intermediary buffer for applying the conjugate RFD (inparticular the R^T part)
            RZ = torch.zeros(m1_intermediate**(d//2), dtype=torch.cfloat, device='cuda')
            
            T_true = random_tucker(n_tuple,r_tuple,orthogonal=True,full=True) 
            T_true = T_true.detach().requires_grad_(False).type(torch.cfloat).to('cuda')
            T_true_reshaped = T_true.reshape(n_squared_tuple)
            
            T_init = random_tucker(n_tuple,r_tuple,orthogonal=True,full=True) 
            T_init = T_init.detach().requires_grad_(False).type(torch.cfloat).to('cuda')
            
            #note currently we don't use FFT for the first modewise sketch. That would be possible, but these maps are pretty small anyways
            A = [create_kfjl_meas(n1**2, m1_intermediate).cuda() for i in range(d//2)]
            
            #Modewise measurements first
            y = tl.tenalg.multi_mode_dot(T_true_reshaped, A).reshape(-1)
            
            #Afinal measurements using FFT
            y = np.sqrt(1/m2)*torch.fft.fft(Dfinal*y, norm="backward")[Rfinal]
            
        elif meas=="Gaussian":
            A = [create_gaussian_meas(n1**2, m1_intermediate).cuda() for i in range(d//2)]
            Afinal = create_gaussian_meas(m1_intermediate**(d//2),m2).cuda()

            T_true = random_tucker(n_tuple,r_tuple,orthogonal=True,full=True) 
            T_true = T_true.detach().requires_grad_(False).to('cuda')
            
            T_true_reshaped = T_true.reshape(n_squared_tuple)
            
            T_init = random_tucker(n_tuple,r_tuple,orthogonal=True,full=True) 
            T_init = T_init.detach().requires_grad_(False).to('cuda')
            
            y = Afinal @ tl.tenalg.multi_mode_dot(T_true_reshaped, A).reshape(-1)
            
        else:
            raise ValueError("Set meas to either 'Fourier' or 'Gaussian'")
        
        #Compute the initial relative error at step 0
        T_true_norm = tl.norm(T_true)
        reconst_rel_error.append([tl.norm(T_init - T_true) / T_true_norm])
        
        start = timeit.default_timer()    
        i = 0
        while reconst_rel_error[trial][-1] > accuracy and i < N_iter:
                                                
            i += 1 
            if meas=="Gaussian":
                #TIHT with generic matrix-matrix multiply
                Z = y -  Afinal @ tl.tenalg.multi_mode_dot(T_init.reshape(n_squared_tuple), A).reshape(-1)
       
                Z = Afinal.T @ Z
                Z = tl.tenalg.multi_mode_dot(Z.reshape(m1_tuple), A, transpose=True).reshape(n_tuple)
            
            if meas=="Fourier":
                #TIHT with generic matrix-matrix multiply for the modewise part, FFT for final reduction
                Z = tl.tenalg.multi_mode_dot(T_init.reshape(n_squared_tuple), A).reshape(-1)
                RZ[Rfinal] = y -  np.sqrt(1/m2)*torch.fft.fft(Dfinal*Z, norm="backward")[Rfinal]
       
                Z = Dfinal*np.sqrt(1/m2)*torch.fft.ifft(RZ,norm="forward")
                Z = tl.tenalg.multi_mode_dot(Z.reshape(m1_tuple), A, transpose=True).reshape(n_tuple)
             
            #Now perform the low rank thresholding step
            #Note the tolerance is scaled of the final accuracy
            #Occassionally for poorly chosen sketching dimension, the eigen value problem, which will involve calls to ARPACK or whatever the backend is using will fail to converge and NaNs will enter the stream which will then cause subsequent iterations to throw unexpected excpetions. If this happens, best bet is to reinitalize and try again. 
            try:
                T_init=tl.tucker_to_tensor(tl.decomposition.tucker(T_init+mu*Z.reshape(n_tuple), r_tuple,tol=accuracy*0.1, init="random"))
            except:
                T_init = random_tucker(n_tuple,r_tuple,orthogonal=True,full=True) 
                if meas=="Fourier":
                    T_init= T_init.detach().requires_grad_(False).type(torch.cfloat).to('cuda')
                elif meas=="Gaussian":
                    T_init = T_init.detach().requires_grad_(False).to('cuda')
                else:
                    raise ValueError("Set meas to either 'Fourier' or 'Gaussian'")
            #How did we do so far?       
            reconst_rel_error[trial].append(tl.norm(T_init - T_true) / T_true_norm)
        #How long did the trial take?    
        stop = timeit.default_timer()
        
        #Clean up, don't want OOM for leaving things behind that are no longer needed for next trial
        if meas=="Gaussian":
            del A, Afinal, T_true, T_init, y, Z
        if meas=="Fourier":
            del A,T_true, T_init, y, Z, RZ, Rfinal, Dfinal
            
        if i < N_iter:
            good_runs += 1
            total_time += stop - start
            total_iters += i

    if good_runs != 0:
        Convergence_percent=100*good_runs/num_samples
        Average_recovery_time= total_time/good_runs
        Average_number_of_iterations= total_iters/good_runs 
   
    else:
        Convergence_percent=0
        Average_recovery_time= np.inf
        Average_number_of_iterations= N_iter
   
    return Convergence_percent, Average_recovery_time, Average_number_of_iterations, reconst_rel_error