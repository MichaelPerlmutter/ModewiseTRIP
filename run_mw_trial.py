
"""
07 JUL 2022

This is a script is for running a job and gathering and saving results into a dataframe and csv 

adapted from michael perlmutters scripts, this is a refactor by Cullen Haselby to allow for ease of use at MSU HPCC
"""
#########################
# IMPORTS
#########################
import mwnumerics_gpu as mw
import torch
import timeit
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from csv import DictWriter,writer
import importlib

############################################################
# RUN EXPERIMENTS
############################################################

def run_trial(n1,d,r1,m1,m1_intermediate,num_samples,mode,meas,accuracy=0.01,mu=0.1,N_iter=1000):

    if mode=="VEC":
        return  mw.VECTIHT(n1,d,r1,m1,num_samples,meas,mu,N_iter,accuracy)
    elif mode=="VEC_DIST":
        return  mw.VECTIHT_DISTRIBUTED(n1,d,r1,m1,num_samples,meas,mu,N_iter,accuracy)
    elif mode =="TWOSTEP":
        return mw.TWOSTEPTIHT(n1,d,r1,m1_intermediate,m1,num_samples,meas,mu,N_iter,accuracy)
    else:
        raise ValueError("Invalid Mode: Please select MW, VEC, TWOSTEP, TWOSTEPNCG")

###########################################################
# STORE RESULTS IN CSV
##########################################################


if __name__ == "__main__":
    
    #Take the command line arguments, get rid of the script name
    args = sys.argv[1:]

    #store the settings into a list, splitting and casting as ints
    ns=[int(n) for n in args[0].split(",")]
    ds=[int(d) for d in args[1].split(",")]
    rs=[int(r) for r in args[2].split(",")]
    m1ints=[int(r) for r in args[3].split(",")]
    target_dims=[int(x) for x in args[4].split(",")]
    #currently all combinations of parameters will have the same number of trials
    num_samples = int(args[5])
    modes=args[6].split(",")
    meases=args[7].split(",")

    #organize the parameters into a list of tuples
    params=[(n1,d,r1,m1s,t,mode,meas) for n1 in ns for d in ds for m1s in m1ints for t in target_dims for r1 in rs for mode in modes for meas in meases]

    #open a file for writing the results in.
    now = datetime.now()
    dt_string = now.strftime("%m%d%H%M")
    name="results/results"+dt_string+".csv"
    print("Writing results to ", name)
    cols=["mode","meas","n","r","target_dim","samples","percent_recovered","avg # iters","avg time","intermediate dimension"]

    with open(name, 'a', newline='') as f_object:  
        # Pass the CSV  file and write in the column headers
        writer_object = writer(f_object)
        writer_object.writerow(cols)  
        f_object.close()

    for p in params:
        print("Running trial(s) for parameters ", p)
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

        #intermediate dimension shouldn't be a thing for vectorized, so just set to one
        if p[5] == "TWOSTEP":
            Convergence_percent, Average_recovery_time, Average_number_of_iterations, errors=run_trial(p[0],p[1],p[2],p[4],p[3],num_samples,p[5],p[6])
        else:
            Convergence_percent, Average_recovery_time, Average_number_of_iterations, errors=run_trial(p[0],p[1],p[2],p[4],1,num_samples,p[5],p[6])

        #after the command is run, gather the stats and other fields into a dictionary
        result_dict={
            "n":p[0],
            "r":p[2],
            "target_dim":p[4],
            "samples": num_samples,
            "percent_recovered":Convergence_percent,
            "avg # iters": Average_number_of_iterations,
            "avg time": Average_recovery_time,
            "intermediate dimension":p[3],
            "mode":p[5],
            "meas":p[6]
        }
        
        with open(name, 'a', newline='') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=cols)
            dictwriter_object.writerow(result_dict)
            f_object.close()