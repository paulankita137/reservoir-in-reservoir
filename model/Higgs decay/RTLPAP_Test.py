

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:54:35 2023

@author: ANKITA PAUL
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import pickle
import pandas as pd


momentum = pd.read_csv('momentum.csv')


dt = 0.001
dt_per_s = round(1/dt)


def orig_system_test(dt, showplots=0):
    dt_per_s = round(1/dt)

    # From the paper, and the online demo:
    t = np.expand_dims(np.linspace(0,2,2*dt_per_s+1),1)
    omega = np.zeros((2*dt_per_s+1,1))
    omega = np.linspace(2*np.pi, 6*np.pi, 1*dt_per_s+1)
    targ = np.zeros((2*dt_per_s+1,1))
    targ[0:(1*dt_per_s+1),0] = np.sin(t[0:(1*dt_per_s+1),0]*omega)
    targ[1*dt_per_s:(2*dt_per_s+1)] = -np.flipud(targ[0:(1*dt_per_s+1)])
    #print('sine target shape',targ.shape)

    # omega = np.ones((2*dt_per_s,1)) * 4 *np.pi
    # targ = np.sin(t*2*omega) * np.sin(t*omega/4)


    #for i in range(len(targ)):
	   #   targ[i][0] = np.arcsin(np.sin(2*np.pi*i*dt*10))


   #momentum prediction
    sample = pd.read_csv('momentum.csv')

    rawvalues1 = sample['px1'].values
    rawvalues2 = sample['py1'].values
    rawvalues3 = sample['pz1'].values

    #rawvalues=sample.values.tolist()
    #print('rawvalues1',rawvalues1)
    #print('rawvalues2',rawvalues2)


    rawvalues1 = rawvalues1[0:4001]
    rawvalues11 = rawvalues1[0:30001]
    rawvalues2 = rawvalues2[0:2001]
    rawvalues22 = rawvalues2[0:30001]
    rawvalues3 = rawvalues3[0:2001]
    rawvalues33 = rawvalues3[0:30001]

    #target signal
    targ = np.zeros((2001, 1))


    #target signal
    targ = np.zeros((2001, 1))



    for d in range(len(targ)):
        targ[d][0] = rawvalues3[d]





    inp = np.zeros(targ.shape)
    inp[0:round(0.05*dt_per_s),0] = np.ones((round(0.05*dt_per_s)))

    hints = np.zeros(targ.shape)
    #print('hints',hints)
    if showplots == 1:
        plt.figure()
        plt.plot(targ)
        plt.plot(hints)
        plt.plot(inp)
        plt.show()
        plt.legend(['Target','Hints','Input'])

    return inp, targ, hints



def system_change_test(dt, showplots=0):
    dt_per_s = round(1/dt)

    # From the paper, and the online demo:
    t = np.expand_dims(np.linspace(0,2,2*dt_per_s+1),1)
    omega = np.zeros((2*dt_per_s+1,1))
    omega = np.linspace(2*np.pi, 6*np.pi, 1*dt_per_s+1)
    targ = np.zeros((2*dt_per_s+1,1))
    targ[0:(1*dt_per_s+1),0] = np.sin(t[0:(1*dt_per_s+1),0]*omega)
    targ[1*dt_per_s:(2*dt_per_s+1)] = -np.flipud(targ[0:(1*dt_per_s+1)])
    #print('sine target shape',targ.shape)

    # omega = np.ones((2*dt_per_s,1)) * 4 *np.pi
    # targ = np.sin(t*2*omega) * np.sin(t*omega/4)
  


    #for i in range(len(targ)):
	   #   targ[i][0] = np.arcsin(np.sin(2*np.pi*i*dt*10))







    inp = np.zeros(targ.shape)
    inp[0:round(0.05*dt_per_s),0] = np.ones((round(0.05*dt_per_s)))

    hints = np.zeros(targ.shape)
    #print('hints',hints)
    if showplots == 1:
        plt.figure()
        plt.plot(targ)
        plt.plot(hints)
        plt.plot(inp)
        plt.show()
        plt.legend(['Target','Hints','Input'])

    return inp, targ, hints







orig_system_test(dt=0.001,showplots=1)
#system_change_test(dt=0.001,showplots=1)

import reservoirfunctions
import pickle
import os


#rnn1.train(fullforce_oscillation_test, monitor_training=1)
def YYO(n1,n2):
  errors=[]
  outputs = []
  targs = []
  for i in range (n1,n2):
     print(i)
   
     for j in range (1,10):
       print(j)
       alpha = j/10.0
       p = reservoirfunctions.create_parameters(network_size=i,dt=0.001,ff_alpha=alpha)
       p['g'] = 1.5 # From paper
       p['ff_num_batches'] = 10
       p['ff_trials_per_batch'] = 10
       p['test_init_trials']=10
       p['network size']=i
       p['ff_alpha']=alpha
       rnn1 = reservoirfunctions.reservoir(p,1,1)
       rnn1.train(orig_system_test, monitor_training=1)
       enorm0,output0,target0= rnn1.test(orig_system_test)
       #outputs.append(output0)
       print('enorm0',enorm0)
       np.savetxt('outputsrv3'+str(i)+str(j)+'.csv',output0)
       np.savetxt('targsrv3.csv',target0)
       with open(os.path.join("model", "rnnrv3"+str(i)+str(j)+".pkl"), "wb") as file:
          pickle.dump(rnn1, file)
       if enorm0[0]<0.3:
          with open(os.path.join("model", "rnn"+str(i)+str(j)+".pkl"), "wb") as file:
             pickle.dump(rnn1, file)
          #return rnn1
       else:
         j+=1      
 
  print('enorm0',enorm0)
  errors.append(enorm0[0])
  np.savetxt('errors.csv',errors)
  return errors,rnn1,enorm0,output0

errors, rnn, enorm0,output0 = YYO(800, 900)


# try:
#     errors, rnn, enorm0,output0 = YYO(2500, 2520)
# except Exception as e:
#     print(f"An error occurred: {str(e)}")


#New system change
system_change_test(dt=0.001,showplots=1)

#with open(os.path.join("model", "rnn803.pkl"), "rb") as file1:
 #     reservoir1 = pickle.load(file1)
#with open(os.path.join("model", "rnn811.pkl"), "rb") as file2:
 #     reservoir2 = pickle.load(file2)

#system_change_test(dt=0.001,showplots=1)

#enorm1,output1,targ1= reservoir1.test(system_change_test)
#enorm2,output2,targ2= reservoir2.test(system_change_test)




