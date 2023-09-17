# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:27:10 2023

@author: ANKITA PAUL
"""

import multiprocessing
import reservoirfunctions
import os
import blackbox
import numpy as np
import pickle
import pandas as pd

def rtlpaplearning(n1,n2,n3,trials,batch):
       errors=[]
       outputs = []
       targs = []
  #for i in range (n1,n2):
   #  print(i)

    # for j in range (1,10):
     #  print(j)
       alpha = 0.2
       p1 = reservoirfunctions.create_parameters(network_size=n1,dt=0.001,ff_alpha=alpha)
       p1['g'] = 1.5
       p1['ff_num_batches'] = batch
       p1['ff_trials_per_batch'] = trials
       p1['test_init_trials']=3
       p1['network size']=n1
       p1['ff_alpha']=0.2



       rnn1 = reservoirfunctions.Reservoir(p1,1,1)

       p2 = reservoirfunctions.create_parameters(network_size=n2,dt=0.001,ff_alpha=alpha)
       p2['g'] = 1.5
       p2['ff_num_batches'] = batch
       p2['ff_trials_per_batch'] = trials
       p2['test_init_trials']=3
       p2['network size']=n2
       p2['ff_alpha']=0.2



       rnn2 = reservoirfunctions.Reservoir(p2,1,1)

       p3 = reservoirfunctions.create_parameters(network_size=n3,dt=0.001,ff_alpha=alpha)
       p3['g'] = 1.5
       p3['ff_num_batches'] = batch
       p3['ff_trials_per_batch'] = trials
       p3['test_init_trials']=3
       p3['network size']=n3
       p3['ff_alpha']=0.2



       rnn3 = reservoirfunctions.Reservoir(p3,1,1)





       process1 = multiprocessing.Process(target=rnn1.train(blackbox.orig_system_test1, monitor_training=1))
       process2 = multiprocessing.Process(target=rnn2.train(blackbox.orig_system_test2, monitor_training=1))
       process3 = multiprocessing.Process(target=rnn3.train(blackbox.orig_system_test3, monitor_training=1))

       # Start both processes
       process1.start()
       process2.start()
       process3.start()

       # Wait for both processes to finish
       process1.join()
       process2.join()
       process3.join()

       # Get the results from both processes
       result1 = process1.exitcode  # You can use any variable you like here
       result2 = process2.exitcode  # You can use any variable you like here
       result3 = process3.exitcode

       #print("Result 1:", result1)
       #print("Result 2:", result2)
       #rnn1.train(orig_system_test, monitor_training=1)

       with open(os.path.join("model", "rnn1.pkl"), "wb") as file:
          pickle.dump(rnn1, file)
       with open(os.path.join("model", "rnn2.pkl"), "wb") as file:
          pickle.dump(rnn2, file)
       with open(os.path.join("model", "rnn3.pkl"), "wb") as file:
          pickle.dump(rnn3, file)

       #Test the processes

       enorm1,output1,target1= rnn1.test(blackbox.orig_system_test1)
       enorm2,output2,target2= rnn2.test(blackbox.orig_system_test2)
       enorm3,output3,target3= rnn3.test(blackbox.orig_system_test3)

       #print and save outputs

       print('enorm1',enorm1)
       print('enorm2',enorm2)
       print('enorm3',enorm3)

       np.savetxt('outputs1.csv',output1)
       np.savetxt('targs1.csv',target1)
       np.savetxt('outputs2.csv',output2)
       np.savetxt('targs2.csv',target2)
       np.savetxt('outputs3.csv',output3)
       np.savetxt('targs3.csv',target3)
       return rnn1,rnn2,rnn3

  #errors.append(enorm0[0])
  #np.savetxt('errors.csv',errors)


#Experiment 1
#rnn1, rnn2, rnn3 = rtlpaplearning(1250, 1250, 1250,1300)

#Experiment 2
rnn1, rnn2, rnn3 = rtlpaplearning(1250, 1250, 1250,10,10)



    # Create a pool of reservoirs


