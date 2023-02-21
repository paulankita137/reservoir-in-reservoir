# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:28:41 2023

@author: ap3737
"""


import sys
import os

sys.path.insert(1, "..")
from snn_online import *

import numpy as np

import pickle

# general network parameters
N_neurons = 200
dt = 0.00005
alpha = 20*dt

N_inputs = 1
input_dims = 1

N_outputs = 1
output_dims = 1

N_hints = 0
hint_dims = 0

# parameters
gg = gp = 1.5
Qg = Qp = 0.01
tm_g = tm_p = 0.01
td_g = td_p = 0.02
tr_g = tr_p = 0.002
ts_g = ts_p = 0.01
E_Lg = E_Lp = -60
v_actg = v_actp = -45
bias_g = bias_p = v_actg
std_Jg = std_Jp = gg/np.sqrt(N_neurons)
mu_wg = mu_wp = 0
std_wg = std_wp = 1
std_uing = std_uinp = 0.2

# training parameters
T = 15
init_T = 5
init_steps = int(init_T/dt)
train_T = 5
train_steps = int(train_T/dt) + init_steps
p = 20


def training(Network, f, f_out, h, init_steps, training_steps, p):

    print("----- TRAINING -----")
    print("--- Initializing ---")

    Network.Gen.reset_activity()
    Network.Per.reset_activity()

    for t in range(500):
        
        Network.Gen.step(f, t, f_out=f_out, h=h)
        Network.Per.step(f, t)

    print("STARTING TRAINING")
    
    #f = f[init_steps:training_steps]
    
    f = f[0:500]

    f_out = f_out[0:500]

    if h != None:
        h = h[init_steps:]

    Network.train_once(len(f), f, f_out, h, p)
    print(f"--- Avg Spike-Rate: {np.mean(Network.Per.spike_count)/((training_steps-init_steps)*Network.dt)} Hz")

    with open(os.path.join("model", "triangleandsine5000steps_force.pkl"), "wb") as file:
        pickle.dump(Network, file)


N = S_RNN(N_neurons=N_neurons, \
        N_inputs=N_inputs, \
        input_dims=input_dims, \
        N_outputs=N_outputs, \
        output_dims=output_dims, \
        alpha=alpha, \
        gg=gg, \
        gp=gp, \
        Qg=Qg, \
        Qp=Qp, \
        dt=dt, \
        tm_g=tm_g, \
        tm_p=tm_p, \
        td_g=td_g, \
        td_p=td_p, \
        tr_g=tr_g, \
        tr_p=tr_p, \
        ts_g=ts_g, \
        ts_p=ts_p, \
        E_Lg=E_Lg, \
        E_Lp=E_Lp, \
        v_actg=v_actg, \
        v_actp=v_actp, \
        bias_g=bias_g, \
        bias_p=bias_p, \
        std_Jg=std_Jg, \
        std_Jp=std_Jp, \
        mu_wg=mu_wg, \
        mu_wp=mu_wp, \
        std_wg=std_wg, \
        std_wp=std_wp, \
        std_uing=std_uing, \
        std_uinp=std_uinp, \
        hints=False, \
        N_hints=N_hints, \
        hint_dims=hint_dims) 


length = int(T/dt)

# input signal
f_in = np.zeros((5000, 1, 1))
makeSpike = True
for t in range(len(f_in)):
    if makeSpike:
        f_in[t] = 1
    if t%int(1/dt) > int(0.05/dt):
        makeSpike = False
    if (t+1) % (int(1/dt)) == 0:
        makeSpike = True

# target signal
#f_out = np.zeros((length, 1, 1))
#for i in range(len(f_out)):
#    f_out[i][0][0] = np.arcsin(np.sin(2*np.pi*i*dt*10))




# f_out = np.zeros((length, 1, 1))
# for i in range(len(f_out)):
#         f_out[i][0][0] = np.arcsin(np.sin(2*np.pi*i*dt*10))+np.sin(4*np.pi*i*dt)


import pandas as pd
sample = pd.read_csv('f_outtrisine.csv')
rawvalues = sample['a'].values
rawvalues = list(rawvalues)



m = []

#f_out = np.zeros((length, 1, 1))

f_out = np.zeros((5000, 1, 1))

# for i in range(len(f_out)):
#         f_out[i][0][0] = np.arcsin(np.sin(2*np.pi*i*dt*10))+np.sin(4*np.pi*i*dt)
#         v = f_out[i][0][0]
#         m.append(f_out[i][0][0])





#for d in range(len(f_out)):
#    f_out[d][0][0] = rawvalues[d]





for d in range(5000):
    f_out[d][0][0] = rawvalues[d]



# hints
h = None

training(Network=N, \
        f=f_in, \
        f_out=f_out, \
        h=h, \
        init_steps=init_steps, \
        training_steps=train_steps, \
        p=p)








import sys
import os

sys.path.insert(1, "..")
from snn_online import *

import numpy as np

import pickle

import matplotlib.pyplot as plt

# # general network parameters
N_neurons = 200
dt = 0.00005
alpha = 20*dt

N_inputs = 1
input_dims = 1

N_outputs = 1
output_dims = 1

N_hints = 0
hint_dims = 0

# parameters
gg = gp = 1.5
Qg = Qp = 0.01
tm_g = tm_p = 0.01
td_g = td_p = 0.02
tr_g = tr_p = 0.002
ts_g = ts_p = 0.01
E_Lg = E_Lp = -60
v_actg = v_actp = -45
bias_g = bias_p = v_actg
std_Jg = std_Jp = gg/np.sqrt(N_neurons)
mu_wg = mu_wp = 0
std_wg = std_wp = 1
std_uing = std_uinp = 0.2

# testing parameters
T = 15
init_T = 5
init_steps = int(init_T/dt)
train_T = 5
train_steps = int(train_T/dt) + init_steps


def test(Network, f, f_out, h, train_end):

    f = f[train_end:]
    f_out = f_out[train_end:]
    if h != None:
        h = h[train_end:]

    x = np.zeros((len(f_out), 1))

    Network.reset_spike_count()

    print("Testing ...")

    MSE = 0

    for t in range(len(f)):
        x[t] = Network.step(f, t)
        MSE += np.sum((x[t] - f_out[t])**2)
        
        MSE = MSE/ len(f)
        #print(MSE)
        if MSE >0.25 : 
            training(Network=N, \
        f=f_in, \
        f_out=f_out, \
        h=h, \
        init_steps=init_steps, \
        training_steps=train_steps, \
        p=p)


    MSE = MSE / len(f)
    print(f"MSE: {MSE}")
    print(f"--- Avg Spike-Rate: {np.mean(Network.Per.spike_count)/(len(f)*Network.dt)} Hz")

    spacing = np.linspace(train_end*dt, (train_end+len(f_out))*dt, len(f_out)).flatten()

    plt.plot(spacing, x.flatten(), label="output")
    plt.plot(spacing, f_out.flatten(), label="target", linestyle="dashed")
    plt.plot(spacing, f.flatten(), label="input", linestyle="dotted")
    plt.legend(loc="upper right")
    plt.show()


N = S_RNN(N_neurons=N_neurons, \
        N_inputs=N_inputs, \
        input_dims=input_dims, \
        N_outputs=N_outputs, \
        output_dims=output_dims, \
        alpha=alpha, \
        gg=gg, \
        gp=gp, \
        Qg=Qg, \
        Qp=Qp, \
        dt=dt, \
        tm_g=tm_g, \
        tm_p=tm_p, \
        td_g=td_g, \
        td_p=td_p, \
        tr_g=tr_g, \
        tr_p=tr_p, \
        ts_g=ts_g, \
        ts_p=ts_p, \
        E_Lg=E_Lg, \
        E_Lp=E_Lp, \
        v_actg=v_actg, \
        v_actp=v_actp, \
        bias_g=bias_g, \
        bias_p=bias_p, \
        std_Jg=std_Jg, \
        std_Jp=std_Jp, \
        mu_wg=mu_wg, \
        mu_wp=mu_wp, \
        std_wg=std_wg, \
        std_wp=std_wp, \
        std_uing=std_uing, \
        std_uinp=std_uinp, \
        hints=False, \
        N_hints=N_hints, \
        hint_dims=hint_dims)


length = int(T/dt)

f_in = np.zeros((length, 1, 1))
makeSpike = True
for t in range(len(f_in)):
    if makeSpike:
        f_in[t] = 1
    if t%int(1/dt) > int(0.05/dt):
        makeSpike = False
    if (t+1) % (int(1/dt)) == 0:
        makeSpike = True

# target signal
#f_out = np.zeros((length, 1, 1))
#for i in range(len(f_out)):
#    f_out[i][0][0] = np.arcsin(np.sin(2*np.pi*i*dt*10))

import pandas as pd
sample = pd.read_csv('f_outtrisine.csv')
rawvalues = sample['a'].values
rawvalues = list(rawvalues)



k = []

#f_out = np.zeros((length, 1, 1))

f_out = np.zeros((length, 1, 1))

# for i in range(len(f_out)):
#         f_out[i][0][0] = np.arcsin(np.sin(2*np.pi*i*dt*10))+np.sin(4*np.pi*i*dt)
#         v = f_out[i][0][0]
#         m.append(f_out[i][0][0])





#for d in range(len(f_out)):
#    f_out[d][0][0] = rawvalues[d]





for d in range(len(f_out)):
    f_out[d][0][0] = rawvalues[d]



#np.savetxt('f_out2.csv',m)
# hints
h = None


with open(os.path.join("model", "triangleandsine5000steps_force.pkl"), "rb") as file:
    N = pickle.load(file)

test(N, f_in, f_out, h, train_steps)
