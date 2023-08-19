#!/usr/bin/env python
# coding: utf-8

# # Apartments2Thermal notebook example

# In this example, a rule based controller is used to control the model "Apartments2Thermal-v0". At first we import the controller:

# In[1]:


from energym.examples.Controller import LabController


# Next, we import Energym and create the simulation environment by specifying the model, a weather file and the number of simulation days.

# In[2]:


import energym

weather = "ESP_CT_Barcelona"
env = energym.make("Apartments2Thermal-v0", weather=weather, simulation_days=30)


# The control inputs can be inspected using the `get_inputs_names()` method and to construct a controller, we pass the list of inputs and further parameters. This controller determines inputs to get close to the temperature setpoints and uses fixed setpoints during the night.

# In[3]:


inputs = env.get_inputs_names()
print(inputs)
controller = LabController(control_list=inputs, lower_tol=0.3, upper_tol=0.8, nighttime_setback=True, nighttime_start=18, nighttime_end=6, nighttime_temp=18)


# To run the simulation, a number of steps is specified (here 480 steps per day for 10 days) and the obtained control inputs are passed to the simulation model with the `step()` method. To generate some plots later on, we save all the inputs and outputs in lists.

# In[4]:


steps = 480*10
out_list = []
outputs = env.get_output()
controls = []
hour = 0
for i in range(steps):
    control = controller.get_control(outputs, 21, hour)
    control['Bd_Ch_EV1Bat_sp'] = [0.0]
    control['Bd_Ch_EV2Bat_sp'] = [0.0]
    controls +=[ {p:control[p][0] for p in control} ]
    outputs = env.step(control)
    _,hour,_,_ = env.get_date()
    out_list.append(outputs)


# Since the inputs and outputs are given as dictionaries and are collected in lists, we can simply load them as a pandas.DataFrame.

# In[ ]:


import pandas as pd
out_df = pd.DataFrame(out_list)


# To generate plots, we can directly get the data from the DataFrames, by using the key names. Displayed are the zone temperatures and the setpoints determined by the controller for zone 1 as well as the corresponding heat pump on/off switching, the external temperature, and the total power demand.

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')

f, (ax1,ax2,ax3) = plt.subplots(3,figsize=(10,15))#


ax1.plot(out_df['Z01_T'], 'r')
ax1.plot(out_df['P1_T_Thermostat_sp_out'], 'b--')
ax1.plot(out_df['P1_onoff_HP_sp_out'], 'orange')
ax1.set_ylabel('Temp')
ax1.set_xlabel('Steps')

ax2.plot(out_df['Ext_T'], 'r')
ax2.set_ylabel('Temp')
ax2.set_xlabel('Steps')

ax3.plot(out_df['Fa_Pw_All'], 'g')
ax3.set_ylabel('Power')
ax3.set_xlabel('Steps')

plt.subplots_adjust(hspace=0.4)

plt.show()


# To end the simulation, the `close()` method is called. It deletes files that were produced during the simulation and stores some information about the simulation in the *runs* folder.

# In[7]:


env.close()


# In[14]:


env.get_kpi(start_ind=0, end_ind=)


# In[ ]:




