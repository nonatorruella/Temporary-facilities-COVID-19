#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from itertools import product
from math import sqrt

import gurobipy as gp
from gurobipy import GRB


# # Read excel file

# In[2]:


filename = 'input_data.xlsx'
data = pd.read_excel(filename,header = None)
data.head()
#sort the data
patients_demand = data[[0,1,2]]
existing_hospitals = data[[3,4,5]]
temporary_facilities = data[[6,7,8]]
data.head()


# In[3]:


def create_dictionary(df):
    multidata = dict()
    for index,row in df.iterrows():
        multidata[index] = [row[1],row[2]]
    return multidata


# ### 1. Demand of patients by ABS

# In[4]:


new_header = patients_demand.iloc[1] #grab the first row for the header
patients_demand = patients_demand[2:] #take the data less the header row
patients_demand.columns = new_header #set the header row as the df header
#demand = demand.set_index(['ABS'])
patients_demand = patients_demand.reset_index(drop = True)
patients_demand = patients_demand.dropna()
patients_demand.head()


# In[5]:


dictionary_demand = create_dictionary(patients_demand)
dictionary_demand


# In[6]:


dictionary_name_abs = patients_demand.ABS.to_dict()
dictionary_name_abs


# ### 2. Capacity of existing hospitals

# In[7]:


new_header = existing_hospitals.iloc[1] #grab the first row for the header
existing_hospitals = existing_hospitals[2:] #take the data less the header row
existing_hospitals.columns = new_header #set the header row as the df header
#existing_hospitals = existing_hospitals.set_index(['Hospital'])
existing_hospitals = existing_hospitals.reset_index(drop = True)
existing_hospitals = existing_hospitals.dropna()
existing_hospitals.head()


# In[8]:


dictionary_existing_hospitals = create_dictionary(existing_hospitals)
dictionary_existing_hospitals


# In[9]:


dictionary_name_facilities = existing_hospitals.Hospital.to_dict()


# ### 3. Possible new facilities

# In[10]:


new_header = temporary_facilities.iloc[1] #grab the first row for the header
temporary_facilities = temporary_facilities[2:] #take the data less the header row
temporary_facilities.columns = new_header #set the header row as the df header
temporary_facilities = temporary_facilities.dropna()
indices = [*range(len(dictionary_existing_hospitals),len(dictionary_existing_hospitals)+len(temporary_facilities))]
temporary_facilities.index = indices

temporary_facilities.head()


# In[11]:


temporary_facilities_dictionary = create_dictionary(temporary_facilities)
temporary_facilities_dictionary


# In[12]:


dictionary_name_facilities = {**dictionary_name_facilities,**temporary_facilities['Facility name'].to_dict()} #temporary_facilities['Facility name'].to_dict()
dictionary_name_facilities


# ### 4. Cost of building a new facility and cost per 10km to drive patients

# In[13]:


cost_new_facility =  data.iloc[0,10]
cost_drive_patients = data.iloc[1,10]


# ##  Definition of help functions: distance computations and solver

# In[14]:


def compute_distance(loc1, loc2):
    # This function determines the Euclidean distance between a facility and an ABS centroid.
    loc1= loc1[1:-1]
    loc2= loc2[1:-1]
    loc1 = [float(x.strip()) for x in loc1.split(',')]
    loc2 = [float(x.strip()) for x in loc2.split(',')]
    dx = loc1[0] - loc2[0]
    dy = loc1[1] - loc2[1]
    return sqrt(dx*dx + dy*dy)


# In[15]:


# Create a dictionary to capture the coordinates of an ABS and the demand of COVID-19 treatment
ABS, c_coordinates, demand  = gp.multidict(dictionary_demand)


# ## Setting parameters

# In[16]:


# Indices for the ABS
ABS = [*range(0,len(dictionary_demand))]
# Indices for the facilities
facilities = [*range(0,len(dictionary_existing_hospitals)+len(temporary_facilities_dictionary))]

#existing facility and capacity of treating COVID-19 patients
existing, e_coordinates, e_capacity  = gp.multidict(dictionary_existing_hospitals)

#temporary facility and capacity of treating COVID-19 patients
temporary, t_coordinates, t_capacity  = gp.multidict(temporary_facilities_dictionary)

# Cost of driving 10 km
dcost = cost_drive_patients
# Cost of building a temporary facility with capacity of COVID-19
tfcost = cost_new_facility

# Compute key parameters of MIP model formulation
f_coordinates = {}
for e in existing:
    f_coordinates[e] = e_coordinates[e]

for t in temporary:
    f_coordinates[t] = t_coordinates[t]

# Cartesian product of ABS and facilities
cf = []

for c in ABS:
    for f in facilities:
        tp = c,f
        cf.append(tp)

# Compute distances between ABS centroids and facility locations
distance = {(c,f): compute_distance(c_coordinates[c], f_coordinates[f]) for c, f in cf}


# ## Open output file

# In[17]:


import sys

orig_stdout = sys.stdout
f = open('output_temporary_facilities.txt', 'w')
sys.stdout = f


# ## Model MIP
# 
# ### Objective function: Minimize total distance to drive to a COVID-19 facility
# ### ABS demand constraints
# ### Existing facilities capacity constraints
# ### temporary facilities capacity constraints

# In[18]:


m = gp.Model('FACILITY_LOCATION_COVID19')

# Build temporary facility
y = m.addVars(temporary, vtype=GRB.BINARY, name='temporary')

# Assign COVID-19 patients of ABS to facility
x = m.addVars(cf, vtype=GRB.CONTINUOUS, name='Assign')

# Add capacity to temporary facilities
z = m.addVars(temporary, vtype=GRB.CONTINUOUS, name='addCap')

# Objective function: Minimize total distance to drive to a COVID-19 facility
# Big penalty for adding capacity at a temporary facility
penaltyC = 1e9
m.setObjective(gp.quicksum(dcost*distance[c,f]*x[c,f] for c,f in cf) 
               + tfcost*y.sum()
               + penaltyC*z.sum(), GRB.MINIMIZE)

# ABS demand constraints
demandConstrs = m.addConstrs((gp.quicksum(x[c,f] for f in facilities) == demand[c] for c in ABS), 
                             name='demandConstrs')

# Existing facilities capacity constraints
existingCapConstrs = m.addConstrs((gp.quicksum(x[c,e]  for c in ABS) <= e_capacity[e] for e in existing ), 
                                  name='existingCapConstrs')

# temporary facilities capacity constraints
temporaryCapConstrs = m.addConstrs((gp.quicksum(x[c,t]  for c in ABS) -z[t] 
                                    <= t_capacity[t]*y[t] for t in temporary ),
                                   name='temporaryCapConstrs')
# Run optimization engine
m.optimize()


# In[19]:


# Total cost of building temporary facility locations
temporary_facility_cost = 0

print(f"\n\n_____________Optimal costs______________________")
for t in temporary:
    if (y[t].x > 0.5):
        temporary_facility_cost += tfcost*round(y[t].x)

patient_allocation_cost = 0
for c,f in cf:
    if x[c,f].x > 1e-6:
        patient_allocation_cost += dcost*round(distance[c,f]*x[c,f].x)

print(f"The total cost of building COVID-19 temporary healthcare facilities is €{temporary_facility_cost:,}") 
print(f"The total cost of allocating COVID-19 patients to healthcare facilities is ${patient_allocation_cost:,}")  

# Build temporary facility at location

print(f"\n_____________Plan for temporary facilities______________________")
for t in temporary:
    if (y[t].x > 0.5):
        #print(f"Build a temporary facility at location {t}")
        t = dictionary_name_facilities[t]
        print(f"Build a temporary facility at location {t}")

# Extra capacity at temporary facilities
print(f"\n_____________Plan to increase Capacity at temporary Facilities______________________")
for t in temporary:
    if (z[t].x > 1e-6):
        t = dictionary_name_facilities[t]
        print(f"Increase  temporary facility capacity at location {t} by {round(z[t].x)} beds")

# Demand satisfied at each facility
f_demand = {}

print(f"\n_____________Allocation of ABS patients to COVID-19 healthcare facility______________________")
for f in facilities:
    temp = 0
    for c in ABS:
        allocation = round(x[c,f].x)
        if allocation > 0:
            f_ = dictionary_name_facilities[f]
            c_ = dictionary_name_abs[c]
            print(f"{allocation} COVID-19 patients from {c_} are treated at facility {f_} ") #FROM ABS
        temp += allocation
    f_demand[f] = temp
    
    if temp > 0:
        print(f"{temp} is the total number of COVID-19 patients that are treated at facility {f_}. ")
        print(f"\n________________________________________________________________________________")

# Test total demand = total demand satisfied by facilities
total_demand = 0

for c in ABS:
    total_demand += demand[c]

demand_satisfied = 0
for f in facilities:
    demand_satisfied += f_demand[f]

print(f"\n_____________Test demand = supply______________________")
print(f"Total demand is: {total_demand:,} patients")
print(f"Total demand satisfied is: {demand_satisfied:,} beds")


# ##  References
# [1] Katherine Klise and Michael Bynum. *Facility Location Optimization Model for COVID-19 Resources*. April 2020. Joint DOE Laboratory Pandemic Modeling and Analysis Capability. SAND2020-4693R.

# In[ ]:




