from controller import Robot
import random
import numpy.random as rand
import math 

import numpy as np
import pandas as pd

TIME_STEP = 64          #Time step for phhyscial simulation
SAMPLE_TIME = TIME_STEP #Time step for Sensor/controller 

# #Custom codes
# #sourcehttps://medium.com/@aleksej.gudkov/python-pid-controller-example-a-complete-guide-5f35589eec86
class PID():
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.dt = dt

        self.previous_error = 0
        self.integral = 0
 
     # #Behaves like the TF given thhe error will calculate the control signal
    def compute(self, dt, error):
        # Proportional term
        P_out = self.Kp * error
        
        # Integral term
        self.integral += error * dt
        I_out = self.Ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        D_out = self.Kd * derivative
        
        # Compute total output
        output = P_out + I_out + D_out
        
        # Update previous error
        self.previous_error = error

        return output
        
    def saturate(self, signal, limits):
       if abs(signal) > abs(limits): 
           return abs(limits) * math.copysign(1,signal)
       
       return signal 
  
  
# #Code to generate noise for our signal such that the noise will have a normal distribiution about signal/10
def Noise(signal, sigma=0.05):
    return rand.normal(signal/10 ,sigma,1)[0]

# #._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.
# #._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.   

import pandas as pd
import numpy as np
import os
import random
import math
# from simple_pid import PID

class GeneticAlg():
    def __init__(self, pop_size:float ,step_size:float, mutation_rate:float, cross_over_rate:float, stm_aneal_rate, file_path, start_point_init_pop= 0.01, end_point_init_pop = 1000, decay_rate=0.01):
        self.file_path = file_path
        self.pop_size = pop_size
        self.step_size = step_size * np.exp(-1*decay_rate*self.generation_num(file_path))  #Exponential decay rate
        self.mutation_rate = mutation_rate
        self.cross_over_rate = cross_over_rate
        self.stm_aneal_rate = stm_aneal_rate    #Step size will decreas by a factor of *e^rate


        self.generation_cur = self.load_generation(file_path, start_point_init_pop, end_point_init_pop , pop_size)
        print(self.generation_cur)
        pass

    def __call__(self, robot):
        genom = self.get_genom()
        
    #Saving the rsualts
    def save(self, time, genom):
        for j in range(self.pop_size):
            if self.generation_cur["fitness"][j] == np.nan or self.generation_cur["fitness"][j] == None or pd.isna(ga.generation_cur['fitness'][j]):

                #Saving the generation
                num_gen = self.generation_num(self.file_path)  #The number of current generation

                #Getting the entire populations of all gen
                records = pd.read_csv(self.file_path)

                print("ERR")
        
                #Perparing the data of new generation
                records[f'Kp_{num_gen}'] = self.generation_cur["Kp"]
                records[f'Ki_{num_gen}'] = self.generation_cur["Ki"]
                records[f'Kd_{num_gen}'] = self.generation_cur["Kd"]
    
                #Modifing the row
                records[f'fitness_{num_gen}'][j] = time
                #Finding the diversity of the genom
                # records[f'diversity_{num_gen}'] = self.diversity_factor(genom , population=self.generation_cur.to_numpy()[:,0:3])
                #
                print("GW:", genom.to_numpy()[0:3])
                print("SDF", self.generation_cur.to_numpy()[:,0:3])

                records[f'diversity_{num_gen}'][j] = self.diversity_factor( genom.to_numpy()[0:3] , population=self.generation_cur.to_numpy()[:,0:3])

                #Saving the data in the csv format
                records.to_csv(self.file_path, index=False)

                #Success messaage
                break

            else:
                continue 

    #Tested succefully
    def fitness(self, time_stable):  
    #The fitness of a genom is the time it has managed to stablize the system
        return time_stable
        
    #Tested sucessfully
    def cross_over(self, genom1, genom2):
        #Performing Crossover between genom1 and genom2
    
        index = random.randint(1,2)
        gen1_new = np.hstack((genom1[:index], genom2[index:]))
        gen2_new = np.hstack((genom2[:index], genom1[index:]))

        return gen1_new, gen2_new
    
    def mutation(self, genom):
        #Mutaate the genom
        random = np.random.uniform(-1*self.step_size,+1*self.step_size,3)
        new_genom = genom.copy() 
        # new_genom["Kp"] += random[0]
        # new_genom["Ki"] += random[0]
        # new_genom["Kd"] += random[0]
        print("new:", new_genom)
        

        return new_genom

    def diversity_factor(self, genom, population):
        #Finding How diverse is the genom to the rest of the population
        #D = sum of the distance of the kp and kd and ki from everyother point
        distance = np.square(population - genom)
        return (np.sum(distance))**0.5
        pass

    #Tested sucessfully
    def gen_init_pop(self, lower_bounds, upper_bounds, number):
        #Generating the inital population with a uniform distribiutoin
        #Genom [Kp , Ki, Kd]
        row = np.linspace(lower_bounds, upper_bounds, number)
        init_pop = np.vstack((row,row,row))
        return np.transpose(init_pop)

    def select_survival(self):
        #Using diversity and fitness to determine the genoms to pass by
        
        #Choosing the survivals
        div_and_fit = self.generation_cur.copy()
        div_and_fit["score"] = self.generation_cur['diversity'] + 2 * self.generation_cur['fitness']

        sorted_div = div_and_fit.sort_values(by='score',ascending=False)
        genom1_div = sorted_div.iloc[0]
        genom2_div = sorted_div.iloc[1]
        genom3_div = sorted_div.iloc[2]

        return genom1_div, genom2_div, genom3_div

    def save_next_generation(self, next_gen):
        num_gen = 1+self.generation_num(self.file_path)  #The number of current generation

        #Getting the entire populations of all gen
        records = pd.read_csv(self.file_path)

        print("ERR")
        print(next_gen)
        #Perparing the data of new generation
        records[f'Kp_{num_gen}'] = next_gen["Kp"]
        records[f'Ki_{num_gen}'] = next_gen["Ki"]
        records[f'Kd_{num_gen}'] = next_gen["Kd"]
        records[f'fitness_{num_gen}'] = None
        records[f'diversity_{num_gen}'] = None
        records[f'seperator_{num_gen}'] = "|"

        #Saving the data in the csv format
        records.to_csv(self.file_path, index=False)

        #Success messaage
        print("New Generation ", num_gen ,"has been made succefully")

        #Changing the current popluation
        self.generation_cur = next_gen

        return self.generation_cur.iloc[0]
 
    #Tested succefully -> Should be rewritten to complie with DRY and be more modular
    def next_gen(self):
        #Using select_servival and mutioan and corssover to generate the next generation
        next_generation = self.generation_cur.copy()
        next_generation.drop(next_generation.index,inplace=True)     #Making the empty data_frame

        #Survivals
        genom1 , genom2, genom3 = self.select_survival()

        print("GENOM")
        print(genom1)
        print(genom2)
        print(genom3)

        genom1["diversity"] = pd.NA
        genom1["fitness"] = pd.NA
        genom1.drop("score")

        genom2["diversity"] = pd.NA
        genom2["fitness"] = pd.NA
        genom2.drop("score")

        genom3["diversity"] = pd.NA
        genom3["fitness"] = pd.NA
        genom3.drop("score")
        
        #Generating the next generation
        #The 3 original one and 3 mutated and 6 corssovered and mutated

        #Original
        next_generation.loc[len(next_generation)] = genom1
        next_generation.loc[len(next_generation)] = genom2
       

        #Mutated only
        next_generation.loc[len(next_generation)] = self.mutation(genom1)
        next_generation.loc[len(next_generation)] = self.mutation(genom2)
        

        #Crossovered and mutated
        next_generation.loc[len(next_generation)] = self.mutation(np.hstack([self.cross_over(genom1[0:3],genom2[0:3])[0], np.array([np.nan, np.nan])]))
        next_generation.loc[len(next_generation)] = self.mutation(np.hstack([self.cross_over(genom1[0:3],genom2[0:3])[1], np.array([np.nan, np.nan])]))
        next_generation.loc[len(next_generation)] = self.mutation(np.hstack([self.cross_over(genom1[0:3],genom3[0:3])[0], np.array([np.nan, np.nan])]))
        next_generation.loc[len(next_generation)] = self.mutation(np.hstack([self.cross_over(genom1[0:3],genom3[0:3])[1], np.array([np.nan, np.nan])]))
        next_generation.loc[len(next_generation)] = self.mutation(np.hstack([self.cross_over(genom2[0:3],genom3[0:3])[0], np.array([np.nan, np.nan])]))
        next_generation.loc[len(next_generation)] = self.mutation(np.hstack([self.cross_over(genom2[0:3],genom3[0:3])[1], np.array([np.nan, np.nan])]))

        # print("Next generation")
        # print(next_generation)

        #Saving the new generation
        geo = self.save_next_generation(next_generation)

        return geo

    def generation_num(self, path):
        if os.path.exists(path):
            records = pd.read_csv(path)
            return int(len(records.columns)/6)
        else:
            return 0
        
    # ✅ 
    def load_generation(self, path, lower_bounds, upper_bounds , number):
        #Loading the current generation

        #Checking if it is the first generation or not
        if os.path.exists(path):
            #If the record.csv file exist load data
            print("File exist")

            records = pd.read_csv(path)
            num_gen = self.generation_num(path)  #Finding the generatioin count
            print("Num gen:", num_gen)
            #Reshaping and triming the
            data = pd.DataFrame(np.array([records[f'Kp_{num_gen}'].values, records[f'Ki_{num_gen}'].values, records[f'Kd_{num_gen}'].values, records[f'fitness_{num_gen}'].values, records[f'diversity_{num_gen}'].values]).transpose(), columns=[f'Kp', f'Ki', f'Kd', f'fitness', f'diversity'])
            
            # Temporary 
            # records['Kp_2'] = data["Kp"]
            # records['Ki_2'] = data["Ki"]
            # records['Kd_2'] = data["Kd"]
            # records['fitness_2'] = None
            # records["diversity_2"] = None
            # records["seperator_2"] = "|"
            # records.to_csv('data.csv', index=False)
            

        else:
            #Making the initial population
            print("File doesn't exist")

            population = self.gen_init_pop(lower_bounds, upper_bounds , number) # return 3*10 
            
            #Saving the inital population in the records.csv
            records = pd.DataFrame(population, columns=['Kp_1', 'Ki_1', 'Kd_1'])
            data = records.copy() 
            data.rename(columns={'Kp_1':'Kp', 'Ki_1':'Ki', 'Kd_1':'Kd'}, inplace=True)

            records['fitness_1'] = None
            records["diversity_1"] = None
            records["seperator_1"] = "|"
            records.to_csv(path, index=False)
            

        return data

    def get_genom(self):
        for j in range(self.pop_size):    #This loop goes over the entire generation 
            genom = self.generation_cur.iloc[j]   #Getting one sspecific genom

            if pd.isna(ga.generation_cur['fitness'][j]): #If the genom was not yet used
                print("MODE1")
                return genom
            else:
                continue

        #If we have got to here then the generation has been fully studied
        #Generating a new generation
        print("MODE2")
        return self.next_gen()

#Using the genetic algorithm
ga = GeneticAlg(pop_size= 10 ,step_size=10 , mutation_rate=0.1, cross_over_rate=0.3, stm_aneal_rate=0.2, file_path="./data.csv", start_point_init_pop= 0.01, end_point_init_pop = 1000, decay_rate=0.1)

genom = ga.get_genom()

controller = PID(genom["Kp"], genom["Ki"], genom["Kd"], SAMPLE_TIME)

# #._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.
# #._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.     
# #Setup for the sensors and plant
robot = Robot()

#Motor
m = robot.getDevice("rotational motor")
m.setPosition(float('inf'))                   #Must be set or the setVelocity wont work

#Sensor
sensor = robot.getDevice("position sensor")
sensor.enable(SAMPLE_TIME)

#Using the genetic algorithem
i=0
#Simulation loop
while (robot.step(TIME_STEP) != -1):
    #Itteration counter
    i +=1
    
    #Finding the e(t) = 0 - y(t)
    error =-1 * sensor.getValue()
    
    
    #Finding the controller's signal part
    u = controller.compute(error=error, dt=SAMPLE_TIME)
    #Saturate the signal
    u_sat = controller.saturate(signal=u, limits=90) 
    
    #Noise
    d = Noise(signal = u_sat)
    
    #Applying noise every 10 steps
    if i%10 != 0:
        d = 0
    
    #Determining the signals 
    #print("Saturated control signal:", u_sat)
    #print("noise:", d)
    
    #Theta1_dot
    m.setVelocity(u_sat + d)
    
    #Finding the simulation time
    #print(robot.getTime())
  
# #._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.
# #._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.._.   