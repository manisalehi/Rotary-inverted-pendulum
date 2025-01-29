import pandas as pd
import numpy as np
import os
import random

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

    #Tested sucessfully
    def mutation(self, genom):
        #Mutaate the genom
        new_genom = genom + np.random.uniform(-1*self.step_size,+1*self.step_size,len(genom))
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

    def select_survival(self, population):
        #Using diversity and fitness to determine the genoms to pass by
        pass

    def next_gen(self):
        #Using select_servival and mutioan and corssover to generate the next generation
        pass

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
            data = pd.DataFrame(2*np.array([records[f'Kp_{num_gen}'].values, records[f'Ki_{num_gen}'].values, records[f'Kd_{num_gen}'].values, records[f'fitness_{num_gen}'].values, records[f'diversity_{num_gen}'].values]).transpose(), columns=[f'Kp', f'Ki', f'Kd', f'fitness', f'diversity'])
            
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

    def save_next_generation(self, next_gen):
        pass


ga = GeneticAlg(pop_size= 10 ,step_size=10 , mutation_rate=0.1, cross_over_rate=0.3, stm_aneal_rate=0.2, file_path="./data.csv", start_point_init_pop= 0.01, end_point_init_pop = 1000, decay_rate=0.1)



# crossed = ga.cross_over([1,2,3],[4,5,6])[0]
# mutated = ga.mutation(crossed)

# print(crossed)
# print(mutated)
