import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import random
import copy
import datetime
import time
import multiprocessing as mp

# 매개변수
INPUT_FILES = [
    't_500_20_mon.csv', 
    't_500_20_tue.csv', 
    # 't_500_20_wed.csv',
    't_500_20_thu.csv',
    # 't_500_20_fri.csv'
]
   
NUM_JOBS = [120, 60, 120]   # 일별 생산요구량
TIME_LIMIT = 3600   # 60분 시간제한

# =========================================== NMA ===========================================
class FlowShopNMA:
    def __init__(self, INPUT_FILE, num_jobs, TIME_LIMIT):   
        self.df = pd.read_csv(INPUT_FILE)
        self.job_ids = self.df.iloc[:num_jobs, 0].tolist()
        self.processing_times = self.df.iloc[:num_jobs, 1:].values
        self.num_jobs = num_jobs
        self.num_machines = self.processing_times.shape[1]
        self.TIME_LIMIT = TIME_LIMIT
        self.t_max = 10000
        self.N_p = 50
        self.n_tip = 10000
        self.crossover_probability = 0.9
        self.breeding_probability = 0.5
        self.breeder_group_ratio = 0.2
        self.history = []
        self.x = []
        self.y = []

    def calculate_makespan(self, sequence):
        completion_times = np.zeros((self.num_jobs, self.num_machines))
        for i, job in enumerate(sequence):
            for m in range(self.num_machines):
                if i == 0 and m == 0:
                    completion_times[i, m] = self.processing_times[job, m]
                elif i == 0:
                    completion_times[i, m] = completion_times[i, m-1] + self.processing_times[job, m]
                elif m == 0:
                    completion_times[i, m] = completion_times[i-1, m] + self.processing_times[job, m]
                else:
                    completion_times[i, m] = max(completion_times[i-1, m], completion_times[i, m-1]) + self.processing_times[job, m]
        return completion_times[-1, -1]

    def opposite_learning(self, sequence):
        return [self.num_jobs - 1 - x for x in sequence]

    def neighborhood_search(self, sequence):
        method_list = ['single_swap', 'single_insert', 'double_swap', 'double_insert', 'reverse']
        selected_method = random.choice(method_list)
        result_seq = sequence.copy()

        if selected_method == 'single_swap':
            i, j = random.sample(range(self.num_jobs), 2)
            result_seq[i], result_seq[j] = result_seq[j], result_seq[i]
        elif selected_method == 'single_insert':
            i, j = random.sample(range(self.num_jobs), 2)
            job = result_seq.pop(i)
            result_seq.insert(j, job)
        elif selected_method == 'double_swap':
            i, j, k, l = random.sample(range(self.num_jobs), 4)
            result_seq[i], result_seq[j] = result_seq[j], result_seq[i]
            result_seq[k], result_seq[l] = result_seq[l], result_seq[k]
        elif selected_method == 'double_insert':
            i, j, k = random.sample(range(self.num_jobs), 3)
            job1, job2 = result_seq[i], result_seq[j]
            result_seq = [x for x in result_seq if x not in [job1, job2]]
            result_seq.insert(k, job1)
            result_seq.insert(k+1, job2)
        elif selected_method == 'reverse':
            i, j = sorted(random.sample(range(self.num_jobs), 2))
            result_seq[i:j+1] = reversed(result_seq[i:j+1])

        return result_seq

    def crossover(self, p1_seq, p2_seq):
        crossover_point = random.randint(1, self.num_jobs - 1)
        child1 = p1_seq[:crossover_point] + [x for x in p2_seq if x not in p1_seq[:crossover_point]]
        child2 = p2_seq[:crossover_point] + [x for x in p1_seq if x not in p2_seq[:crossover_point]]
        return child1, child2

    def sim(self):
        population = [random.sample(range(self.num_jobs), self.num_jobs) for _ in range(self.N_p)]

        fitness = [self.calculate_makespan(seq) for seq in population]

        population = [x for _, x in sorted(zip(fitness, population))]
        
        best_makespan = float('inf')
        best_sequence = None
        
        start_time = datetime.datetime.now()
        iteration = 0
        no_improvement = 0
        
        while iteration < self.t_max and no_improvement < self.n_tip:
            breeder_size = int(self.N_p * self.breeder_group_ratio)
            breeder_group = population[:breeder_size]
            worker_group = population[breeder_size:]

            for breeder in breeder_group:
                if random.random() < self.breeding_probability:
                    new_sequence = self.neighborhood_search(breeder)
                    new_makespan = self.calculate_makespan(new_sequence)
                    if new_makespan < self.calculate_makespan(breeder):
                        breeder[:] = new_sequence

            for worker in worker_group:
                if random.random() < self.crossover_probability:
                    partner = random.choice(population)
                    child1, child2 = self.crossover(worker, partner)
                    child1_makespan = self.calculate_makespan(child1)
                    child2_makespan = self.calculate_makespan(child2)
                    worker_makespan = self.calculate_makespan(worker)
                    if child1_makespan < worker_makespan or child2_makespan < worker_makespan:
                        worker[:] = child1 if child1_makespan < child2_makespan else child2

            fitness = [self.calculate_makespan(seq) for seq in population]
            
            min_makespan = min(fitness)
            if min_makespan < best_makespan:
                best_makespan = min_makespan
                best_sequence = population[fitness.index(min_makespan)]
                no_improvement = 0
            else:
                no_improvement += 1

            population = [x for _, x in sorted(zip(fitness, population))]
            
            self.history.append(best_makespan)
            self.x.append(iteration)
            self.y.append(best_makespan)
            
            iteration += 1
            
            if (datetime.datetime.now() - start_time).total_seconds() > self.TIME_LIMIT:
                break
        
        elapsed_time = datetime.datetime.now() - start_time
        return best_makespan, best_sequence, elapsed_time, self.x, self.y

# ========================================= Rollout =========================================
def calculate_makespan(schedule):
    return max(max(job[2] for job in machine) for machine in schedule if machine)

def apply_spt_lpt(jobs, rule):
    if rule == 'SPT':
        return sorted(jobs, key=lambda x: sum(x[1]))
    elif rule == 'LPT':
        return sorted(jobs, key=lambda x: sum(x[1]), reverse=True)
    else:
        raise ValueError("Invalid rule. Use 'SPT' or 'LPT'.")

def simulate_remaining_jobs(fixed_jobs, remaining_jobs, n_machines, rule):
    schedule = [[] for _ in range(n_machines)]
    
    for job in fixed_jobs:
        job_id, processing_times = job
        for m in range(n_machines):
            if m == 0:
                start_time = 0 if not schedule[m] else schedule[m][-1][2]
            else:
                start_time = max(schedule[m-1][-1][2] if schedule[m-1] else 0,
                                 schedule[m][-1][2] if schedule[m] else 0)
            end_time = start_time + processing_times[m]
            schedule[m].append((job_id, start_time, end_time))
    
    sorted_remaining = apply_spt_lpt(remaining_jobs, rule)
    for job in sorted_remaining:
        job_id, processing_times = job
        for m in range(n_machines):
            if m == 0:
                start_time = 0 if not schedule[m] else schedule[m][-1][2]
            else:
                start_time = max(schedule[m-1][-1][2] if schedule[m-1] else 0,
                                 schedule[m][-1][2] if schedule[m] else 0)
            end_time = start_time + processing_times[m]
            schedule[m].append((job_id, start_time, end_time))
    
    return schedule

def simulate_job(args):
    fixed_jobs, job, remaining_jobs, n_machines, rule = args
    temp_fixed_jobs = fixed_jobs + [job]
    temp_remaining_jobs = [j for j in remaining_jobs if j != job]
    
    temp_schedule = simulate_remaining_jobs(temp_fixed_jobs, temp_remaining_jobs, n_machines, rule)
    temp_makespan = calculate_makespan(temp_schedule)
    
    return job, temp_makespan

def job_scheduling(job_ids, processing_times, rule, time_limit):
    n_jobs, n_machines = processing_times.shape
    jobs = list(zip(job_ids, [list(times) for times in processing_times]))
    
    fixed_jobs = []
    best_makespan = float('inf')
    best_schedule = None
    
    start_time = time.time()
    
    with Pool(processes=cpu_count()) as pool:
        for iteration in range(n_jobs):
            remaining_jobs = [j for j in jobs if j not in fixed_jobs]
            
            sim_args = [(fixed_jobs, job, remaining_jobs, n_machines, rule) for job in remaining_jobs]
            results = pool.map(simulate_job, sim_args)
            
            best_job, best_job_makespan = min(results, key=lambda x: x[1])
            
            fixed_jobs.append(best_job)
            
            if best_job_makespan < best_makespan:
                best_makespan = best_job_makespan
                best_schedule = simulate_remaining_jobs(fixed_jobs, remaining_jobs, n_machines, rule)
            
            if time.time() - start_time > time_limit:
                break
    
    return best_schedule, best_makespan, [job[0] for job in fixed_jobs]

# ========================================= GA =============================================
class GA():
    def __init__(self, INPUT_FILE, num_jobs, TIME_LIMIT):
        self.df = pd.read_csv(INPUT_FILE)
        self.processing_times = self.df.iloc[:num_jobs, 1:].values  
        self.num_jobs = num_jobs
        self.job_ids = self.df.iloc[:self.num_jobs, 0].tolist()    
        self.num_machines = self.processing_times.shape[1]  
        self.parameters = {
            'MUT': 0.8,
            'END': 0.01,
            'POP_SIZE': 10000,
            'NUM_OFFSPRING': 10,
            'NUM_OF_ITERATION': 50,
        }
        self.mutation_rate = self.parameters['MUT']
        self.generation = 0
        self.schedule = []
        self.start_time = time.time()
        self.TIME_LIMIT = TIME_LIMIT

    def calculate_makespan(self, sequence):
        num_jobs = len(sequence)
        completion_times = np.zeros((num_jobs, self.num_machines))

        for j in range(num_jobs):
            for m in range(self.num_machines):
                if j == 0 and m == 0:
                    completion_times[j][m] = self.processing_times[sequence[j]][m]
                elif j == 0:
                    completion_times[j][m] = completion_times[j][m - 1] + self.processing_times[sequence[j]][m]
                elif m == 0:
                    completion_times[j][m] = completion_times[j - 1][m] + self.processing_times[sequence[j]][m]
                else:
                    completion_times[j][m] = max(completion_times[j - 1][m], completion_times[j][m - 1]) + self.processing_times[sequence[j]][m]

        return completion_times[-1][-1]

    def neh_algorithm(self):
        total_processing_times = self.df.iloc[:self.num_jobs, 1:].sum(axis=1).tolist()

        sorted_jobs = sorted(range(self.num_jobs), key=lambda i: total_processing_times[i], reverse=True)
        job_sequence = [sorted_jobs[0]]

        for i in range(1, self.num_jobs):
            current_job = sorted_jobs[i]
            best_makespan = float('inf')
            best_position = 0

            for j in range(len(job_sequence) + 1):
                temp_sequence = job_sequence[:j] + [current_job] + job_sequence[j:]  
                makespan = self.calculate_makespan(temp_sequence)

                if makespan < best_makespan:
                    best_makespan = makespan
                    best_position = j 

            job_sequence.insert(best_position, current_job)

        return job_sequence

    def generate_random_population(self):
        population = []
        for _ in range(self.parameters['POP_SIZE'] - 1):
            sequence = np.random.permutation(self.num_jobs).tolist()
            population.append(sequence)
        return population

    def get_fitness(self, chromosome):
        makespan = self.calculate_makespan(chromosome)
        return makespan 

    def ranneh_algorithm(self):
        neh_sequence = self.neh_algorithm()
        random_population = self.generate_random_population()

        population = [[neh_sequence, self.get_fitness(neh_sequence)]]  
        for sequence in random_population:
            fitness = self.get_fitness(sequence)
            population.append([sequence, fitness])  

        return population
    
    def uniform_selection(self, population, num_parents=2):
        selected_parents = random.sample(population, num_parents)  
        return selected_parents
    
    def ordered_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        size = len(parent1)  
        child = [None] * size 
        
        start, end = sorted(np.random.choice(range(size), 2, replace=False))
        
        child[start:end + 1] = parent1[start:end + 1]
        
        fill_values = [item for item in parent2 if item not in child]
        
        fill_pos = [i for i in range(size) if child[i] is None]
      
        for i, value in zip(fill_pos, fill_values):
            child[i] = value
        return child 

    def swap_mutation(self, sequence: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.mutation_rate:
            idx1, idx2 = np.random.choice(range(len(sequence)), 2, replace=False)
            sequence[idx1], sequence[idx2] = sequence[idx2], sequence[idx1]  
        return sequence
    
    def replacement_operator(self, population, offsprings):
        population.extend(offsprings)
        population.sort(key=lambda x: x[1], reverse=False)
        new_population = population[:self.parameters['POP_SIZE']]
        return new_population
     
    def search(self):
        population = self.ranneh_algorithm()  
        
        while True:
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= self.TIME_LIMIT:
                break

            offsprings = []
            for _ in range(self.parameters["NUM_OFFSPRING"]):
                mom_ch, dad_ch = self.uniform_selection(population)  
                offspring = self.ordered_crossover(mom_ch[0], dad_ch[0])  
                offspring = self.swap_mutation(offspring)
                offsprings.append([offspring, self.get_fitness(offspring)])

            population = self.replacement_operator(population, offsprings)
            self.generation += 1

        best_solution = population[0]
        return best_solution[1], best_solution[0], self.generation

#========================================= IGA =============================================
class IGA():
    def __init__(self, INPUT_FILE, T, num_jobs): 
        self.df = pd.read_csv(INPUT_FILE)
        self.processing_times = self.df.iloc[:, 1:].values
        self.num_jobs = num_jobs  # 작업 수
        self.num_machines = self.processing_times.shape[1]  # 기계 수
        self.job_ids = self.df.iloc[:self.num_jobs, 0].tolist()    
        self.schedule = []
        self.T = 0.1
        self.temperature = self.calculate_temperature()

    def calculate_temperature(self):
        total_processing_time = np.sum(self.processing_times)
        temperature = self.T * total_processing_time / (self.num_jobs * self.num_machines * 10)
        return temperature    

    def calculate_makespan(self, sequence):
        num_jobs = len(sequence)
        completion_times = np.zeros((num_jobs, self.num_machines))
        
        # Makespan calculation (unchanged)
        for j in range(num_jobs):
            for m in range(self.num_machines):
                if j == 0 and m == 0:
                    completion_times[j][m] = self.processing_times[sequence[j]][m]
                elif j == 0:
                    completion_times[j][m] = completion_times[j][m - 1] + self.processing_times[sequence[j]][m]
                elif m == 0:
                    completion_times[j][m] = completion_times[j - 1][m] + self.processing_times[sequence[j]][m]
                else:
                    completion_times[j][m] = max(completion_times[j - 1][m], completion_times[j][m - 1]) + self.processing_times[sequence[j]][m]
        
        return completion_times[-1][-1]  

    # NEH algorithm function (unchanged)
    def neh_algorithm(self):
        total_processing_times = self.df.iloc[:, 1:].sum(axis=1).tolist()
        sorted_jobs = sorted(range(self.num_jobs), key=lambda i: total_processing_times[i], reverse=True)
        job_sequence = [sorted_jobs[0]]

        for i in range(1, self.num_jobs):
            current_job = sorted_jobs[i]
            best_makespan = float('inf')
            best_position = 0

            for j in range(len(job_sequence) + 1):
                temp_sequence = job_sequence[:j] + [current_job] + job_sequence[j:]  
                makespan = self.calculate_makespan(temp_sequence)

                if makespan < best_makespan:
                    best_makespan = makespan
                    best_position = j 

            job_sequence.insert(best_position, current_job)

        return job_sequence
    
    def destruction(self, job_sequence):
        num_jobs_to_destroy = max(1, int(len(job_sequence) * 0.1))  
        Pr = random.sample(job_sequence, num_jobs_to_destroy)  
        Pd = copy.deepcopy(job_sequence)  
        for job in Pr:
            Pd.remove(job)  
        return Pd, Pr  

    def construction(self, Pd, Pr):
        for job in Pr: 
            best_makespan = float('inf') 
            best_position = 0     
            previous_makespan = self.calculate_makespan(Pd)  

            for position in range(len(Pd) + 1):  
                temp_sequence = Pd[:position] + [job] + Pd[position:]  
                makespan = self.calculate_makespan(temp_sequence)  

                if makespan < best_makespan:
                    best_makespan = makespan  
                    best_position = position  

            Pd.insert(best_position, job)

        return Pd 
    
    def local_search(self, best_sequence, time_limit):
        best_makespan = self.calculate_makespan(best_sequence)
        best_solution = best_sequence[:]
        improved = True

        start_time = time.time()
        while improved:
            if time.time() - start_time > time_limit:
                #print("시간 제한을 초과하여 local search를 중단합니다.")
                break

            improved = False
            jobs = best_solution[:]  
            positions = range(len(best_solution) + 1)

            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = []
                job_to_insert_info = []

                for i in range(len(jobs)):  
                    job_to_insert = jobs[i]  
                    for position in positions:  
                        job_to_insert_info.append((job_to_insert, position))
                        results.append(pool.apply_async(self.evaluate_insertion, (job_to_insert, best_solution, position)))

                for idx, result in enumerate(results):
                    job_to_insert, position = job_to_insert_info[idx]
                    new_makespan = result.get()

                    if new_makespan < best_makespan:
                        best_makespan = new_makespan
                        temp_solution = best_solution[:]
                        temp_solution.remove(job_to_insert)
                        new_solution = temp_solution[:position] + [job_to_insert] + temp_solution[position:]
                        best_solution = new_solution[:]
                        improved = True

        return best_solution, best_makespan

    def optimize_schedule(self):
        start_time = time.time()  
       
        current_sequence = self.neh_algorithm()
        current_makespan = self.calculate_makespan(current_sequence)
        
        #print("초기 스케줄:", current_sequence)
        #print("초기 makespan:", current_makespan)

        best_makespan = current_makespan
        best_sequence = current_sequence.copy()

        total_time_limit = 1800
        search_time_limit = 900

        while time.time() - start_time < total_time_limit:
            iteration_start_time = time.time()

            Pd, Pr = self.destruction(best_sequence)
            new_sequence = self.construction(Pd, Pr)
            new_makespan = self.calculate_makespan(new_sequence)

            if new_makespan < best_makespan:
                #print(f"새로운 스케줄을 찾았습니다. Makespan: {new_makespan}")
                best_sequence = new_sequence
                best_makespan = new_makespan
            #else:
                #print("새로운 스케줄이 현재 스케줄보다 좋지 않습니다. 다시 시도합니다.")

            iteration_duration = time.time() - iteration_start_time
            if iteration_duration > search_time_limit:
                #print("한 번의 destruction/construction 단계에서 시간이 너무 오래 소요되었습니다. 다음 반복으로 넘어갑니다.")
                continue  

            remaining_time = total_time_limit - (time.time() - start_time)
            if remaining_time < 60:  
                #print("남은 시간이 부족하여 탐색을 종료합니다.")
                break

            #print("local_search를 실행합니다.")
            improved_solution, improved_makespan = self.local_search(best_sequence, remaining_time)

            if improved_makespan < best_makespan:
                best_sequence = improved_solution
                best_makespan = improved_makespan
                #print("로컬 서치 후 해가 더 좋습니다.")

            #if best_makespan < self.calculate_makespan(current_sequence):
                #print("새로운 최적 해를 찾았습니다.")
            #else:
                #print("기존 해가 최적입니다.")
            
        end_time = time.time()  # End timer

        # Total execution time output
        total_time = end_time - start_time
        return best_makespan, best_sequence

    def evaluate_insertion(self, job_to_insert, best_solution, position):
        temp_solution = best_solution[:]
        temp_solution.remove(job_to_insert)
        new_solution = temp_solution[:position] + [job_to_insert] + temp_solution[position:]
        new_makespan = self.calculate_makespan(new_solution)
        return new_makespan

if __name__ == "__main__":
    for index, n in enumerate(NUM_JOBS):
        if index >= len(INPUT_FILES):
            print(f"No input file for NUM_JOBS index {index}.")
            continue

        best_makespan = float('inf')
        best_sequence = None
        best_model = None

        INPUT_FILE = INPUT_FILES[index]
        df = pd.read_csv(INPUT_FILE)

        job_ids = df.iloc[:, 0].tolist()
        processing_times = df.iloc[:, 1:].values
        
        job_ids = job_ids[:n]
        processing_times = processing_times[:n]
        num_machines = processing_times.shape[1]

        # rollout-SPT 실행
        RULE = 'SPT' 
        SPT_schedule, SPT_makespan, SPT_optimal_job_sequence = job_scheduling(job_ids, processing_times, RULE, TIME_LIMIT)
        if SPT_makespan < best_makespan:
            best_makespan, best_sequence, best_model = SPT_makespan, SPT_optimal_job_sequence, 'SPT'

        # rollout-LPT 실행
        RULE = 'LPT' 
        LPT_schedule, LPT_makespan, LPT_optimal_job_sequence = job_scheduling(job_ids, processing_times, RULE, TIME_LIMIT)
        if LPT_makespan < best_makespan:
            best_makespan, best_sequence, best_model = LPT_makespan, LPT_optimal_job_sequence, 'LPT'

        # # FlowShopNMA 실행
        nma = FlowShopNMA(INPUT_FILE, n, TIME_LIMIT)
        NMA_makespan, NMA_optimal_job_sequence, elapsed_time, x, y = nma.sim()
        if NMA_makespan < best_makespan:
            best_makespan, best_sequence, best_model = NMA_makespan, NMA_optimal_job_sequence, 'NMA'

        # GA 실행
        ga = GA(INPUT_FILE, n, TIME_LIMIT)
        GA_makespan, GA_optimal_job_sequence, GA_generations = ga.search()
        if GA_makespan < best_makespan:
            best_makespan, best_sequence, best_model = GA_makespan, GA_optimal_job_sequence, 'GA'

        # IGA 실행
        iga = IGA(INPUT_FILE='t_500_20_mon.csv', T=0.1, num_jobs=n)
        IGA_makespan, IGA_optimal_job_sequence = iga.optimize_schedule()
        if IGA_makespan < best_makespan:
            best_makespan, best_sequence, best_model = IGA_makespan, IGA_optimal_job_sequence, 'IGA'

        print(f"\nNUM_JOBS {n} Final Results:")
        print(f"Best Model: {best_model}")
        print(f"Best Makespan: {best_makespan}")
        print(f"Optimal job sequence: {','.join(map(str, [job_ids[job] for job in best_sequence]))}")

        print("\nAll Models Results:")
        print(f"SPT Makespan: {SPT_makespan}")
        print(f"LPT Makespan: {LPT_makespan}")
        print(f"NMA Makespan: {NMA_makespan}")
        print(f"GA Makespan: {GA_makespan}")
        print(f"IGA Makespan: {IGA_makespan}")
