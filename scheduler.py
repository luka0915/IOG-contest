import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import random
import copy
import datetime

# 매개변수
INPUT_FILES = [
    't_500_20_mon.csv', 
    't_500_20_tue.csv', 
    't_500_20_wed.csv', 
    't_500_20_thu.csv', 
    't_500_20_fri.csv'
    ]
   
NUM_JOBS = [10, 15, 11, 12, 13]   # 일별 생산요구량
TIME_LIMIT = 300   # NMA 시간제한

# =========================================== NMA ===========================================
class FlowShopNMA:
    def __init__(self, INPUT_FILE, num_jobs, TIME_LIMIT):   
        self.df = pd.read_csv(INPUT_FILE)  # 단일 파일 경로로 수정
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

    def plot_gantt_chart(self, sequence):
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

        fig, ax = plt.subplots(figsize=(15, 10))
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(self.num_jobs)]

        for m in range(self.num_machines):
            for i, job in enumerate(sequence):
                start_time = completion_times[i, m] - self.processing_times[job, m]
                duration = self.processing_times[job, m]
                ax.broken_barh([(start_time, duration)], (m * 10, 9), facecolors=colors[job], edgecolor='black')

        ax.set_yticks([10 * i + 5 for i in range(self.num_machines)])
        ax.set_yticklabels([f"Machine {i+1}" for i in range(self.num_machines)])
        ax.set_xlabel("Time")
        ax.set_ylabel("Machines")
        plt.title('Gantt Chart for Flow Shop Scheduling')
        
        legend_handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(self.num_jobs)]
        plt.legend(legend_handles, [f"Job {self.job_ids[i]}" for i in range(self.num_jobs)], 
                   bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show() 
# =========================================== NMA ===========================================


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

def job_scheduling(job_ids, processing_times, rule):
    n_jobs, n_machines = processing_times.shape
    jobs = list(zip(job_ids, [list(times) for times in processing_times]))
    
    fixed_jobs = []
    best_makespan = float('inf')
    best_schedule = None
    
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
    
    return best_schedule, best_makespan, [job[0] for job in fixed_jobs]
# ========================================= Rollout =========================================

if __name__ == "__main__":
    
    for index, n in enumerate(NUM_JOBS):
        if index >= len(INPUT_FILES):
            print(f"No input file for NUM_JOBS index {index}.")
            continue

        best_makespan = 0

        INPUT_FILE = INPUT_FILES[index]
        df = pd.read_csv(INPUT_FILE)

        job_ids = df.iloc[:, 0].tolist()
        processing_times = df.iloc[:, 1:].values
        
        job_ids = job_ids[:n]
        processing_times = processing_times[:n]
        num_machines = processing_times.shape[1]

        # rollout-SPT 실행
        RULE = 'SPT' 
        SPT_schedule, SPT_makespan, SPT_optimal_job_sequence = job_scheduling(job_ids, processing_times, RULE)
        best_makespan, optimal_job_sequence, model = SPT_makespan, SPT_optimal_job_sequence, 'SPT'

        # rollout-LPT 실행
        RULE = 'LPT' 
        LPT_schedule, LPT_makespan, LPT_optimal_job_sequence = job_scheduling(job_ids, processing_times, RULE)
        if best_makespan >= LPT_makespan: 
            best_makespan, optimal_job_sequence, model = LPT_makespan, LPT_optimal_job_sequence, 'LPT'

        # FlowShopNMA 실행
        nma = FlowShopNMA(INPUT_FILE, n, TIME_LIMIT)
        NMA_makespan, NMA_optimal_job_sequence, elapsed_time, x, y = nma.sim()
        if best_makespan >= NMA_makespan:
            best_makespan, optimal_job_sequence, model = NMA_makespan, NMA_optimal_job_sequence, 'NMA'

        print(f"NUM_JOBS {n} Final Results:")
        print(f"Model: {model}")
        print(f"Makespan: {best_makespan}")
        print(f"Optimal job sequence: {','.join(map(str, [job + 1 for job in optimal_job_sequence]))}\n")
