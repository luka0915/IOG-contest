import random
import copy
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 매개변수
INPUT_FILE = 't_500_20_mon.csv'
NUM_JOBS = [10, 25, 52, 53, 56, 62, 75]  # 스케줄링 목록
TIME_LIMIT = 600  # 시간 제한

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

for n in NUM_JOBS:
    # FlowShopNMA 실행
    nma = FlowShopNMA(INPUT_FILE, n, TIME_LIMIT)
    best_makespan, best_sequence, elapsed_time, x, y = nma.sim()

    print(f"NUM_JOBS: {n}")
    print(f"Best makespan: {best_makespan}")
    print(f"Best sequence: {[nma.job_ids[i] for i in best_sequence]}")
    print(f"Elapsed time: {elapsed_time}")

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b', label='Makespan')
    plt.xlabel('Iteration')
    plt.ylabel('Makespan')
    plt.title('Makespan over Iterations')
    plt.legend()
    plt.grid(True)
    # plt.show()

    # nma.plot_gantt_chart(best_sequence)
