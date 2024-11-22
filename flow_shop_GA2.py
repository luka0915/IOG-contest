import pandas as pd
import numpy as np
import random
import time

class GA():
    def __init__(self, INPUT_FILE):
        self.df = pd.read_csv(INPUT_FILE)
        self.processing_times = self.df.iloc[:, 1:].values  
        self.num_jobs = num_jobs
        self.job_ids = list(range(self.num_jobs))  # 0부터 num_jobs-1까지의 작업 ID    
        self.num_machines = self.processing_times.shape[1]  
        self.parameters = {
            'MUT': 0.3,
            'END': 0.01,
            'POP_SIZE': 100,
            'NUM_OFFSPRING': 50,
            'TOURNAMENT_SIZE': 3,
            'ELITE_SIZE': 5
        }
        self.mutation_rate = self.parameters['MUT']
        self.generation = 0
        self.schedule = []
        self.start_time = time.time()
        self.best_fitness_history = []
        self.current_best = float('inf')

    def calculate_makespan(self, sequence):
        num_jobs = len(sequence)
        completion_times = np.zeros((num_jobs, self.num_machines))
        
        completion_times[0][0] = self.processing_times[sequence[0]][0]
        for m in range(1, self.num_machines):
            completion_times[0][m] = completion_times[0][m-1] + self.processing_times[sequence[0]][m]
            
        for j in range(1, num_jobs):
            completion_times[j][0] = completion_times[j-1][0] + self.processing_times[sequence[j]][0]
            for m in range(1, self.num_machines):
                completion_times[j][m] = max(completion_times[j-1][m], completion_times[j][m-1]) + \
                                       self.processing_times[sequence[j]][m]
        
        return completion_times[-1][-1]

    def get_fitness(self, sequence):
        return self.calculate_makespan(sequence)

    def modified_neh_algorithm(self):
        total_processing_times = []
        weighted_processing_times = []
        
        for i in range(self.num_jobs):
            total_time = sum(self.processing_times[i])
            weighted_time = sum(self.processing_times[i] * np.linspace(1, 2, self.num_machines))
            total_processing_times.append(total_time)
            weighted_processing_times.append(weighted_time)
            
        sorted_jobs = sorted(range(self.num_jobs), 
                           key=lambda i: (weighted_processing_times[i], total_processing_times[i]), 
                           reverse=True)
        
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
        neh_solution = self.modified_neh_algorithm()
        population.append([neh_solution, self.get_fitness(neh_solution)])
        
        for _ in range(self.parameters['POP_SIZE'] - 1):
            if random.random() < 0.3:
                perturbed_sequence = neh_solution.copy()
                for _ in range(random.randint(1, 3)):
                    i, j = random.sample(range(self.num_jobs), 2)
                    perturbed_sequence[i], perturbed_sequence[j] = perturbed_sequence[j], perturbed_sequence[i]
                sequence = perturbed_sequence
            else:
                sequence = list(range(self.num_jobs))  # 0부터 num_jobs-1까지의 순열
                random.shuffle(sequence)  # 무작위로 섞기
            
            fitness = self.get_fitness(sequence)
            population.append([sequence, fitness])
            
        return sorted(population, key=lambda x: x[1])

    def tournament_selection(self, population):
        tournament = random.sample(population, self.parameters['TOURNAMENT_SIZE'])
        return min(tournament, key=lambda x: x[1])

    def pmx_crossover(self, parent1, parent2):
        # 자식 초기화
        child = [-1] * len(parent1)
        
        # 교차 지점 선택
        cx_points = sorted(random.sample(range(len(parent1)), 2))
        
        # 교차 구간 복사
        for i in range(cx_points[0], cx_points[1] + 1):
            child[i] = parent1[i]
        
        # 나머지 위치 채우기
        for i in range(len(parent2)):
            if cx_points[0] <= i <= cx_points[1]:
                continue
            
            current = parent2[i]
            while current in child:  # 이미 있는 값이면
                current = parent2[parent1.index(current)]  # 매핑된 값으로 대체
            child[i] = current
            
        # 누락된 값 채우기
        unused = set(range(self.num_jobs)) - set(child)
        for i, val in enumerate(child):
            if val == -1:
                child[i] = unused.pop()
                
        return child

    def insertion_mutation(self, sequence):
        if random.random() < self.mutation_rate:
            sequence = sequence.copy()  # 원본 시퀀스 보존
            pos1, pos2 = random.sample(range(len(sequence)), 2)
            value = sequence.pop(pos1)
            sequence.insert(pos2, value)
        return sequence

    def local_search(self, sequence):
        improved = True
        best_makespan = self.calculate_makespan(sequence)
        best_sequence = sequence.copy()
        
        while improved:
            improved = False
            for i in range(len(sequence)-1):
                new_sequence = best_sequence.copy()
                new_sequence[i], new_sequence[i+1] = new_sequence[i+1], new_sequence[i]
                new_makespan = self.calculate_makespan(new_sequence)
                
                if new_makespan < best_makespan:
                    best_sequence = new_sequence
                    best_makespan = new_makespan
                    improved = True
                    break
                    
        return best_sequence, best_makespan

    def search(self):
        population = self.generate_random_population()
        best_solution = min(population, key=lambda x: x[1])
        self.current_best = best_solution[1]
        stagnation_counter = 0
        
        while True:
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= 60:
                break
                
            new_population = population[:self.parameters['ELITE_SIZE']]
            
            while len(new_population) < self.parameters['POP_SIZE']:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                offspring = self.pmx_crossover(parent1[0], parent2[0])
                offspring = self.insertion_mutation(offspring)
                
                if random.random() < 0.1:
                    offspring, fitness = self.local_search(offspring)
                else:
                    fitness = self.get_fitness(offspring)
                    
                # 중복 체크
                if len(set(offspring)) == self.num_jobs:  # 중복이 없는 경우만 추가
                    new_population.append([offspring, fitness])
            
            population = sorted(new_population, key=lambda x: x[1])
            
            if population[0][1] < self.current_best:
                self.current_best = population[0][1]
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            if stagnation_counter >= 20:
                population = population[:self.parameters['ELITE_SIZE']] + \
                           self.generate_random_population()[self.parameters['ELITE_SIZE']:]
                stagnation_counter = 0
            
            self.generation += 1
            self.best_fitness_history.append(self.current_best)
            
            if self.generation % 10 == 0:
                print(f"Generation {self.generation}, Best Makespan: {self.current_best}")
        
        print(f"Final Generation: {self.generation}")
        print(f"Best Sequence: {population[0][0]}")
        print(f"Best Makespan: {population[0][1]}")
        return population[0]

if __name__ == "__main__":
    INPUT_FILE = 't_500_20_mon.csv'
    num_jobs = 25
    ga = GA(INPUT_FILE)
    ga.search()
