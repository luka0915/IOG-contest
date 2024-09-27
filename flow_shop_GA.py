import pandas as pd
import numpy as np
import random
import time

# 파라미터 설정
params = {
    'MUT': 0.8,                        # 변이 확률 (%)
    'END': 0.01,                       # 수렴 종료 비율
    'POP_SIZE': 10000,                    # 해 집단 크기 (10 ~ 100)
    'NUM_OFFSPRING': 10,               # 자식 개체 수
    'NUM_OF_ITERATION': 50,            # 반복 횟수 # 시간으로 바꿀 예정
} 

# 유전 알고리즘 클래스 정의
class GA():
    def __init__(self, INPUT_FILE):
        self.df = pd.read_csv(INPUT_FILE)
        self.processing_times = self.df.iloc[:, 1:].values  
        self.num_jobs = self.processing_times.shape[0] 
        self.job_ids = self.df.iloc[:self.num_jobs, 0].tolist()    
        self.num_machines = self.processing_times.shape[1]  
        self.parameters = params
        self.mutation_rate = params['MUT']
        self.generation = 0
        self.schedule = []
        self.start_time = time.time() 

    # makespan 계산 함수
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

        return completion_times[-1][-1]  # 마지막 작업의 마지막 기계에서의 완료 시간

    # NEH 알고리즘 함수
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

    def generate_random_population(self):
        population = []
        for _ in range(self.parameters['POP_SIZE'] - 1):  # NEH로 만든 한 개의 해를 제외한 나머지 해 생성
            sequence = np.random.permutation(self.num_jobs).tolist()
            population.append(sequence)
        return population

    # 적합도 계산 함수
    def get_fitness(self, chromosome):
        makespan = self.calculate_makespan(chromosome)
        return makespan 

    # RanNEH 알고리즘
    def ranneh_algorithm(self):
        neh_sequence = self.neh_algorithm()
        print(f"NEH 최적 해: {neh_sequence}")
        random_population = self.generate_random_population()

        population = [[neh_sequence, self.get_fitness(neh_sequence)]]  
        for sequence in random_population:
            fitness = self.get_fitness(sequence)
            population.append([sequence, fitness])  

        for idx, (chromosome, fitness) in enumerate(population):
            print(f"해집단 {idx + 1}의 작업 시퀀스: {chromosome}, Makespan: {fitness}")

        return population
    
    def uniform_selection(self, population, num_parents=2):
        selected_parents = random.sample(population, num_parents)  
        return selected_parents
    
    def ordered_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        size = len(parent1)  
        child = [None] * size 
        
        start, end = sorted(np.random.choice(range(size), 2, replace=False))
        #print(f"선택된 인덱스: start={start}, end={end}") 

        child[start:end + 1] = parent1[start:end + 1]
       # print(f"자식1: {child}")  
        
        fill_values = [item for item in parent2 if item not in child]
        #print(f"남은 값: {fill_values}")  
        
        fill_pos = [i for i in range(size) if child[i] is None]
        #print(f"비어 있는 위치: {fill_pos}") 
      
        for i, value in zip(fill_pos, fill_values):
            child[i] = value
            #print(f"자식 업데이트: {child}")  
        return child 

    def swap_mutation(self, sequence: np.ndarray) -> np.ndarray:
        #print(f"변이 전 자식: {sequence}") 
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
        # 초기 집단 초기화
        population = self.ranneh_algorithm()  
        
        while True:  # 반복을 무한 루프를 통해 진행
            # 경과 시간 체크
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= 600:  # 600초 (10분) 경과 시 종료
                break

            offsprings = []
            for _ in range(self.parameters["NUM_OFFSPRING"]):
                mom_ch, dad_ch = self.uniform_selection(population)  
                offspring = self.ordered_crossover(mom_ch[0], dad_ch[0])  
                offspring = self.swap_mutation(offspring)
                offsprings.append([offspring, self.get_fitness(offspring)])

            population = self.replacement_operator(population, offsprings)
            self.generation += 1

        # 결과 출력
        print("최종 세대수: {}, 최종 해: {}, Makespan: {}".format(self.generation, population[0][0], population[0][1]))

# 메인 실행
if __name__ == "__main__":
    INPUT_FILE = r"D:\새 폴더\PSFP\t_500_20_mon.csv"
    ga = GA(INPUT_FILE)
    ga.search()
