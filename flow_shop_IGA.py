import pandas as pd
import numpy as np
import random
import time
import copy

# 파라미터 설정
params = {
    'TEMPERATURE': 1.0,               # 초기 온도
    'COOLING_RATE': 0.99               # 냉각 비율
} 
num_jobs = 51
# 유전 알고리즘 클래스 정의
class IGA():
    def __init__(self, INPUT_FILE):
        self.df = pd.read_csv(INPUT_FILE)
        self.processing_times = self.df.iloc[:, 1:].values  
        self.num_jobs = num_jobs
        self.job_ids = self.df.iloc[:self.num_jobs, 0].tolist()    
        self.num_machines = self.processing_times.shape[1]  
        self.parameters = params
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
    
    def destruction(self, job_sequence):
        Pd = random.sample(job_sequence, 2)  
        Pr = copy.deepcopy(job_sequence)

        for job in Pd:
            Pr.remove(job)  
        print("선택된 작업 (제거된 작업):", Pd)  # 제거된 작업을 출력
        print("남은 작업 (새로운 스케줄):", Pr)  # 남은 작업을 출력

        return Pd, Pr  # 남은 스케줄과 제거된 작업을 반환
    
    def construction(self, Pd, Pr):
        for job in Pr:  # Pr의 각 작업에 대해 반복
            best_makespan = float('inf')
            best_position = 0
            
            for position in range(len(Pd) + 1):  # Pd의 모든 가능한 위치에서 반복
                temp_sequence = Pd[:position] + [job] + Pd[position:]  # 현재 작업을 삽입한 임시 시퀀스
                makespan = self.calculate_makespan(temp_sequence)

                # 현재 시퀀스와 그에 대한 makespan 출력
                print(f"  - 위치 {position}: {temp_sequence} -> makespan: {makespan}")

                if makespan < best_makespan:
                    best_makespan = makespan
                    best_position = position

            Pd.insert(best_position, job)  # 최적의 위치에 작업 삽입
            print(f"  - 최적 위치: {best_position}, 업데이트된 Pd: {Pd}")  # 업데이트된 Pd 출력

        return Pd  # 최종적으로 완성된 Pd 반환
    
    def local_search(self, current_solution):
        improved = True  
        best_makespan = self.calculate_makespan(current_solution)  # 현재 해의 makespan
        best_solution = current_solution[:]  # 현재 해의 복사본

        while improved:
            improved = False 

            for i in range(len(best_solution)):
                job_to_insert = best_solution[i]  

                temp_solution = best_solution[:i] + best_solution[i + 1:]
                for position in range(len(temp_solution) + 1):
                    new_solution = temp_solution[:position] + [job_to_insert] + temp_solution[position:]
                
                    new_makespan = self.calculate_makespan(new_solution)
                    if new_makespan < best_makespan:
                        best_makespan = new_makespan
                        best_solution = new_solution[:]
                        improved = True  

        return best_solution, best_makespan  

# 메인 실행 코드
if __name__ == "__main__":
    INPUT_FILE = 't_500_20_mon.csv'  # 파일 경로
    iga = IGA(INPUT_FILE)
    
    # 초기 해 생성
    initial_solution = iga.neh_algorithm()
    print("초기 해:", initial_solution)

    # 초기 해의 makespan 계산
    initial_makespan = iga.calculate_makespan(initial_solution)
    print("초기 해의 makespan:", initial_makespan)

    # 파괴 절차
    Pd, Pr = iga.destruction(initial_solution)

    # 구성 절차
    final_sequence = iga.construction(Pd, Pr)
    print("구성된 최종 해:", final_sequence)

    # 로컬 서치 적용
    optimized_solution, optimized_makespan = iga.local_search(final_sequence)
    print("로컬 서치 후 최적화된 해:", optimized_solution)
    print("최적화된 해의 makespan:", optimized_makespan)

    # 수용 기준 적용
    if optimized_makespan < initial_makespan:
        accepted_solution = optimized_solution
        print("새로운 해가 수용되었습니다.")
    else:
        # 확률적으로 새 해 수용
        acceptance_probability = np.exp(-(optimized_makespan - initial_makespan) / params['TEMPERATURE'])
        random_value = np.random.rand()  # 0과 1 사이의 무작위 값 생성

        if random_value <= acceptance_probability:
            accepted_solution = optimized_solution
            print("새로운 해가 확률적으로 수용되었습니다.")
        else:
            accepted_solution = initial_solution
            print("새로운 해가 수용되지 않았습니다. 초기 해를 유지합니다.")

    print("최종 수용된 해:", accepted_solution)
