import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # New import for colormaps
import itertools
import matplotlib.patches as mpatches

# 매개변수
INPUT_FILE = 't_500_20_mon.csv'  # 입력 파일 이름
n = 500
 # 처리할 작업 수

# 파일 불러오기
df = pd.read_csv(INPUT_FILE)

# Job ID와 각 공정별 생산시간 불러오기
job_ids = df.iloc[:n, 0].tolist()  # 첫번째 열은 Job ID
processing_times = df.iloc[:n, 1:].values  # 두번째 열부터는 공정별 생산시간

# 공정 개수
num_machines = processing_times.shape[1]  # M1, M2, ..., M20
M = num_machines
# 가능한 작업 순서


class JobScheduler:
    def __init__(self, rule):
        self.rule = rule
        self.processing_times = processing_times
        self.num_jobs = len(processing_times)
        self.num_machines = len(processing_times[0])
        self.time = 0
        self.makespan = 0
        self.schedule = [[] for _ in range(M)]
        self.machine_end_time = []  # 머신 끝나는 시간만
        self.job_list = [(job_id, sum(times)) for job_id, times in enumerate(processing_times)]  # (job_id, 총 처리 시간)

    def schedule_jobs(self):
        while self.job_list:
            job_sequence = self.select_job_by_rule()
            print("Job Sequence Selected by", self.rule, ":", [job[0] for job in job_sequence])  # 작업 ID만 출력
            schedule = [[] for _ in range(M)]

            for job_id, _ in job_sequence: 
                for m in range(M):
                    processing_time = self.processing_times[job_id][m]  
                    if m == 0:  
                        start_time = 0 if not schedule[m] else schedule[m][-1][2]  
                    else:
                        start_time = max(schedule[m-1][-1][2] if schedule[m-1] else 0,
                                         schedule[m][-1][2] if schedule[m] else 0)

                    end_time = start_time + processing_time
                    schedule[m].append((job_id, start_time, end_time))

            self.makespan = self.calculate_makespan(schedule)
            self.schedule = schedule  # 최종 스케줄 저장
            
            # job_list에서 처리한 작업 제거
            self.job_list = [job for job in self.job_list if job[0] not in [j[0] for j in job_sequence]]

    def calculate_makespan(self, schedule):
        self.machine_end_time = []
        for machine in schedule:
            if machine:  
                self.machine_end_time.append(machine[-1][2])  
        return max(self.machine_end_time) if self.machine_end_time else 0

    def select_job_by_rule(self):
        if self.rule == 'SPT':
            # SPT: 총 처리 시간이 가장 짧은 작업부터 선택
            sorted_jobs = sorted(self.job_list, key=lambda x: x[1])  
        #elif self.rule == 'LPT':
            # LPT: 총 처리 시간이 가장 긴 작업부터 선택
            #sorted_jobs = sorted(self.job_list, key=lambda x: x[1], reverse=True)

        return sorted_jobs

def scheduling(rule):
    scheduler = JobScheduler(rule)
    scheduler.schedule_jobs()
    print(f"Rule: {rule}")
    print("Total Makespan:", scheduler.makespan)

    # Gantt 차트 생성: 모든 작업 ID 수집
    job_ids = [job_id for machine_schedule in scheduler.schedule for job_id, _, _ in machine_schedule]
    create_gantt_chart(scheduler.schedule, job_ids)

def create_gantt_chart(schedule, job_ids):
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.get_cmap("tab20")
    colors = {job_id: cmap(i % 20) for i, job_id in enumerate(set(job_ids))}
    
    for m, machine_schedule in enumerate(schedule):
        for job_id, start_time, end_time in machine_schedule:
            ax.broken_barh([(start_time, end_time - start_time)], (m * 10, 9), 
                           facecolors=colors[job_id], edgecolor='black')
    
    ax.set_yticks([10 * i + 5 for i in range(len(schedule))])
    ax.set_yticklabels([f"Machine {i+1}" for i in range(len(schedule))])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    plt.title('Gantt Chart for Flow Shop Scheduling')
    
    legend_handles = [mpatches.Patch(color=colors[job_id], label=f'Job {job_id}') for job_id in set(job_ids)]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # SPT rule로 스케줄링
    scheduling('SPT')

    # LPT rule로 스케줄링
    #scheduling('LPT')
