import itertools

# 작업 수
n = 3  # 작업 수
# 각 작업의 각 머신에서 소요되는 공정 시간
processing_times = [
    [10, 20, 30],  # Job 1의 각 머신에서의 공정 시간 (M1, M2, M3 순서)
    [15, 25, 35],  # Job 2의 각 머신에서의 공정 시간
    [12, 22, 32],  # Job 3의 각 머신에서의 공정 시간
]
# 기계수
M = len(processing_times[0])

# 가능한 작업 순서 조합
job_list = list(itertools.permutations(range(n)))
#print(job_list)

# 스케줄 생성
def generate_schedule(job_order, processing_times, M):
    schedule =  [[] for _ in range(M)] # (job_id, start_time, end_time)
    for job_id in job_order:
        for m in range(M):
            processing_time = processing_times[job_id][m]
            #print(job_id,m,processing_time)
            if m == 0: # 첫번째 기계면
                start_time = 0 if not schedule[m] else schedule[m][-1][2] # 스케줄에 아무것도 없으면 0 있으면 마지막꺼 반환함
            else:
                start_time = max(schedule[m-1][-1][2] if schedule[m-1] else 0,
                                 schedule[m][-1][2] if schedule[m] else 0)
            end_time = start_time + processing_time
            schedule[m].append((job_id, start_time, end_time))
    return schedule

for job_order in job_list:
 schedule = generate_schedule(job_order, processing_times, M)
 #print(f"Job Order: {job_order} -> Schedule: {schedule}")


# 스케줄에 따른 makespan 계산하기 
def calculate_makespan(schedule):
    machine_end_times = [] 
    for machine in schedule:
        if machine:  
            machine_end_times.append(machine[-1][2])  
    return max(machine_end_times) if machine_end_times else 0


# 최적의 스케줄을 찾기 위한 변수 초기화
min_makespan = float('inf')
optimal_schedule = None

# 모든 작업 순서에 대해 스케줄 생성 및 makespan 계산
for job_order in job_list:
    schedule = generate_schedule(job_order, processing_times, M)
    makespan = calculate_makespan(schedule)
    
    # 최적의 makespan을 찾기
    if makespan < min_makespan:
        min_makespan = makespan
        optimal_schedule = schedule

# 최적 스케줄과 그에 대한 makespan 출력
print("Optimal Schedule:")
for m, machine_schedule in enumerate(optimal_schedule):
    print(f"Machine {m + 1}: {machine_schedule}")
print("Minimum Makespan:", min_makespan)
