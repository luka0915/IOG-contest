import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import time

# 매개변수
INPUT_FILE = 't_500_20_mon.csv'  # 입력 파일 이름
NUM_JOBS = 150  # 처리할 작업 수
TIME_LIMIT = 1800  # 최적화 시간 제한 (초)

# 파일 불러오기
df = pd.read_csv(INPUT_FILE)

# Job ID와 각 공정별 생산시간 불러오기
job_ids = df.iloc[:NUM_JOBS, 0].tolist()  # 첫번째 열은 Job ID
processing_times = df.iloc[:NUM_JOBS, 1:].values  # 두번째 열부터는 공정별 생산시간

# 공정 개수
num_machines = processing_times.shape[1]  # M1, M2, ..., M20

# 모델 생성
model = gp.Model("flow_shop_makespan")

# 변수 생성
C = model.addVars(NUM_JOBS, num_machines, vtype=GRB.CONTINUOUS, name="CompletionTime")
S = model.addVars(NUM_JOBS, num_machines, vtype=GRB.CONTINUOUS, name="StartTime")
makespan = model.addVar(vtype=GRB.CONTINUOUS, name="makespan")

# Job 순서를 결정하는 이진 변수
X = model.addVars(NUM_JOBS, NUM_JOBS, vtype=GRB.BINARY, name="X")

# 목적 함수: makespan 최소화
model.setObjective(makespan, GRB.MINIMIZE)

# 제약 조건 추가
# 각 공정에서 완료시간은 시작시간 + 소요시간
for j in range(NUM_JOBS):
    for m in range(num_machines):
        model.addConstr(C[j, m] == S[j, m] + processing_times[j][m])
        
        # 첫 번째 공정은 0 이상에서 시작
        if m == 0:
            model.addConstr(S[j, m] >= 0)
        else:
            # 다음 공정은 이전 공정이 완료된 후에 시작
            model.addConstr(S[j, m] >= C[j, m - 1])

# 각 공정에서 다음 job은 이전 job의 완료 이후에 시작
for m in range(num_machines):
    for i in range(NUM_JOBS):
        for j in range(NUM_JOBS):
            if i != j:
                # Job i가 job j보다 앞서 수행될 경우
                model.addConstr(S[j, m] >= C[i, m] - (1 - X[i, j]) * 1e6)

# 순서 제약 추가 (X[i, j] + X[j, i] = 1)
for i in range(NUM_JOBS):
    for j in range(i + 1, NUM_JOBS):
        model.addConstr(X[i, j] + X[j, i] == 1)

# makespan은 모든 job이 마지막 공정을 마친 후의 최대 시간
for j in range(NUM_JOBS):
    model.addConstr(makespan >= C[j, num_machines - 1])

# 시간 제한 설정
model.setParam('TimeLimit', TIME_LIMIT)

# 최적화 실행
model.optimize()

# 최적화 결과 출력 및 Gantt chart 데이터 생성
if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
    print(f"Best makespan found: {model.objVal}")
    print(f"Optimization status: {model.status}")
    
    # 최종 job 순서를 추출하는 함수
    def get_job_sequence():
        visited = [False] * NUM_JOBS
        sequence = []
        for _ in range(NUM_JOBS):
            for i in range(NUM_JOBS):
                if not visited[i]:
                    all_prev_done = True
                    for j in range(NUM_JOBS):
                        if X[j, i].x > 0.5 and not visited[j]:
                            all_prev_done = False
                            break
                    if all_prev_done:
                        sequence.append(i)
                        visited[i] = True
                        break
        return sequence
    
    # Job 순서 추출
    job_sequence = get_job_sequence()
    best_job_sequence = " -> ".join(str(job_ids[j]) for j in job_sequence)
    print(f"Best job sequence found: {best_job_sequence}")
    
    # Gantt chart 데이터 생성 및 표시
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(NUM_JOBS)]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 각 machine에 대해 Gantt 차트 생성
    for m in range(num_machines):
        for j in range(NUM_JOBS):
            start_time = S[j, m].x
            duration = processing_times[j][m]
            ax.broken_barh([(start_time, duration)], (m * 10, 9), facecolors=(colors[j]), edgecolor='black')
    
    # y축 레이블을 machine 번호로 설정
    ax.set_yticks([10 * i + 5 for i in range(num_machines)])
    ax.set_yticklabels([f"Machine {i+1}" for i in range(num_machines)])
    
    # x축 레이블
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    
    # 간트 차트 제목
    plt.title('Gantt Chart for Flow Shop Scheduling')
    
    # Job마다 색상 범례 추가
    legend_handles = [mpatches.Patch(color=colors[i], label=f'Job {job_ids[i]}') for i in range(NUM_JOBS)]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 간트 차트 표시
    plt.tight_layout()
    plt.show()

else:
    print("No solution found within the time limit!")
