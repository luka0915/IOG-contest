import numpy as np
import pandas as pd
import time

def schedule_time_one_machine(M, j_seq):
   m_seq = [i for i in range(len(M[0]))]
   D = [0] * len(m_seq)
   D[0] = M[j_seq[0]][m_seq[0]]
   for i in range(1, len(m_seq)):
       D[i] = D[i - 1] + M[j_seq[0]][m_seq[i]]  
   for i in range(1, len(j_seq)):
       D[0] += M[j_seq[i]][m_seq[0]]
       for j in range(1, len(m_seq)):
           D[j] = max(D[j], D[j - 1]) + M[j_seq[i]][m_seq[j]]
   makespan = D[len(m_seq) - 1]
   return makespan

def my_sort(pt):
   seq = []
   for item in pt:
       seq.append(np.sum(item))
   seq_ = sorted(range(len(seq)), key=lambda k: seq[k], reverse=True)
   return seq_

def my_insert(seq_tmp, pt, machine_num, start, time_limit):
   if len(seq_tmp) < 2:
       return seq_tmp
   seq = seq_tmp[:2]
   for i in range(len(seq_tmp)-2):
       if time.time() - start > time_limit:  # 시간 제한 체크
           print("시간 제한 도달!")
           return seq
       new_job_index = len(seq)
       new_job = seq_tmp[new_job_index]
       det = []
       for i in range(len(seq)+1):
           tmp = seq[:]
           tmp.insert(i, new_job)
           makespan = schedule_time_one_machine(pt, tmp)
           det.append(makespan)
       det_index = det.index(min(det))
       seq.insert(det_index, new_job)
   return seq

def neh(pt, machine_num, start, time_limit):
   seq_tmp = my_sort(pt)
   seq = my_insert(seq_tmp, pt, machine_num, start, time_limit)
   seq = np.array(seq)
   makespan = schedule_time_one_machine(pt, seq)
   return makespan, seq

if __name__=='__main__':
   INPUT_FILE = 't_500_20_mon.csv'
   num_jobs = 100  # 원하는 작업 수 지정
   time_limit = 5 * 60 * 60  # 5시간을 초로 변환 (5 * 60 * 60 = 18000초)
   
   # 데이터 불러오기
   df = pd.read_csv(INPUT_FILE)
   processing_times = df.iloc[:num_jobs, 1:].values  # num_jobs만큼만 데이터 사용
   num_machines = processing_times.shape[1]  # 기계 수
   job_ids = df.iloc[:num_jobs, 0].tolist()  # 작업 ID 리스트
   
   start = time.time()
   machine_num = [num_machines]  # 기계 수에 맞게 설정
   
   try:
       makespan, sequence = neh(processing_times, machine_num, start, time_limit)
       optimal_sequence = [job_ids[i] for i in sequence]
       print(f"최적 작업 순서: {optimal_sequence}")
       print(f"Makespan: {makespan}")
       
   except Exception as e:
       print(f"오류 발생: {e}")
