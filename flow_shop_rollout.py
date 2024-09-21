import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from multiprocessing import Pool, cpu_count

def calculate_makespan(schedule):
    return max(max(job[2] for job in machine) for machine in schedule if machine)

def apply_spt_lpt(jobs, rule='SPT'):
    return sorted(jobs, key=lambda x: sum(x[1]), reverse=(rule == 'LPT'))

def simulate_remaining_jobs(fixed_jobs, remaining_jobs, n_machines):
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
    
    sorted_remaining = apply_spt_lpt(remaining_jobs, 'SPT')
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
    fixed_jobs, job, remaining_jobs, n_machines = args
    temp_fixed_jobs = fixed_jobs + [job]
    temp_remaining_jobs = [j for j in remaining_jobs if j != job]
    
    temp_schedule = simulate_remaining_jobs(temp_fixed_jobs, temp_remaining_jobs, n_machines)
    temp_makespan = calculate_makespan(temp_schedule)
    
    return job, temp_makespan

def job_scheduling(job_ids, processing_times):
    n_jobs, n_machines = processing_times.shape
    jobs = list(zip(job_ids, [list(times) for times in processing_times]))
    
    fixed_jobs = []
    best_makespan = float('inf')
    best_schedule = None
    
    with Pool(processes=cpu_count()) as pool:
        for iteration in range(n_jobs):
            print(f"\nIteration {iteration + 1}:")
            print(f"Fixed jobs: {[job[0] for job in fixed_jobs]}")
            print("Simulating remaining jobs:")
            
            remaining_jobs = [j for j in jobs if j not in fixed_jobs]
            
            sim_args = [(fixed_jobs, job, remaining_jobs, n_machines) for job in remaining_jobs]
            results = pool.map(simulate_job, sim_args)
            
            best_job, best_job_makespan = min(results, key=lambda x: x[1])
            
            print(f"Selected Job {best_job[0]} with Makespan {best_job_makespan}")
            
            fixed_jobs.append(best_job)
            
            if best_job_makespan < best_makespan:
                best_makespan = best_job_makespan
                best_schedule = simulate_remaining_jobs(fixed_jobs, remaining_jobs, n_machines)
    
    return best_schedule, best_makespan, [job[0] for job in fixed_jobs]

def create_gantt_chart(schedule, job_ids):
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.get_cmap("tab20")
    colors = {job_id: cmap(i % 20) for i, job_id in enumerate(job_ids)}
    
    for m, machine_schedule in enumerate(schedule):
        for job_id, start_time, end_time in machine_schedule:
            ax.broken_barh([(start_time, end_time - start_time)], (m * 10, 9), 
                           facecolors=colors[job_id], edgecolor='black')
    
    ax.set_yticks([10 * i + 5 for i in range(len(schedule))])
    ax.set_yticklabels([f"Machine {i+1}" for i in range(len(schedule))])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    plt.title('Gantt Chart for Flow Shop Scheduling')
    
    legend_handles = [mpatches.Patch(color=colors[job_id], label=f'Job {job_id}') for job_id in job_ids]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    df = pd.read_csv('t_500_20_mon.csv')

    job_ids = df.iloc[:, 0].tolist()
    processing_times = df.iloc[:, 1:].values

    # job과 공정 개수
    num_jobs = 51
    num_machines = processing_times.shape[1]  # M1, M2, ..., M20

    job_ids = job_ids[:num_jobs]
    processing_times = processing_times[:num_jobs]

    # 스케줄링 실행
    final_schedule, makespan, optimal_job_sequence = job_scheduling(job_ids, processing_times)

    # 결과 출력
    print(f"\nFinal Results:")
    print(f"Makespan: {makespan}")
    print(f"Optimal job sequence: {optimal_job_sequence}")

    # Gantt 차트 생성
    # create_gantt_chart(final_schedule, job_ids)