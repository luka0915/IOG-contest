import csv
import sys

def calculate_makespan(filename, job_order):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        data = list(reader)

    job_order = [job_id - 1 for job_id in job_order]

    num_machines = len(data[0]) - 1

    completion_times = [0] * num_machines

    for job_id in job_order:
        for machine in range(num_machines):
            processing_time = int(data[job_id][machine + 1])
            if machine == 0:
                completion_times[machine] += processing_time
            else:
                completion_times[machine] = max(completion_times[machine], completion_times[machine-1]) + processing_time

    makespan = completion_times[-1]
    return makespan

if __name__ == "__main__":
    filename = "t_500_20_mon.csv"
    job_order = [1, 3, 5, 4, 2]

    makespan = calculate_makespan(filename, job_order)
    print(f"Makespan: {makespan}")
