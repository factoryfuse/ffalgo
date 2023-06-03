import random
import pygad
import numpy
import pandas as pd
import random
import seaborn as sns
from bokeh.io import show, output_notebook
from bokeh.io import output_file
from bokeh.models import ColumnDataSource, FactorRange, HoverTool
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.io.export import get_screenshot_as_png

SOL_PER_POP = 1000
NUM_GENERATIONS = 50
NUM_PARENTS_MATING = 2
PARENT_SELECTION = "rank"
CROSS_TYPE = "single_point"
MUT_TYPE = "random"                            
STOPPING_CRITERIA = f"saturate_{50}" 

def import_and_transform(file_path1):
    df = pd.read_excel(file_path1)
    num_operations = (len(df.columns) - 2) // 2

    jobs = {}
    machines = {}
    demands = {}

    for index, row in df.iterrows():
        job_id = int(row['Job'])

        operations = [row[f'Op{i + 1}'] for i in range(num_operations) if pd.notna(row[f'Op{i + 1}'])]
        jobs[job_id] = operations

        for i in range(num_operations):
            machine = row[f'Machine.{i}' if i > 0 else 'Machine']
            if pd.notna(machine):
                machine = int(machine) 
                if machine not in machines:
                    machines[machine] = []
                machines[machine].append((job_id, i + 1))

        demands[job_id] = int(row['Demand'])

    return jobs, machines, demands


def import_and_transform2(file_path2):
    df = pd.read_excel(file_path2, sheet_name=1)
    num_operations = df.shape[1] - 1 
    category = {}
    for index, row in df.iterrows():
        job_id = int(row['Job'])
        operations = [row[f'Op{i + 1}'] for i in range(num_operations) if pd.notna(row[f'Op{i + 1}'])]
        category[job_id] = operations
    return category

def load_data(file_name):
    xls = pd.ExcelFile(file_name)
    df = xls.parse(3,header=None)
    data_dict = df.set_index(df.columns[0])[df.columns[1]].to_dict()
    return data_dict

def encode_operations(jbs):
    operations = len(sum(list(jbs.values()), []))
    oprs_dictionary = {}
    for key in range(1, operations+1):
        j,old_count,count = 0,0,0
        for job in list(jbs.values()):
            count += len(job)
            j += 1
            if key <= count:
                o = key - old_count
                oprs_dictionary[key] = (j, o)
                break
            old_count += len(job)
    return oprs_dictionary

def encode_category(category):
    operations_category = {} 
    count = 0

    for key in category.keys():
        for value in category[key]:
            count += 1
            operations_category[count] = value
    return operations_category


def encode_processing_times(jbs, dems):

    operations_processing_times = {} 
    count = 0
    for key in jbs:
        for i in range(len(jbs[key])):
            jbs[key][i] *= dems[key]
    for key in jbs.keys():
        for value in jbs[key]:
            count += 1
            operations_processing_times[count] = value
    return operations_processing_times


def precedence_constraint(chrom, opers, jbs):
    oprs_prec = [[i for i in opers.keys() if opers[i][0]==j] for j in jbs.keys()]
    decoded_chromosome = decode_chromosome(chrom, opers, jbs)
    if oprs_prec == decoded_chromosome:
        is_feasible = True
    else:
        is_feasible = False

    return is_feasible


def prec_dict(prec):
    predecessors = {}
    for op in range(1, 1+len(sum(prec,[]))):
        pred = None
        for sublist in prec:
            if op in sublist:
                if sublist.index(op) > 0:
                    pred = sublist[sublist.index(op) - 1]
                break
        predecessors[op] = pred

    return predecessors


def machines_schedule(chrs, oprs, machs):
    decoded_machines = {}
    for i in machs.keys():
        decoded_machines[i] = [list(oprs.keys())[list(oprs.values()).index(i)] for i in machs[i]]
    schedule = [[] for i in range(len(machs))]
    temp_dict = {n: l for l in decoded_machines.values() for n in l}
    for i in chrs:
        machine_number = list(decoded_machines.keys())[list(decoded_machines.values()).index(temp_dict[i])]
        schedule[machine_number-1].append(i)
    return schedule

def calculate_makespan(schdl, oprs, jbs, proc_t, stp, operations_category, setup_times, return_completion_times=False):
    precedence = [[i for i in oprs.keys() if oprs[i][0]==j] for j in jbs.keys()]
    completion_times = {i: 0 for i in range(1, 27)}
    candidate_operations = [job[0] for job in precedence]
    done_operations = []
    precedence_dictionary = prec_dict(precedence)

    while sum(precedence, []) != []:
        for machine in schdl:
            for i, operation in enumerate(machine):
                if (operation not in candidate_operations) and (operation not in done_operations):
                    break

                elif (operation in done_operations):
                    pass

                elif (operation in candidate_operations):
                    pred = precedence_dictionary[operation]
                    if i == 0:
                        completion_times[operation] = completion_times[pred] + proc_t[operation] if pred else proc_t[operation]
                    else:
                        pred_completion = completion_times[pred] if pred else 0
                        if i>0:
                            last_op = machine[i-1]

                            stp = setup_times[int(operations_category[last_op]-1)][int(operations_category[operation]-1)]
                        start_time = (stp/5) + max(completion_times[machine[i-1]], pred_completion)
                        completion_times[operation] = start_time + proc_t[operation]
                    
                    temp = {n: l for l in precedence for n in l}
                    precedence[precedence.index(temp[operation])].remove(operation)
                    candidate_operations = [job[0] for job in precedence if len(job) >= 1]
                    done_operations.append(operation)

                    break
    if return_completion_times:
        return completion_times
    else:
        makespan = max(completion_times.values())
        return makespan



def create_initial_population(nmr, oprs, jbs):
    pop = []
    while len(pop) < nmr:
        prec = [[i for i in oprs.keys() if oprs[i][0]==j] for j in jbs.keys()]
        sample = []
        while sum(prec, []) != []:
            job = random.randint(1,len(prec)) - 1
            try:
                add = prec[job][0]
                sample.append(add)
                prec[job].remove(add)
            except:
                pass
        pop.append(sample)
    return pop


def decode_chromosome(chrom, opers, jbs):
    decoded_chromosome = [[] for i in jbs.keys()]
    for i in chrom:
        j = opers[i][0]
        decoded_chromosome[j-1].append(i)
    return decoded_chromosome

def compute_final_schedule(sols, oprs, mchs):
    initial_schedule = machines_schedule(sols, oprs, mchs)
    final_schedule = {}
    for idx, machine in enumerate(initial_schedule):
        temp = []
        for opr in machine:
            temp.append(oprs[opr])
        final_schedule[idx+1] = temp
    return final_schedule

def operation_start_end_times(completion_times, processing_times, chromosome, operations, machines):
    operation_times = {}
    machine_schedule = machines_schedule(chromosome, operations, machines)

    for operation, completion_time in completion_times.items():
        start_time = completion_time - processing_times[operation]
        machine_id = None
        for machine, operations_list in enumerate(machine_schedule):
            if operation in operations_list:
                machine_id = machine + 1
                break
        operation_key = operations[operation]
        operation_times[operation_key] = {
            "start_time": start_time,
            "completion_time": completion_time,
            "machine": machine_id,
        }

    return operation_times

def plot_gantt_chart(operation_times, parts_names):
    data = []
    for operation, info in operation_times.items():
        job_id = operation[0]
        machine = info["machine"]
        start_time = info["start_time"]
        end_time = info["completion_time"]
        part_name = parts_names.get(job_id, f"Job {job_id}")  # get the part name or use the job_id if not in parts_names
        data.append([f"Machine {machine}", start_time, end_time, part_name, f"({part_name}, {operation[1]})"])

    df = pd.DataFrame(data, columns=["machine", "start_time", "end_time", "job", "operation"])
    df["duration"] = df["end_time"] - df["start_time"]

    source = ColumnDataSource(df)

    num_jobs = len(df["job"].unique())
    palette = sns.color_palette("husl", num_jobs).as_hex()

    p = figure(y_range=FactorRange(*sorted(df["machine"].unique())), x_range=(0, df["end_time"].max() + 10), width=800, height=400, title="Gantt Chart", toolbar_location="below")
    p.hbar(y="machine", left="start_time", right="end_time", height=0.8, source=source,
           color=factor_cmap("job", palette=palette, factors=df["job"].unique()))

    hover = HoverTool(tooltips=[("Operation", "@operation"), ("Duration", "@duration{0.0}")], mode="mouse", point_policy="follow_mouse")
    p.add_tools(hover)

    p.xaxis.axis_label = "Time (minutes)"
    p.yaxis.axis_label = "Machines"
    output_file("plot.html")
    # show(p)

    f = open("plot.html", "rb")
    resu = f.raw.readall()
    f.close()

    return resu



def initialize(file_bytes):
    xl = pd.ExcelFile(file_bytes)
    df = xl.parse(xl.sheet_names[2], skiprows=1, header=None, usecols=lambda x: x != 0)
    setup_times = df.values
    setup = 0

    # output_notebook()

    print('Starting Genetic Algorithm with parameters: ', 
        SOL_PER_POP, 
        NUM_GENERATIONS, 
        NUM_PARENTS_MATING, 
        PARENT_SELECTION, 
        CROSS_TYPE, 
        MUT_TYPE)

    parts_names = load_data(file_bytes)

    category = import_and_transform2(file_bytes)

    jobs, machines, demands = import_and_transform(file_bytes)

    operations = encode_operations(jobs)
    operations_category = encode_category(category)
    processing_times = encode_processing_times(jobs, demands)
    init_population = create_initial_population(SOL_PER_POP, operations, jobs)

    def fitness_func(ga_instance, solution, solution_idx):
        print(f"Generation no: {solution_idx}")

        if precedence_constraint(solution, operations, jobs):
            schedule = machines_schedule(solution, operations, machines)
            makespan = calculate_makespan(schedule, operations, jobs, processing_times, setup, operations_category, setup_times)
        else:
            makespan = numpy.inf

        fitness = 1.0 / (numpy.abs(makespan) + 0.000001)
        
        return fitness


    ga_instance = pygad.GA(num_generations = NUM_GENERATIONS, 
                        num_parents_mating = NUM_PARENTS_MATING, 
                        fitness_func = fitness_func,
                        initial_population = init_population,
                        sol_per_pop = SOL_PER_POP, 
                        num_genes = len(operations),
                        gene_type = numpy.int32,
                        gene_space=list(operations.keys()),
                        parent_selection_type = PARENT_SELECTION,
                        keep_parents = -1,
                        crossover_type = CROSS_TYPE,
                        mutation_type = MUT_TYPE,
                        mutation_by_replacement = False,
                        mutation_percent_genes = 10,
                        random_mutation_min_val = -1.0,
                        random_mutation_max_val = 1.0,
                        allow_duplicate_genes = False,
                        stop_criteria = STOPPING_CRITERIA)

    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    obtained_schedule = compute_final_schedule(solution, operations, machines)
    obtained_makespan = calculate_makespan(machines_schedule(solution, operations, machines), operations, jobs, processing_times, setup, operations_category, setup_times)

    print(f"The algorithm suggests the following schedule: {obtained_schedule}")
    print(f"The makespan by the suggested schedule is: {obtained_makespan}")

    completion_times = calculate_makespan(machines_schedule(solution, operations, machines), operations, jobs, processing_times, setup, operations_category, setup_times, return_completion_times=True)
    operation_times = operation_start_end_times(completion_times, processing_times, solution, operations, machines)

    png_res = plot_gantt_chart(operation_times,parts_names)    

    print(ga_instance.best_solution_generation)

    return png_res
