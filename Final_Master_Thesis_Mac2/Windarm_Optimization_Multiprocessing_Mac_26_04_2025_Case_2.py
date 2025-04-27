#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 18:01:09 2025

@author: juanramonjimenezmogollon
"""

# Improved version of your wind farm layout optimization script comparing multiprocessing with different core counts
# Using input data from the second provided code block

import multiprocessing as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch  # Importa PathPatch desde matplotlib.patches
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import os
import time
import random  # Import the random module

# Load site and turbine
site = Hornsrev1Site()
wt = V80()
windFarmModel = Bastankhah_PorteAgel_2014(site, wt, k=0.0324555)

# --- Input data from the second code block ---
wt_x, wt_y = site.initial_position.T.tolist()
WT_Num = len(wt_x)
aep_ref = round(float(windFarmModel(wt_x, wt_y).aep().sum()), 4)
WT_Rad = 40
Iter_Length = 10
Iter_Num = 1
Gen_Num = 1
Border_Margin = 1.05
POPULATION_MULTIPLIER = 10 # Vamos a ajustar el tamaño de la población
# --- End of input data ---

# Define output directory
output_dir = "optimization_results_comparison_parallel_fitness"
os.makedirs(output_dir, exist_ok=True)

# Fitness function (sin cambios)
def fitness(coords):
    wt_x, wt_y = coords
    penalty = 0
    for i in range(WT_Num):
        x1, y1 = wt_x[i], wt_y[i]
        for j in range(i + 1, WT_Num):
            x2, y2 = wt_x[j], wt_y[j]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance <= 5 * 2 * WT_Rad:
                penalty += -abs(distance - 5 * 2 * WT_Rad) * 100
    return float(windFarmModel(wt_x, wt_y).aep().sum()) * 1e6 + penalty

# Geometry helpers (sin cambios)
def select_border_points(points):
    hull = ConvexHull(points)
    return points[hull.vertices]

def enlarge_polygon(points, factor):
    center = np.mean(points, axis=0)
    return np.array([center + (p - center) * factor for p in points])

def create_mesh_border(points):
    # Aseguramos que el último punto sea el mismo que el primero para cerrar el polígono
    if len(points) > 0 and not np.all(points[-1] == points[0]):
        points = np.vstack([points, points[0]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(points) - 1)
    return Path(points, codes)

def filter_points_inside_border(border_path, point):
    return border_path.contains_point((point[0], point[1]))

# Random layout modifier - Modified to iterate through positions randomly (sin cambios)
def improve_layout(coords, length, border_path, num_iterations):
    lst1, lst2 = coords
    n = len(lst1)
    best_x, best_y = list(lst1), list(lst2)
    best_fitness = fitness((best_x, best_y))

    for _ in range(num_iterations):
        indices = list(range(n))
        random.shuffle(indices)
        for i in indices:
            current_x, current_y = list(best_x), list(best_y)
            for _ in range(3): # Small local search around each turbine
                while True:
                    new_x_i = current_x[i] + length * np.random.uniform(-1, 1)
                    new_y_i = current_y[i] + length * np.random.uniform(-1, 1)
                    if filter_points_inside_border(border_path, (new_x_i, new_y_i)):
                        break
                trial_x, trial_y = list(current_x), list(current_y)
                trial_x[i], trial_y[i] = new_x_i, new_y_i
                trial_fitness = fitness((trial_x, trial_y))
                if trial_fitness > best_fitness:
                    best_x, best_y = trial_x, trial_y
                    best_fitness = trial_fitness
    return best_x, best_y

# Función para evaluar el fitness en paralelo
def evaluate_fitness_parallel(population, num_processes=mp.cpu_count()):
    with mp.Pool(processes=num_processes) as pool:
        fitness_values = pool.map(fitness, population)
    return fitness_values

def optimize_generation(population, iter_length, border_path, num_iterations, num_processes):
    optimized_population = []
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(improve_layout, [(ind, iter_length, border_path, num_iterations) for ind in population])
        optimized_population.extend(results)
    return optimized_population

def generate_initial_population(initial_layout_x, initial_layout_y, population_size, border_path, iter_length):
    population = [(list(initial_layout_x), list(initial_layout_y))]
    for _ in range(population_size - 1):
        new_x = [x + iter_length * 0.1 * np.random.uniform(-1, 1) for x in initial_layout_x]
        new_y = [y + iter_length * 0.1 * np.random.uniform(-1, 1) for y in initial_layout_y]
        # Ensure new layouts are within the border (simplified for initial population)
        valid_layout_x, valid_layout_y = [], []
        for x_i, y_i in zip(new_x, new_y):
            if filter_points_inside_border(border_path, (x_i, y_i)):
                valid_layout_x.append(x_i)
                valid_layout_y.append(y_i)
            else:
                # Fallback to original if outside (can be improved)
                valid_layout_x.append(initial_layout_x[len(valid_layout_x)])
                valid_layout_y.append(initial_layout_y[len(valid_layout_y)])
        population.append((valid_layout_x[:WT_Num], valid_layout_y[:WT_Num])) # Ensure correct number of turbines
    return population

def getGenSolutions_multiprocess(initial_layout_x, initial_layout_y, iter_length, iter_num, gen_num, border_margin, num_processes, results_file):
    print(f"Running optimization with {num_processes} processes (parallel fitness)...")
    results_file.write(f"\n--- Running with {num_processes} Cores (Parallel Fitness) ---\n")
    points = np.array(list(zip(initial_layout_x, initial_layout_y)))
    border_points = select_border_points(points)
    enlarged_border = enlarge_polygon(border_points, border_margin)
    circular_border_path = create_mesh_border(enlarged_border)

    if len(initial_layout_x) != len(initial_layout_y):
        error_message = f"Error: initial_layout_x ({len(initial_layout_x)}) and initial_layout_y ({len(initial_layout_y)}) tienen longitudes diferentes."
        print(error_message)
        results_file.write(error_message + "\n")
        return None, None, None, None, None, None, None, None, None

    population_size = num_processes * POPULATION_MULTIPLIER # Ajustamos el tamaño de la población
    current_population = generate_initial_population(initial_layout_x, initial_layout_y, population_size, circular_border_path, iter_length)
    fitness_values = evaluate_fitness_parallel(current_population, num_processes) # Evaluación paralela del fitness
    ranked_population = sorted(zip(fitness_values, current_population), key=lambda x: x[0], reverse=True)
    best_sol = ranked_population[0][1]
    aep_best = round(ranked_population[0][0] / 1e6, 4)
    ranked_initial = [(fitness_values[0], current_population[0])]
    initial_time = 0.0

    results_file.write("\n--- Input Parameters ---\n")
    results_file.write(f"Number of Turbines: {WT_Num}\n")
    results_file.write(f"Reference AEP: {aep_ref} GWh\n")
    results_file.write(f"Iteration Length: {Iter_Length}\n")
    results_file.write(f"Number of Iterations per Generation: {Iter_Num}\n")
    results_file.write(f"Number of Generations: {Gen_Num}\n")
    results_file.write(f"Border Margin: {Border_Margin}\n")
    results_file.write(f"Number of Cores Used: {num_processes}\n")
    results_file.write(f"Population Size: {population_size}\n\n")

    results_file.write("--- Initial Layout ---\n")
    results_file.write(f"Initial Layout AEP: {round(ranked_initial[0][0] / 1e6, 4)} GWh\n")
    results_file.write("Initial Layout Coordinates (x, y):\n")
    for x, y in zip(initial_layout_x, initial_layout_y):
        results_file.write(f"({round(x, 2)}, {round(y, 2)})\n")
    results_file.write("\n")

    Gen = []
    GenBest = []
    aep_best_series = [aep_best]
    generation_times = []

    results_file.write("--- Genetic Optimization ---\n")
    results_file.write("AEP Evolution per Generation:\n")
    results_file.write("Time per Generation (seconds):\n")

    for i in range(1, gen_num + 1):
        Gen.append(i)
        start_time = time.time()

        # Paralelizar la mejora del layout
        optimized_population = optimize_generation(current_population, iter_length, circular_border_path, iter_num, num_processes)

        # Paralelizar la evaluación del fitness
        fitness_values = evaluate_fitness_parallel(optimized_population, num_processes)
        ranked_population = sorted(zip(fitness_values, optimized_population), key=lambda x: x[0], reverse=True)
        current_best_fitness, current_best_coords = ranked_population[0]
        aep = round(current_best_fitness / 1e6, 4)

        end_time = time.time()
        elapsed_time = end_time - start_time
        generation_times.append(elapsed_time)

        if aep > aep_best:
            best_sol = current_best_coords
            aep_best = aep

        GenBest.append(best_sol)
        aep_best_series.append(aep_best)
        current_population = [ranked_population[k][1] for k in range(population_size // 2)] # Keep top half
        # Introduce some variation (can be more sophisticated)
        next_generation = current_population[:]
        while len(next_generation) < population_size:
            parent1 = random.choice(current_population)
            parent2 = random.choice(current_population)
            child_x = [(p1 + p2) / 2 + iter_length * 0.05 * np.random.uniform(-1, 1) for p1, p2 in zip(parent1[0], parent2[0])]
            child_y = [(p1 + p2) / 2 + iter_length * 0.05 * np.random.uniform(-1, 1) for p1, p2 in zip(parent1[1], parent2[1])]
            next_generation.append((child_x, child_y))
        current_population = next_generation[:population_size] # Ensure population size

        print(f"Gen {i}: AEP = {aep_best} GWh, Time = {elapsed_time:.2f} seconds (parallel fitness with {num_processes} cores)")
        results_file.write(f"Generation {i}: AEP = {aep_best} GWh, Time = {elapsed_time:.2f} (parallel fitness)\n")

    results_file.write("\n--- Final Layout ---\n")
    results_file.write(f"Final AEP: {aep_best} GWh\n")
    results_file.write("Final Layout Coordinates (x, y):\n")
    for x, y in zip(best_sol[0], best_sol[1]):
        results_file.write(f"({round(x, 2)}, {round(y, 2)})\n")
    results_file.write("\n")

    if len(best_sol[0]) != len(best_sol[1]):
        error_message = f"Error: Al finalizar getGenSolutions, best_sol[0] ({len(best_sol[0])}) y best_sol[1] ({len(best_sol[1])}) tienen longitudes diferentes."
        print(error_message)
        results_file.write(error_message + "\n")
        return None, None, None, None, None, None, None, None, None

    return best_sol, Gen, GenBest, aep_best_series, generation_times, aep_best, ranked_initial, initial_time, circular_border_path

def plot_results(wt_x_orig, wt_y_orig, final_sol, gens, gens_best, aep_list, border_path_initial, title_suffix=""):
    output_dir = "optimization_results_comparison_parallel_fitness"
    os.makedirs(output_dir, exist_ok=True)

    if len(wt_x_orig) != len(wt_y_orig):
        print(f"Error en plot_results ({title_suffix}): Las longitudes de wt_x_orig ({len(wt_x_orig)}) y wt_y_orig ({len(wt_y_orig)}) no coinciden.")
        return

    # Plot 1
    plt.figure(figsize=(10, 8))
    plt.title(f'Original (blue) and Evolutions of Generation Layouts (yellow) {title_suffix}')
    plt.plot(wt_x_orig, wt_y_orig, 'b.', label='Original Layout')
    for i, s in enumerate(gens_best):
        if isinstance(s, (tuple, list)) and len(s) == 2 and isinstance(s[0], (list, np.ndarray)) and isinstance(s[1], (list, np.ndarray)):
            plt.plot(s[0], s[1], 'y.', label=f'Generation {gens[i]} Layout' if i == 0 else "")
    if final_sol is not None:
        plt.plot(final_sol[0], final_sol[1], 'g.', label='Final Layout')
    # Plot the border
    patch = PathPatch(border_path_initial, facecolor='none', edgecolor='r', linestyle='--', linewidth=0.5, label="Optimization Border")
    plt.gca().add_patch(patch)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"original_vs_genetic_layouts{title_suffix.replace(' ', '_')}.png"))
    plt.close()

    # Plot 2 - Final vs Original
    plt.figure(figsize=(10, 8))
    plt.title(f'Original (blue) and Final (black) Layouts {title_suffix}')
    plt.plot(wt_x_orig, wt_y_orig, 'b.', label='Original Layout')
    if final_sol is not None:
        plt.plot(final_sol[0], final_sol[1], 'k.', label='Final Layout')
    # Plot the border
    patch = PathPatch(border_path_initial, facecolor='none', edgecolor='r', linestyle='--', linewidth=0.5, label="Optimization Border")
    plt.gca().add_patch(patch)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"initial_vs_final_layouts{title_suffix.replace(' ', '_')}.png"))
    plt.close()

    # Plot 3 - AEP evolution
    if len(aep_list) > len(gens):
        aep_list = aep_list[1:]  # remove initial reference to match gens

    plt.figure(figsize=(10, 6))
    plt.title(f'Evolution of improvement through generations {title_suffix}')
    plt.scatter(gens, aep_list, label='AEP Data Points')
    if len(gens) > 2:
        x_vals = np.linspace(min(gens), max(gens), 100)
        y_trend = np.poly1d(np.polyfit(gens, aep_list, 2))(x_vals)
        plt.plot(x_vals, y_trend, 'r-', label='Parabolic Trend Line')
    plt.xlabel('Generation')
    plt.ylabel('AEP (GWh)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"aep_evolution{title_suffix.replace(' ', '_')}.png"))
    plt.close()

    # Plot 4 - Delta AEP
    delta = np.diff([aep_ref] + aep_list)
    if len(delta) == len(gens):
        plt.figure(figsize=(10, 6))
        plt.title(f'Evolution of AEP improvements per generation {title_suffix}')
        plt.scatter(gens, delta, label='Delta AEP Data Points')
        if len(gens) > 2:
            x_vals = np.linspace(min(gens), max(gens), 100)
            y_trend = np.poly1d(np.polyfit(gens, delta, 2))(x_vals)
            plt.plot(x_vals, y_trend, 'r-', label='Parabolic Trend Line')
        plt.xlabel('Generation')
        plt.ylabel('Delta AEP (GWh)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"delta_aep_evolution{title_suffix.replace(' ', '_')}.png"))
        plt.close()
    else:
        print(f"Skipping Delta AEP plot ({title_suffix}) due to insufficient data.")


if __name__ == '__main__':
    output_dir_original = "optimization_results_comparison"
    os.makedirs(output_dir_original, exist_ok=True)
    results_filename_original = os.path.join(output_dir_original, "optimization_comparison_results.txt")

    output_dir_optimized = "optimization_results_comparison_parallel_fitness"
    os.makedirs(output_dir_optimized, exist_ok=True)
    results_filename_optimized = os.path.join(output_dir_optimized, "optimization_comparison_results_parallel_fitness.txt")

    num_cores_list = [2, mp.cpu_count()]
    results_dict_original = {}
    results_dict_optimized = {}

    try:
        with open(results_filename_original, "w") as f_orig:
            try:
                f_orig.write("Wind Farm Layout Optimization Results - Core Comparison (Original)\n\n")
                for num_cores in num_cores_list:
                    title_suffix_orig = f"({num_cores} Cores - Original)"
                    f_orig.write(f"\n--- Running Original with {num_cores} Cores ---\n")
                    results_from_get_gen_original = getGenSolutions_multiprocess(
                        wt_x, wt_y, Iter_Length, Iter_Num, Gen_Num, Border_Margin, num_cores, f_orig
                    )

                    if results_from_get_gen_original is not None:
                        final_sol_orig, gens_orig, gens_best_orig, aep_list_orig, generation_times_orig, final_aep_orig, ranked_initial_orig, initial_time_orig, circular_border_path_initial_orig = results_from_get_gen_original
                        aep_start_orig = round(ranked_initial_orig[0][0] / 1e6, 4)

                        results_dict_original[num_cores] = {
                            'final_sol': final_sol_orig,
                            'gens': gens_orig,
                            'gens_best': gens_best_orig,
                            'aep_list': aep_list_orig,
                            'generation_times': generation_times_orig,
                            'final_aep': final_aep_orig,
                            'aep_start': aep_start_orig,
                            'circular_border_path_initial': circular_border_path_initial_orig
                        }
                        plot_results(wt_x, wt_y, final_sol_orig, gens_orig, gens_best_orig, aep_list_orig, circular_border_path_initial_orig, title_suffix_orig)
                    else:
                        print(f"Error: getGenSolutions_multiprocess devolvió None para {num_cores} núcleos (original).")
            except Exception as e:
                print(f"Error al escribir en {results_filename_original}: {e}")
    except Exception as e:
        print(f"Error al abrir {results_filename_original}: {e}")

    try:
        with open(results_filename_optimized, "w") as f_opt:
            try:
                f_opt.write("Wind Farm Layout Optimization Results - Core Comparison (Parallel Fitness)\n\n")
                for num_cores in num_cores_list:
                    title_suffix_opt = f"({num_cores} Cores - Parallel Fitness)"
                    f_opt.write(f"\n--- Running Optimized with {num_cores} Cores (Parallel Fitness) ---\n")
                    results_from_get_gen_optimized = getGenSolutions_multiprocess(
                        wt_x, wt_y, Iter_Length, Iter_Num, Gen_Num, Border_Margin, num_cores, f_opt
                    )

                    if results_from_get_gen_optimized is not None:
                        final_sol_opt, gens_opt, gens_best_opt, aep_list_opt, generation_times_opt, final_aep_opt, ranked_initial_opt, initial_time_opt, circular_border_path_initial_opt = results_from_get_gen_optimized
                        aep_start_opt = round(ranked_initial_opt[0][0] / 1e6, 4)

                        results_dict_optimized[num_cores] = {
                            'final_sol': final_sol_opt,
                            'gens': gens_opt,
                            'gens_best': gens_best_opt,
                            'aep_list': aep_list_opt,
                            'generation_times': generation_times_opt,
                            'final_aep': final_aep_opt,
                            'aep_start': aep_start_opt,
                            'circular_border_path_initial': circular_border_path_initial_opt
                        }
                        plot_results(wt_x, wt_y, final_sol_opt, gens_opt, gens_best_opt, aep_list_opt, circular_border_path_initial_opt, title_suffix_opt)
                    else:
                        print(f"Error: getGenSolutions_multiprocess devolvió None para {num_cores} núcleos (parallel fitness).")
            except Exception as e:
                print(f"Error al escribir en {results_filename_optimized}: {e}")
    except Exception as e:
        print(f"Error al abrir {results_filename_optimized}: {e}")

    print("\n--- Comparación de Tiempos de Ejecución (Original) ---")
    for num_cores, results in results_dict_original.items():
        if 'generation_times' in results:
            total_time = sum(results['generation_times'])
            print(f"Tiempo total de ejecución original con {num_cores} núcleos: {total_time:.2f} segundos")
        else:
            print(f"No se encontraron tiempos de ejecución originales para {num_cores} núcleos.")

    print("\n--- Comparación de Tiempos de Ejecución (Parallel Fitness) ---")
    for num_cores, results in results_dict_optimized.items():
        if 'generation_times' in results:
            total_time = sum(results['generation_times'])
            print(f"Tiempo total de ejecución con paralelización del fitness ({num_cores} núcleos): {total_time:.2f} segundos")
        else:
            print(f"No se encontraron tiempos de ejecución con paralelización del fitness para {num_cores} núcleos.")

    print(f"Original results saved to: {results_filename_original}")
    print(f"Optimized results (parallel fitness) saved to: {results_filename_optimized}")