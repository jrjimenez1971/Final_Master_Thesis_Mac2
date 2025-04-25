#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 18:01:09 2025

@author: juanramonjimenezmogollon
"""

# Improved version of your wind farm layout optimization script comparing multiprocessing with different core counts

import multiprocessing as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Function to load the coordinates of a Wind Farm different than Hornsrev1
def read_coordenates_excel(excel_file, column_x, column_y):
    try:
        # Read Excel file using pandas
        df = pd.read_excel(excel_file)
        # Check if the columns exist
        if column_x not in df.columns or column_y not in df.columns:
            raise ValueError(f"The columns '{column_x}' or '{column_y}' do not exist in the Excel file.")
        # Converts columns to lists of strings
        coordinates_x = ', '.join(df[column_x].astype(str).tolist())
        coordinates_y = ', '.join(df[column_y].astype(str).tolist())
        return coordinates_x, coordinates_y
    except FileNotFoundError:
        print(f"Error: The file '{excel_file}' was not found.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

# Function to convert string to list
def string_to_list(input_string):
    items = input_string.split(',')  # Split the string by commas
    result_list = []
    for item in items:
        item = item.strip()  # Remove leading/trailing whitespace
        if item:  # if item is not empty string.
            try:
                # Attempt to convert to integer first
                result_list.append(int(item))
            except ValueError:
                try:
                    # If integer conversion fails, try float
                    result_list.append(float(item))
                except ValueError:
                    # If both fail, handle as string (optional)
                    result_list.append(item) #or you can raise a ValueError
    return result_list

wt_x = []
wt_y = []
excel_file = '/Users/juanramonjimenezmogollon/Downloads/PRUEBA_PARQUE2.xlsx'
column_x = 'COORDINATE_X'
column_y = 'COORDINATE_Y'
wt_x_str, wt_y_str = read_coordenates_excel(excel_file, column_x, column_y)
wt_x = string_to_list(wt_x_str)
wt_y = string_to_list(wt_y_str)
WT_Num = len(wt_x)
aep_ref = round(float(windFarmModel(wt_x, wt_y).aep().sum()), 4)
WT_Rad = 40
Iter_Length = 30
Iter_Num = 15
Gen_Num = 3
Border_Margin = 1.05

# Define output directory
output_dir = "optimization_results_comparison"
os.makedirs(output_dir, exist_ok=True)

# Fitness function
def fitness(s):
    penalty = 0
    for i in range(WT_Num):
        x1, y1 = s[0][i], s[1][i]
        for j in range(i + 1, WT_Num):
            x2, y2 = s[0][j], s[1][j]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance <= 5 * 2 * WT_Rad:
                penalty += -abs(distance - 5 * 2 * WT_Rad) * 100
    return float(windFarmModel(s[0], s[1]).aep().sum()) * 1e6 + penalty

# Geometry helpers
def select_border_points(points):
    hull = ConvexHull(points)
    return points[hull.vertices]

def calculate_bounding_circle(points):
    if not points.any():
        return 0, 0, 0
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])
    distances = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2)
    radius = np.max(distances)
    return center_x, center_y, radius

def create_circular_border(center_x, center_y, radius, margin_factor=1.0):
    adjusted_radius = radius * margin_factor
    return center_x, center_y, adjusted_radius

def filter_points_inside_border(center_x, center_y, radius, point):
    distance_from_center = np.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
    return distance_from_center <= radius

# Random layout modifier - Modified to iterate through positions randomly
def list_random_values_one_by_one(lst1, lst2, length, border_params):
    n = len(lst1)
    lst1_rnd, lst2_rnd = lst1[:], lst2[:]
    current_fitness = fitness((lst1_rnd, lst2_rnd))
    indices = list(range(n))  # Create a list of indices
    random.shuffle(indices)  # Shuffle the indices at the beginning of each outer loop
    center_x, center_y, radius = border_params

    for i in indices:    # Iterate to modify all positions
        bestx, besty = lst1_rnd[:], lst2_rnd[:]
        for _ in range(Iter_Num):
            while True:
                newx = bestx[i] + length * np.random.uniform(-1, 1)
                newy = besty[i] + length * np.random.uniform(-1, 1)
                if filter_points_inside_border(center_x, center_y, radius, (newx, newy)):
                    break
            trialx, trialy = bestx[:], besty[:]
            trialx[i], trialy[i] = newx, newy
            if fitness((trialx, trialy)) > fitness((bestx, besty)):
                bestx, besty = trialx, trialy

        if fitness((bestx, besty)) >= current_fitness:
            lst1_rnd, lst2_rnd = bestx, besty
            current_fitness = fitness((lst1_rnd, lst2_rnd))

    return lst1_rnd, lst2_rnd

def single_solution_evaluation(args):
    return list_random_values_one_by_one(*args)

# Combined solution generator and optimizer
def getGenSolutions(initial_layout_x, initial_layout_y, iter_length, iter_num, gen_num, border_margin, num_processes, results_file):
    print(f"Running optimization with {num_processes} processes...")
    results_file.write(f"\n--- Running with {num_processes} Cores ---\n")
    points = np.array(list(zip(initial_layout_x, initial_layout_y)))
    center_x_border, center_y_border, radius_border = calculate_bounding_circle(points)
    circular_border_params = create_circular_border(center_x_border, center_y_border, radius_border, border_margin)

    # --- Agregar verificación inicial de longitudes ---
    if len(initial_layout_x) != len(initial_layout_y):
        error_message = f"Error: initial_layout_x ({len(initial_layout_x)}) and initial_layout_y ({len(initial_layout_y)}) tienen longitudes diferentes."
        print(error_message)
        results_file.write(error_message + "\n")
        return None, None, None, None, None, None, None, None, None

    best_sol = (list(initial_layout_x), list(initial_layout_y))
    aep_best = round(float(windFarmModel(best_sol[0], best_sol[1]).aep().sum()), 4)
    ranked_initial = [(fitness(best_sol), best_sol)]
    initial_time = 0.0

    results_file.write("\n--- Input Parameters ---\n")
    results_file.write(f"Number of Turbines: {WT_Num}\n")
    results_file.write(f"Reference AEP: {aep_ref} GWh\n")
    results_file.write(f"Iteration Length: {Iter_Length}\n")
    results_file.write(f"Number of Iterations per Generation: {Iter_Num}\n")
    results_file.write(f"Number of Generations: {Gen_Num}\n")
    results_file.write(f"Border Margin: {Border_Margin}\n")
    results_file.write(f"Number of Cores Used: {num_processes}\n\n")

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

    NewGen = best_sol
    for i in range(1, gen_num + 1):
        Gen.append(i)

        start_time = time.time()
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(single_solution_evaluation, [(NewGen[0], NewGen[1], iter_length, circular_border_params)] * iter_num)
        end_time = time.time()
        elapsed_time = end_time - start_time
        generation_times.append(elapsed_time)

        ranked = sorted([(fitness(sol), sol) for sol in results], key=lambda x: x[0])
        current_best_fitness, current_best_coords = ranked[-1]
        aep = round(float(windFarmModel(current_best_coords[0], current_best_coords[1]).aep().sum()), 4)

        if aep > aep_best:
            NewGen = current_best_coords
            aep_best = aep

        GenBest.append(NewGen)
        aep_best_series.append(aep_best)

        print(f"Gen {i}: AEP = {aep_best} GWh, Time = {elapsed_time:.2f} seconds (using {num_processes} cores)")
        results_file.write(f"Generation {i}: AEP = {aep_best} GWh, Time = {elapsed_time:.2f}\n")

    results_file.write("\n--- Final Layout ---\n")
    results_file.write(f"Final AEP: {aep_best} GWh\n")
    results_file.write("Final Layout Coordinates (x, y):\n")
    for x, y in zip(NewGen[0], NewGen[1]):
        results_file.write(f"({round(x, 2)}, {round(y, 2)})\n")
    results_file.write("\n")

    # --- Agregar verificación final de longitudes ---
    if len(NewGen[0]) != len(NewGen[1]):
        error_message = f"Error: Al finalizar getGenSolutions, NewGen[0] ({len(NewGen[0])}) y NewGen[1] ({len(NewGen[1])}) tienen longitudes diferentes."
        print(error_message)
        results_file.write(error_message + "\n")
        return None, None, None, None, None, None, None, None, None

    return NewGen, Gen, GenBest, aep_best_series, generation_times, aep_best, ranked_initial, initial_time, circular_border_params

def plot_results(wt_x_orig, wt_y_orig, final_sol, gens, gens_best, aep_list, circular_border_params_initial, title_suffix=""):
    output_dir = "optimization_results_comparison"
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
    center_x, center_y, radius = circular_border_params_initial
    circle = plt.Circle((center_x, center_y), radius, color='r', fill=False, linestyle='--', linewidth=0.5, label="Circular Border")
    plt.gca().add_patch(circle)
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
    circle = plt.Circle((center_x, center_y), radius, color='r', fill=False, linestyle='--', linewidth=0.5, label="Circular Border")
    plt.gca().add_patch(circle)
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
    output_dir = "optimization_results_comparison"
    os.makedirs(output_dir, exist_ok=True)
    results_filename = os.path.join(output_dir, "optimization_comparison_results.txt")
    with open(results_filename, "w") as f:
        f.write("Wind Farm Layout Optimization Results - Core Comparison\n\n")

        num_cores_list = [2, mp.cpu_count()]
        results_dict = {}

        for num_cores in num_cores_list:
            title_suffix = f"({num_cores} Cores)"
            f.write(f"\n--- Running with {num_cores} Cores ---\n")
            results_from_get_gen = getGenSolutions(
                wt_x, wt_y, Iter_Length, Iter_Num, Gen_Num, Border_Margin, num_cores, f  # Pass the file object
            )

            if results_from_get_gen is not None:
                final_sol, gens, gens_best, aep_list, generation_times, final_aep, ranked_initial, initial_time, circular_border_params_initial = results_from_get_gen
                aep_start = round(ranked_initial[0][0] / 1e6, 4)

                results_dict[num_cores] = {
                    'final_sol': final_sol,
                    'gens': gens,
                    'gens_best': gens_best,
                    'aep_list': aep_list,
                    'generation_times': generation_times,
                    'final_aep': final_aep,
                    'aep_start': aep_start,
                    'circular_border_params_initial': circular_border_params_initial
                }

                plot_results(wt_x, wt_y, final_sol, gens, gens_best, aep_list, circular_border_params_initial, title_suffix)
            else:
                print(f"Error: getGenSolutions devolvió None para {num_cores} núcleos.")

        print("\n--- Comparación de Tiempos de Ejecución ---")
        for num_cores, results in results_dict.items():
            if 'generation_times' in results:
                 total_time = sum(results['generation_times'])
                 print(f"Tiempo total de ejecución con {num_cores} núcleos: {total_time:.2f} segundos")
            else:
                print(f"No se encontraron tiempos de ejecución para {num_cores} núcleos.")
        
        print(f"Results saved to: {results_filename}")