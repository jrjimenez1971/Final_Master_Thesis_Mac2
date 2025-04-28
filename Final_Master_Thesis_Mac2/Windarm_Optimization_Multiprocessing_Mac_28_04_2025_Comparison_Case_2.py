#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 18:01:09 2025

@author: juanramonjimenezmogollon
"""

# Improved version of your wind farm layout optimization script using multiprocessing
# focusing on parameter adjustments to better utilize multi-core processors and comparing core counts

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
from matplotlib.patches import PathPatch  # Import PathPatch for plotting border


# Load site and turbine
site = Hornsrev1Site()
wt = V80()
windFarmModel = Bastankhah_PorteAgel_2014(site, wt, k=0.0324555)

wt_x, wt_y = site.initial_position.T.tolist()
WT_Num = len(wt_x)
aep_ref = round(float(windFarmModel(wt_x, wt_y).aep().sum()), 4)
WT_Rad = 40
Iter_Length = 10
Iter_Num = mp.cpu_count()  # Increase iterations to provide more work
Gen_Num = 2
Border_Margin = 1.05

# Define output directory
output_dir = "optimization_results_core_comparison"
os.makedirs(output_dir, exist_ok=True)

# Fitness function
PENALTY_THRESHOLD = 10 * WT_Rad
def fitness_numpy_optimized_penalty(s):
    x = np.array(s[0])
    y = np.array(s[1])
    distances = np.sqrt((x[:, np.newaxis] - x[np.newaxis, :])**2 + (y[:, np.newaxis] - y[np.newaxis, :])**2)
    distances = np.triu(distances, k=1)
    penalty = np.sum((distances[distances <= PENALTY_THRESHOLD] - PENALTY_THRESHOLD) * 100)
    aep = float(windFarmModel(s[0], s[1]).aep().sum()) * 1e6
    return aep + penalty

# Geometry helpers
def select_border_points(points):
    hull = ConvexHull(points)
    return points[hull.vertices]

def enlarge_polygon(points, factor):
    center = np.mean(points, axis=0)
    return np.array([center + (p - center) * factor for p in points])

def create_mesh_border(points):
    codes = [Path.MOVETO] + [Path.LINETO] * (len(points) - 1)
    return Path(points, codes)

def filter_points_inside_border(border_path, point):
    return border_path.contains_point((point[0], point[1]))

# Random layout modifier - Modified to work with NumPy arrays
def list_random_values_one_by_one_optimized_numpy(arr_x, arr_y, length, border):
    n = len(arr_x)
    arr_x_rnd, arr_y_rnd = arr_x.copy(), arr_y.copy()
    current_fitness = fitness_numpy_optimized_penalty((arr_x_rnd, arr_y_rnd))
    indices = list(range(n))
    random.shuffle(indices)

    for i in indices:
        bestx, besty = arr_x_rnd.copy(), arr_y_rnd.copy()
        best_f = current_fitness
        for _ in range(3):
            while True:
                newx = bestx[i] + length * np.random.uniform(-1, 1)
                newy = besty[i] + length * np.random.uniform(-1, 1)
                if filter_points_inside_border(border, (newx, newy)):
                    break
            trialx, trialy = bestx.copy(), besty.copy()
            trialx[i], trialy[i] = newx, newy
            trial_f = fitness_numpy_optimized_penalty((trialx, trialy))
            if trial_f > best_f:
                bestx, besty = trialx, trialy
                best_f = trial_f

        if best_f >= current_fitness:
            arr_x_rnd, arr_y_rnd = bestx.copy(), besty.copy()
            current_fitness = best_f

    return arr_x_rnd, arr_y_rnd

def single_solution_evaluation_numpy(args):
    lst1, lst2, length, border = args
    arr_x = np.array(lst1)
    arr_y = np.array(lst2)
    return list_random_values_one_by_one_optimized_numpy(arr_x, arr_y, length, border)

# Multiprocessing solution generator - Using NumPy-optimized functions
def getSolutions_numpy_parallel(num_processes):
    points = np.array(list(zip(wt_x, wt_y)))
    border_points = enlarge_polygon(select_border_points(points), Border_Margin)
    mesh_border = create_mesh_border(border_points)

    start_time = time.time()
    with mp.Pool(processes=num_processes) as pool:
        args_list = [(wt_x, wt_y, Iter_Length, mesh_border)] * Iter_Num
        results = pool.map(single_solution_evaluation_numpy, args_list)
    end_time = time.time()
    elapsed_time = end_time - start_time

    ranked = sorted([(fitness_numpy_optimized_penalty(sol), sol) for sol in results], key=lambda x: x[0])
    best_fitness, best_coords = ranked[-1]
    x_best, y_best = best_coords
    aep_best = round(float(windFarmModel(x_best, y_best).aep().sum()), 4)

    return (wt_x, wt_y) if aep_best < aep_ref else (x_best, y_best), max(aep_ref, aep_best), ranked, border_points, elapsed_time

# Genetic solution optimizer - Using NumPy-optimized functions with parallel execution
def getGenSolutions_numpy_parallel(initial, aep_best_prev, initial_border_points, results_file, num_processes):
    NewGen = initial
    Gen = []
    GenBest = []
    aep_best_series = []
    generation_times = []
    mesh_border = create_mesh_border(initial_border_points)

    for i in range(1, Gen_Num + 1):
        Gen.append(i)

        start_time = time.time()
        with mp.Pool(processes=num_processes) as pool:
            args_list = [(NewGen[0], NewGen[1], Iter_Length, mesh_border)] * Iter_Num
            results = pool.map(single_solution_evaluation_numpy, args_list)
        end_time = time.time()
        elapsed_time = end_time - start_time
        generation_times.append(elapsed_time)

        ranked = sorted([(fitness_numpy_optimized_penalty(sol), sol) for sol in results], key=lambda x: x[0])
        best_fitness, best_coords = ranked[-1]
        aep = round(float(windFarmModel(best_coords[0], best_coords[1]).aep().sum()), 4)

        if aep > aep_best_prev:
            NewGen = best_coords
            aep_best_prev = aep

        GenBest.append(NewGen)
        aep_best_series.append(aep_best_prev)

        print(f"Gen {i} ({num_processes} cores): AEP = {aep_best_prev} GWh, Time = {elapsed_time:.2f} seconds")
        results_file.write(f"Generation {i} ({num_processes} cores): AEP = {aep_best_prev} GWh, Time = {elapsed_time:.2f} seconds\n")

    return NewGen, Gen, GenBest, aep_best_series, generation_times

def plot_results(wt_x_orig, wt_y_orig, final_sol, gens, gens_best, aep_list, border_path_initial, filename_suffix=""):
    output_dir = "optimization_results_core_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1 - Original and Genetic Layouts
    plt.figure(figsize=(10, 8))
    plt.title(f'Original (blue) and Evolutions of Generation Layouts (yellow) {filename_suffix}')
    plt.plot(wt_x_orig, wt_y_orig, 'b.', label='Original Layout')
    for i, s in enumerate(gens_best):
        plt.plot(s[0], s[1], 'y.', label=f'Generation {gens[i]} Layout' if i == 0 else "")
    if final_sol is not None:
        plt.plot(final_sol[0], final_sol[1], 'g.', label='Final Layout')
    patch = PathPatch(border_path_initial, facecolor='none', edgecolor='r', linestyle='--', linewidth=0.5, label="Optimization Border")
    plt.gca().add_patch(patch)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"original_vs_genetic_layouts{filename_suffix.replace(' ', '_')}.png"))
    plt.close()
    print(f"Plot saved to: {os.path.join(output_dir, f'original_vs_genetic_layouts{filename_suffix.replace(' ', '_')}.png')}")

    # Plot 2 - Initial vs Final
    plt.figure(figsize=(10, 8))
    plt.title(f'Original (blue) and Final (black) Layouts {filename_suffix}')
    plt.plot(wt_x_orig, wt_y_orig, 'b.', label='Original Layout')
    if final_sol is not None:
        plt.plot(final_sol[0], final_sol[1], 'k.', label='Final Layout')
    patch = PathPatch(border_path_initial, facecolor='none', edgecolor='r', linestyle='--', linewidth=0.5, label="Optimization Border")
    plt.gca().add_patch(patch)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"initial_vs_final_layouts{filename_suffix.replace(' ', '_')}.png"))
    plt.close()
    print(f"Plot saved to: {os.path.join(output_dir, f'initial_vs_final_layouts{filename_suffix.replace(' ', '_')}.png')}")

    # Plot 3 - AEP evolution
    plt.figure(figsize=(10, 6))
    plt.title(f'Evolution of improvement through generations {filename_suffix}')
    x = gens
    y = aep_list
    if len(x) > 1:
        coef = np.polyfit(x, y, 2)
        parabola = np.poly1d(coef)
        x_trend = np.linspace(min(x), max(x), 100)
        y_trend = parabola(x_trend)
        plt.plot(x_trend, y_trend, 'r-', label='Parabolic Trend Line')
    plt.scatter(x, y, label='AEP Data Points')
    plt.xlabel('Generation')
    plt.ylabel('AEP (GWh)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"aep_evolution{filename_suffix.replace(' ', '_')}.png"))
    plt.close()
    print(f"Plot saved to: {os.path.join(output_dir, f'aep_evolution{filename_suffix.replace(' ', '_')}.png')}")

    # Plot 4 - Delta AEP evolution
    delta = np.diff([aep_ref] + aep_list)
    if len(gens) == len(delta):
        plt.figure(figsize=(10, 6))
        plt.title(f'Evolution of AEP improvements per generation {filename_suffix}')
        if len(gens) > 1:
            coef = np.polyfit(gens, delta, 2)
            parabola = np.poly1d(coef)
            x_trend = np.linspace(min(gens), max(gens), 100)
            y_trend = parabola(x_trend)
            plt.plot(x_trend, y_trend, 'r-', label='Parabolic Trend Line')
        plt.scatter(gens, delta, label='Delta AEP Data Points')
        plt.xlabel('Generation')
        plt.ylabel('Delta AEP (GWh)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"delta_aep_evolution{filename_suffix.replace(' ', '_')}.png"))
        plt.close()
        print(f"Plot saved to: {os.path.join(output_dir, f'delta_aep_evolution{filename_suffix.replace(' ', '_')}.png')}")
    else:
        print(f"Skipping Delta AEP plot {filename_suffix} due to data length mismatch.")

if __name__ == '__main__':
    results_filename = os.path.join(output_dir, "optimization_results_core_comparison.txt")
    num_cores_to_test = [1, mp.cpu_count() // 2, mp.cpu_count()]
    all_results = {}

    points = np.array(list(zip(wt_x, wt_y)))
    border_points_initial_array = enlarge_polygon(select_border_points(points), Border_Margin)
    mesh_border_initial = create_mesh_border(border_points_initial_array) # Crear el objeto Path

    with open(results_filename, "w") as f:
        f.write("Wind Farm Layout Optimization Results - Core Comparison\n\n")

        for num_cores in num_cores_to_test:
            f.write(f"\n--- Running with {num_cores} Cores ---\n")
            start_time_total = time.time()
            best_sol_initial, aep_start, ranked_initial, _, initial_time = getSolutions_numpy_parallel(num_cores)
            final_sol, gens, gens_best, aep_list, generation_times = getGenSolutions_numpy_parallel(
                best_sol_initial, aep_start, border_points_initial_array, f, num_cores # Usar el array aquí si es necesario en getGenSolutions
            )
            end_time_total = time.time()
            total_time = end_time_total - start_time_total
            final_aep = round(float(windFarmModel(final_sol[0], final_sol[1]).aep().sum()), 4)

            all_results[num_cores] = {
                'initial_aep': aep_start,
                'final_aep': final_aep,
                'generations': gens,
                'aep_evolution': aep_list,
                'generation_times': generation_times,
                'final_layout': final_sol,
                'gens_best_layouts': gens_best,
                'border': border_points_initial_array,
                'total_time': total_time
            }

            f.write("\n--- Results for {} Cores ---\n".format(num_cores))
            f.write(f"Number of Turbines: {WT_Num}\n")
            f.write(f"Reference AEP: {aep_ref} GWh\n")
            f.write(f"Initial Best AEP: {aep_start} GWh (Time: {initial_time:.2f} s)\n")
            f.write(f"Final AEP: {final_aep} GWh (Total Time: {total_time:.2f} s)\n")
            f.write("AEP Evolution per Generation:\n")
            for gen, aep in zip(gens, aep_list):
                f.write(f"Generation {gen}: {aep} GWh (Time: {generation_times[gen-1]:.2f} s)\n")
            f.write("Final Layout Coordinates (x, y):\n")
            for x, y in zip(final_sol[0], final_sol[1]):
                f.write(f"({round(x, 2)}, {round(y, 2)})\n")
            f.write("\n")

            plot_results(wt_x, wt_y, final_sol, gens, gens_best, aep_list, mesh_border_initial, f"({num_cores} Cores)") # Usar mesh_border_initial

        f.write("\n--- Comparison of Execution Times ---\n")
        for num_cores, results in all_results.items():
            f.write(f"Total execution time with {num_cores} cores: {results['total_time']:.2f} seconds\n")

    print(f"Results saved to: {results_filename}")
    print("Comparison plots generated in the 'optimization_results_core_comparison' directory.")