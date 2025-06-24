#!/usr/bin/env python3

import subprocess
import csv
import sys
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import math

def run_performance_test(num_matrices: int, matrix_size: int = 10, num_runs: int = 3) -> float:
    """
    Run the performance test GPU executable multiple times and return the average execution time.
    
    Args:
        num_matrices: Number of matrices to test
        matrix_size: Size of each matrix (default: 10)
        num_runs: Number of times to run the test for averaging (default: 3)
    
    Returns:
        Average execution time in milliseconds
    """
    execution_times = []
    
    for run in range(num_runs):
        try:
            # Run the performance test
            result = subprocess.run(
                ['./performance-test-gpu.release', '1', str(num_matrices), str(matrix_size)],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the output (should be just the time in milliseconds)
            execution_time = float(result.stdout.strip())
            execution_times.append(execution_time)
            
        except subprocess.CalledProcessError as e:
            print(f"Error running test for {num_matrices} matrices (run {run+1}): {e}")
            print(f"stderr: {e.stderr}")
            return -1.0
        except ValueError as e:
            print(f"Error parsing output for {num_matrices} matrices (run {run+1}): {e}")
            print(f"stdout: {result.stdout}")
            return -1.0
    
    # Return the average execution time
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        return avg_time
    else:
        return -1.0

def plot_performance_results(results: List[Tuple[int, float]]):
    """
    Create and save matplotlib plots of the performance results.
    
    Args:
        results: List of tuples containing (num_matrices, execution_time)
    """
    if not results:
        print("No results to plot")
        return
    
    # Extract data for plotting
    num_matrices = [r[0] for r in results]
    exec_times = [r[1] for r in results]
    
    # Calculate throughput (matrices per second)
    throughput = [(n / t) * 1000 if t > 0 else 0 for n, t in results]
    
    # Create subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
    
    # Plot 1: Execution Time vs Number of Matrices
    ax1.plot(num_matrices, exec_times, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Matrices')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('GPU Performance: POOL')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plot_filename = 'performance_results.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {plot_filename}")
    
    # Close the figure to free memory
    plt.close(fig)

def main():
    """Main function to run performance tests and export results."""
    
    # Check if the executable exists
    if not os.path.exists('./performance-test-gpu.release'):
        print("Error: performance-test-gpu.release not found in current directory")
        sys.exit(1)
    
    # Define test parameters
    matrix_size = 10
    # test_values = [int(pow(math.sqrt(10), i)) for i in range(1, 13)]
    test_values1 = range(10000, 1010001, 100000)
    test_values2 = range(100, 50000, 500)
    test_values = list(test_values1) + list(test_values2)

    
    print(f"Running performance tests with matrix size {matrix_size}x{matrix_size}")
    print("Testing with different numbers of matrices (3 runs each, averaged)...")
    
    # Store results
    results: List[Tuple[int, float]] = []
    
    # Run tests for different numbers of matrices
    for num_matrices in test_values:
        print(f"Testing with {num_matrices} matrices (3 runs)...", end=" ")
        
        execution_time = run_performance_test(num_matrices, matrix_size, num_runs=3)
        
        if execution_time >= 0:
            results.append((num_matrices, execution_time))
            print(f"Avg Time: {execution_time:.3f} ms")
        else:
            print("FAILED")
    
    # Export results to CSV
    csv_filename = 'performance_results.csv'
    
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Number of Matrices', 'Average Execution Time (ms)', 'Matrices per Second'])
            
            # Write data rows
            for num_matrices, exec_time in results:
                matrices_per_second = (num_matrices / exec_time) * 1000 if exec_time > 0 else 0
                writer.writerow([num_matrices, exec_time, matrices_per_second])
        
        print(f"\nResults exported to {csv_filename}")
        print(f"Total tests completed: {len(results)}")
        
        # Print summary
        if results:
            print("\nSummary (averaged over 3 runs):")
            print(f"{'Matrices':<10} {'Avg Time (ms)':<15} {'Matrices/sec':<15}")
            print("-" * 45)
            for num_matrices, exec_time in results:
                matrices_per_second = (num_matrices / exec_time) * 1000 if exec_time > 0 else 0
                print(f"{num_matrices:<10} {exec_time:<15.3f} {matrices_per_second:<15.2f}")
        
        # Generate and display plots
        print("\nGenerating performance plots...")
        plot_performance_results(results)
    
    except IOError as e:
        print(f"Error writing CSV file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
