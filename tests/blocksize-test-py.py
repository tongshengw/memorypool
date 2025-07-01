#!/usr/bin/env python3

import subprocess
import csv

# Test parameters
num_matrices = 1000
matrix_size = 10
block_sizes = [2**i for i in range(1, 11)]  # 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024

results = []

print("Running block size tests...")
for block_size in block_sizes:
    print(f"Testing block size {block_size}...", end=" ")
    
    try:
        result = subprocess.run(
            ['./performance-test-gpu.release', '0', str(num_matrices), str(matrix_size), str(block_size)],
            capture_output=True,
            text=True,
            check=True
        )
        
        time_ms = float(result.stdout.strip())
        results.append([block_size, time_ms])
        print(f"{time_ms:.3f} ms")
        
    except Exception as e:
        print(f"FAILED: {e}")

# Export to CSV
with open('blocksize_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Block Size', 'Time (ms)'])
    writer.writerows(results)

print(f"\nResults exported to blocksize_results.csv")
print(f"Tests completed: {len(results)}")