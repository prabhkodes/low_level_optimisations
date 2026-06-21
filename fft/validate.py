import numpy as np
import sys
import os

def verify_file(filename, numpy_fft, N):
    """
    Compares a specific output file against the NumPy reference.
    Returns: (Status String, Relative Error)
    """
    if not os.path.exists(filename):
        return "MISSING", 0.0

    try:
        # Load data: Column 0 is Real, Column 1 is Imaginary
        c_results = np.loadtxt(filename)
        
        # Check if the file has the correct number of rows
        if c_results.shape[0] != N:
             return f"SIZE MISMATCH", 0.0

        c_fft = c_results[:, 0] + 1j * c_results[:, 1]
    except Exception as e:
        return f"READ ERROR", 0.0
    
    # Compare against Ground Truth
    # Calculate differences
    diff = np.abs(c_fft - numpy_fft)
    max_diff = np.max(diff)
    max_magnitude = np.max(np.abs(numpy_fft))
    
    # Avoid division by zero if signal is empty/zero
    if max_magnitude == 0:
        rel_diff = 0.0 if max_diff == 0 else float('inf')
    else:
        rel_diff = max_diff / max_magnitude
        
    # Threshold for floating point comparisons (1e-5 is standard for single precision)
    threshold = 1e-5 
    
    if rel_diff < threshold:
        return "PASSED", rel_diff
    else:
        return "FAILED", rel_diff

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_all.py N")
        print("Example: python validate_all.py 1024")
        sys.exit(1)
        
    try:
        N = int(sys.argv[1])
    except ValueError:
        print("Error: N must be an integer")
        sys.exit(1)

    if N & (N - 1) != 0 or N <= 0:
        print("Error: N must be a power of 2")
        sys.exit(1)

    print(f"Generating Ground Truth using NumPy (N={N})...")
    
    # 1. Generate the exact same signal as the C code
    # signal = sin(2*pi*5*t) + 0.5*sin(2*pi*10*t)
    t = np.arange(N) / N
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    
    # 2. Compute Reference FFT
    numpy_fft = np.fft.fft(signal)
    
    # 3. List of files to verify
    files_to_check = [
        "fft_cpu.txt",
        "fft_gpu_v1.txt",
        "fft_gpu_v2.txt",
        "fft_gpu_v3.txt",
        "fft_acc_v1.txt",
        "fft_cufft.txt"
    ]

    print(f"\n{'FILENAME':<20} | {'STATUS':<15} | {'ERROR (Rel)':<15}")
    print("-" * 56)

    passed_count = 0
    total_checked = 0

    for fname in files_to_check:
        status, error = verify_file(fname, numpy_fft, N)
        
        print(f"{fname:<20} | {status:<15} | {error:.6e}")

        if status == "PASSED":
            passed_count += 1
        
        if status != "MISSING":
            total_checked += 1

    print("-" * 56)
    print(f"Summary: {passed_count}/{total_checked} files passed verification.")

if __name__ == "__main__":
    main()