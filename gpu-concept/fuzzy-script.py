import subprocess
import sys

def run_fuzzy_test():
    number = 0
    while True:
        try:
            result = subprocess.run(
                ["./fuzzy-gpu", "-s", str(number)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            output = result.stdout
            if "assert" in output:
                print(f"Found 'assert' with -s {number}")
                sys.exit(0)
            else:
                print(f"No 'assert' found with -s {number}")
            number += 1
        except KeyboardInterrupt:
            print("Interrupted by user.")
            sys.exit(1)

if __name__ == "__main__":
    run_fuzzy_test()