import os
import sys
import subprocess


def main():
    print("Building the package...")
    try:
        # Use 'flit build' command instead of 'python -m build'
        subprocess.check_call([sys.executable, "-m", "flit", "build"])
    except subprocess.CalledProcessError as error:
        print(error)
        print(f"An error occurred while building the package: {error}")
        sys.exit(1)
    else:
        print("Package built successfully.")

if __name__ == "__main__":
    main()