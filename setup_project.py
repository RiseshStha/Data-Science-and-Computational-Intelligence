import os
import sys
import subprocess
import platform

def print_step(step, message):
    print(f"\n[{step}] {message}")

def check_command(command, name):
    try:
        subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        return True
    except subprocess.CalledProcessError:
        print(f"[ERROR] {name} is not installed or not in PATH.")
        return False

def install_pip_requirements(requirements_file):
    if os.path.exists(requirements_file):
        print(f"       Installing from {requirements_file}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
    else:
        print(f"       [WARNING] {requirements_file} not found.")

def main():
    print("===================================================")
    print("   Project Setup Script (Python Version)")
    print("===================================================")
    
    # 1. Check Prerequisites
    print_step("1/5", "Checking Prerequisites...")
    if not check_command("python --version", "Python"): return
    if not check_command("npm --version", "Node.js"): return
    
    # 2. Install Backend Dependencies
    print_step("2/5", "Installing Backend Dependencies...")
    install_pip_requirements(os.path.join("backend", "requirements.txt"))
    
    # 3. Install Frontend Dependencies
    print_step("3/5", "Installing Frontend Dependencies...")
    frontend_dir = os.path.join(os.getcwd(), "frontend")
    if os.path.exists(frontend_dir):
        subprocess.check_call("npm install", cwd=frontend_dir, shell=True)
    else:
        print("[ERROR] Frontend directory not found!")
        return

    # 4. Generate Machine Learning Models (Task 2)
    print_step("4/5", "Checking/Generating AI Models...")
    model_path = os.path.join("backend", "models", "kmeans_model.pkl")
    if not os.path.exists(model_path):
        print("       Running 'run_task2_clustering.py' to produce models...")
        # Run the existing clustering script
        try:
            subprocess.check_call([sys.executable, "run_task2_clustering.py"])
        except subprocess.CalledProcessError:
            print("[ERROR] Failed to run clustering script.")
            return
    else:
        print("       Models already exist. Skipping generation.")

    # 5. Database Setup
    print_step("5/5", "Setting up Database...")
    backend_dir = os.path.join(os.getcwd(), "backend")
    
    # Migrations
    subprocess.check_call([sys.executable, "manage.py", "makemigrations"], cwd=backend_dir)
    subprocess.check_call([sys.executable, "manage.py", "migrate"], cwd=backend_dir)
    
    # Populate Data
    populate_script = os.path.join(backend_dir, "populate_clustering_db.py")
    if os.path.exists(populate_script):
        print("       Populating database with clustered documents...")
        subprocess.check_call([sys.executable, populate_script])
    
    print("\n===================================================")
    print("   SETUP COMPLETE! ")
    print("===================================================")
    print("You can now run the project using:")
    print(f"   python run_project.py")
    print("===================================================")

if __name__ == "__main__":
    main()
