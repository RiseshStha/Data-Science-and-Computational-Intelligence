import os
import sys
import subprocess
import webbrowser
import threading
import time

def run_django():
    """Runs the Django backend server."""
    print("[Backend] Starting Django Server at http://127.0.0.1:8000...")
    # Assume venv is active or we are running in a venv environment
    # Use 'python' directly as we expect this script to be run via the desired python interpreter
    # OR we explicitly call the venv python if needed. 
    # For now, we assume the user runs `python run_project.py`
    
    cmd = [sys.executable, "manage.py", "runserver"]
    cwd = os.path.join(os.getcwd(), "backend")
    
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except KeyboardInterrupt:
        print("\n[Backend] Stopping Django Server...")

def run_react():
    """Runs the React frontend server."""
    print("[Frontend] Starting Vite Server at http://localhost:5173...")
    
    # npm run dev
    # Using shell=True for Windows 'npm' command resolution
    cwd = os.path.join(os.getcwd(), "frontend")
    
    try:
        subprocess.run("npm run dev", cwd=cwd, shell=True, check=True)
    except KeyboardInterrupt:
        print("\n[Frontend] Stopping Vite Server...")

def open_browser():
    """Opens the browser after a short delay."""
    time.sleep(5)
    print("\n[System] Opening Browser...")
    webbrowser.open("http://localhost:5173")

def main():
    print("===================================================")
    print("   Coventry University Research Search Engine")
    print("   Python Launcher")
    print("===================================================")
    
    # Check if we are in the right directory
    if not os.path.exists("backend") or not os.path.exists("frontend"):
        print("[ERROR] Please run this script from the project root (Research_Search_Engine).")
        return

    # Create threads
    t_backend = threading.Thread(target=run_django)
    t_frontend = threading.Thread(target=run_react)
    t_browser = threading.Thread(target=open_browser)
    
    # Start threads
    t_backend.start()
    t_frontend.start()
    t_browser.start()
    
    print("\n[System] Servers are starting... Press Ctrl+C to stop.")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[System] Shutting down...")
        # Threads are daemon=False by default, so they might persist unless killed.
        # However, subprocesses usually die with the shell.
        # In a real production script we'd handle signals better.
        os._exit(0)

if __name__ == "__main__":
    main()
