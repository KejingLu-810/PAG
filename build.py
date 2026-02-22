#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
import multiprocessing
import argparse
from pathlib import Path

# ================= Configuration Area =================
ROOT_DIR = Path(__file__).parent.absolute()
BUILD_DIR = ROOT_DIR / "build"
BIN_DIR = ROOT_DIR / "bin"

# CMake target mapping relationship
TARGET_MAP = {
    "l2": "PAG_l2",
    "cos": "PAG_cos",
    "cosine": "PAG_cos",
    "tools": "bin2vec",
    "all": "all",
}

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

def log_info(msg):
    print(f"{Colors.GREEN}[INFO]{Colors.RESET} {msg}")

def log_warn(msg):
    print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {msg}")

def log_error(msg):
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {msg}")

def get_cpu_cores():
    try:
        return multiprocessing.cpu_count()
    except:
        return 4

def run_command(cmd, cwd=None, exit_on_error=True):
    try:
        subprocess.run(cmd, cwd=cwd, check=True, text=True)
    except subprocess.CalledProcessError:
        log_error(f"Command failed: {' '.join(cmd)}")
        if exit_on_error:
            sys.exit(1)
        return False
    return True

def clean_project():
    log_warn("Cleaning build intermediate files (build/)...")
    # Clean up BUILD_DIR while retaining BIN_DIR
    dirs_to_clean = [BUILD_DIR] 
    for d in dirs_to_clean:
        if d.exists():
            shutil.rmtree(d)
            print(f"  - Removed {d}/")
    
    # Clean up compile_commands.json to prevent residual old path information
    cc_json = ROOT_DIR / "compile_commands.json"
    if cc_json.exists():
        cc_json.unlink()
    log_info("Clean complete. (bin/ directory preserved)")

def configure_cmake(options=None):
    if not BUILD_DIR.exists():
        BUILD_DIR.mkdir()
    
    log_info("Configuring CMake...")
    
    cmd = ["cmake", "-DCMAKE_EXPORT_COMPILE_COMMANDS=1"]
    cmd.append("..")
    run_command(cmd, cwd=BUILD_DIR)

    src_json = BUILD_DIR / "compile_commands.json"
    dst_json = ROOT_DIR / "compile_commands.json"
    if src_json.exists():
        if dst_json.exists():
            dst_json.unlink()
        shutil.copy(src_json, dst_json)

def build_target(user_target, without_pes=False):
    cmake_target = TARGET_MAP.get(user_target)
    if not cmake_target and user_target != "all":
        log_warn(f"Unknown target: '{user_target}'. Skipping.")
        return

    jobs = str(32) # get_cpu_cores()
    log_info(f"Building target: {user_target.upper()} (Jobs: {jobs})...")
    
    cmd = ["make", "-j", jobs]
    if user_target != "all":
        cmd.append(cmake_target)
    
    run_command(cmd, cwd=BUILD_DIR)

def main():
    parser = argparse.ArgumentParser(description="Build script for PAG project.")
    parser.add_argument("actions", nargs='*', default=["all"], help="Actions: clean, all, l2, cos, tools")
    args = parser.parse_args()
    actions = args.actions

    if "clean" in actions:
        clean_project()
        if len(actions) == 1:
            sys.exit(0)
    
    configure_cmake()
    
    for action in actions:
        if action == "clean": continue
        if action in TARGET_MAP or action == "all":
            # Pass the ablation flag for subsequent renaming
            build_target(action)
        else:
            log_warn(f"Unrecognized action: {action}")

    if BIN_DIR.exists():
        log_info(f"Build finished. Executables in {BIN_DIR}/")

if __name__ == "__main__":
    main()