#!/usr/bin/env python3
import os
import subprocess
import time
import shutil
import urllib.request
import json
import signal
import threading
import sys

# =========================================================
# Configuration
# =========================================================
AI_ROOT = "/home/exposed/Desktop/hybridAI-nextG"
SIM_ROOT = "/home/exposed/Desktop/simulator-bridge"

AI_SCRIPT = "./src/core/deploy.sh"
SIM_SCRIPT = "./deploy.sh"
RUNS_PER_PHASE = 5

def keep_sudo_alive():
    """Background thread that prevents the sudo password timeout from expiring."""
    while True:
        subprocess.run(["sudo", "-n", "-v"], capture_output=True)
        time.sleep(60)

def boot_infrastructure():
    """Runs the deploy.sh script ONCE to boot the Docker containers and xApp."""
    print("\n" + "="*70)
    print("Booting Persistent Docker Infrastructure (RIC & xApp)...")
    print("="*70)
    
    sim_env = os.environ.copy()
    if "PROJECT_ROOT" in sim_env:
        del sim_env["PROJECT_ROOT"]
        
    cmd = f"{SIM_SCRIPT} --skip-ns3 -y"
    subprocess.run(cmd, shell=True, cwd=SIM_ROOT, env=sim_env)
    print("[+] Infrastructure boot complete. Waiting 10s for connections to stabilize...")
    time.sleep(10)

def clean_state(wipe_memory=False):
    """Deletes old telemetry and optionally wipes the LLM's memory."""
    # Ensure BOTH possible CSV names are cleared so runs don't contaminate each other
    for csv_file in ["kpms.csv", "kpms_baseline.csv"]:
        path = os.path.join(AI_ROOT, csv_file)
        if os.path.exists(path):
            os.remove(path)
            
    if wipe_memory:
        mem_path = os.path.join(AI_ROOT, "models/episodes.jsonl")
        if os.path.exists(mem_path):
            os.remove(mem_path)
            print("    [+] Wiped Reflexion memory (episodes.jsonl).")

def kill_relay_and_xapp():
    """Kills the Relay Server and the xApp process to force a fresh data pipeline."""
    print("    [+] Cleaning up old Relay Server and xApp process...")
    pid_file = os.path.join(SIM_ROOT, ".relay_server.pid")
    
    # 1. Kill relay
    if os.path.exists(pid_file):
        os.system(f"kill $(cat {pid_file}) 2>/dev/null || true")
        os.remove(pid_file)
    os.system("pkill -f ai_relay_server.py 2>/dev/null || true")
    
    # 2. Kill xApp process inside the container
    os.system("docker exec sample-xapp-24 pkill -f run_xapp.py 2>/dev/null || true")
    time.sleep(2)

def stop_ai(proc):
    """Gracefully kills the AI system and all its subprocesses."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=10)
    except Exception:
        pass # Process already dead
    time.sleep(3) # Give TCP ports a moment to unbind

def save_results(prefix, run_id):
    """Renames the output CSV to the phase-specific name."""
    # Handle the fact that Baseline mode outputs to a different filename!
    src_baseline = os.path.join(AI_ROOT, "kpms_baseline.csv")
    src_normal = os.path.join(AI_ROOT, "kpms.csv")
    
    src = src_baseline if prefix == "baseline" else src_normal
    dest = os.path.join(AI_ROOT, f"{prefix}{run_id}.csv")
    
    if os.path.exists(src):
        shutil.copy(src, dest)
        print(f"\n    [+] Saved results to {dest}")
    else:
        print(f"\n    [!] ERROR: {src} not found for {prefix}{run_id}!")

def inject_intent(text):
    """Fires an HTTP POST request to the Orchestrator's web UI."""
    print(f"\n    [+] Injecting manual intent: '{text}'")
    req = urllib.request.Request("http://127.0.0.1:8080/api/intent",
                                 data=json.dumps({"text": text}).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(req)
    except Exception as e:
        print(f"    [!] Failed to inject intent: {e}")

def run_phase(phase_name, ai_args, prefix, wipe_memory_first, energy_intent=False):
    """Executes a full 5-run phase."""
    print(f"\n{'='*70}\nStarting Phase: {phase_name}\n{'='*70}")
    
    sim_env = os.environ.copy()
    if "PROJECT_ROOT" in sim_env:
        del sim_env["PROJECT_ROOT"]
    
    for i in range(1, RUNS_PER_PHASE + 1):
        print(f"\n  --- {phase_name} | Run {i}/{RUNS_PER_PHASE} ---")
        
        # 1. Clean state
        clean_state(wipe_memory=(i == 1 and wipe_memory_first))

        # 2. Kill the old Relay Server AND the xApp to force a fresh connection!
        kill_relay_and_xapp()

        # 3. Start AI System FIRST (It hosts the TCP Server on port 6000)
        print(f"    [+] Starting AI System in the background (logs hidden in logs/ai_{prefix}{i}.log)")
        ai_log_path = os.path.join(AI_ROOT, f"logs/ai_{prefix}{i}.log")
        ai_log = open(ai_log_path, "w")
        ai_cmd = f"{AI_SCRIPT} {ai_args} --log-level INFO"
        ai_proc = subprocess.Popen(ai_cmd, shell=True, preexec_fn=os.setsid, cwd=AI_ROOT, stdout=ai_log, stderr=subprocess.STDOUT)
        
        # 4. Wait for AI to load model into VRAM and bind port 6000
        print("    [+] Waiting 35s for AI LLM to load into VRAM and open ports...")
        time.sleep(35)
        
        # 5. Start Simulator SECOND
        # Notice we removed --skip-xapp and --skip-relay. deploy.sh will bring them both up!
        print("    [+] Starting Simulator (ns-3 logs will stream below)...")
        sim_cmd = f"{SIM_SCRIPT} --skip-import --skip-ric --scenario eval_scenario-static-4UEs.cc --rngSeed {i} --injectAt 45 --injectType spike -y"
        sim_proc = subprocess.Popen(sim_cmd, shell=True, cwd=SIM_ROOT, env=sim_env)
        
        # 6. Inject Intent (if phase 4)
        if energy_intent:
            print("    [+] Waiting 20s for Simulator to stabilize before injecting Energy Intent...")
            time.sleep(20)
            inject_intent("Prepare the network for energy efficiency now.")
        
        # 7. Wait for Simulator to finish
        sim_proc.wait()
        
        # 8. Clean up and save
        stop_ai(ai_proc)
        ai_log.close()
        save_results(prefix, i)

def main():
    os.makedirs(os.path.join(AI_ROOT, "logs"), exist_ok=True)

    print("="*70)
    print("Automated Master's Thesis Experiments")
    print("="*70)
    
    print("Please enter your sudo password. The script will keep the session alive automatically.")
    subprocess.run(["sudo", "-v"], check=True)
    threading.Thread(target=keep_sudo_alive, daemon=True).start()
    
    boot_infrastructure()
    
    # Phase 1: Baseline (No AI actions, just passive logging)
    run_phase("Phase 1: Baseline", "--mode baseline", "baseline", wipe_memory_first=True)

    # Phase 2: AI Managed without Reflexion (Ablation)
    run_phase("Phase 2: AI (No Reflexion)", "--mode deploy --no-episodes", "no_reflexion", wipe_memory_first=True)

    # Phase 3: AI Managed WITH Reflexion (Full System)
    run_phase("Phase 3: AI (With Reflexion)", "--mode deploy", "reflexion", wipe_memory_first=True)

    # Phase 4: Energy Efficiency Steering
    run_phase("Phase 4: Energy Intent", "--mode deploy", "energy", wipe_memory_first=True, energy_intent=True)

    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE! You now have 20 CSV files.")
    print("Run `python evaluate_baseline.py --baseline ...` to generate your thesis graphs.")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Script aborted by user. Cleaning up...")
        os.system("pkill -f 'src/core/main.py'")
        sys.exit(1)