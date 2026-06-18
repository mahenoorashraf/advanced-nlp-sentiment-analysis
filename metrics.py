import time
import psutil
from datetime import datetime


# ==========================================
# TRACK API LATENCY
# ==========================================

def track_latency(start_time):

    end_time = time.time()

    latency = end_time - start_time

    print(f"API Latency: {latency:.4f} seconds")

    return latency


# ==========================================
# CPU USAGE
# ==========================================

def cpu_usage():

    cpu = psutil.cpu_percent(interval=1)

    print(f"CPU Usage: {cpu}%")

    return cpu


# ==========================================
# MEMORY USAGE
# ==========================================

def memory_usage():

    memory = psutil.virtual_memory()

    used_memory = memory.percent

    print(f"Memory Usage: {used_memory}%")

    return used_memory


# ==========================================
# SYSTEM STATUS
# ==========================================

def system_status():

    print("\n===== SYSTEM STATUS =====")

    cpu_usage()

    memory_usage()

    print("=========================\n")


# ==========================================
# TEST METRICS
# ==========================================

if __name__ == "__main__":

    print("Starting monitoring metrics...\n")

    start = time.time()

    # Simulate processing delay
    time.sleep(2)

    track_latency(start)

    system_status()

    print("Metrics monitoring completed successfully")