import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

def plot_energy_history(histories: Dict[str, List[float]], title: str = "Energy Minimization Progress"):
    plt.figure(figsize=(10, 6))
    for label, history in histories.items():
        plt.plot(history, label=label, linewidth=2)
        # Highlight starting point
        plt.scatter(0, history[0], s=100, zorder=5) 
    plt.xlabel("Iterations (Temperature Steps)")
    plt.ylabel("Energy")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("visualization/energy_plot.png")
    plt.close()

def plot_benchmark_results(results: List[Dict]):
    solvers = [r['solver'] for r in results]
    runtimes = [r['runtime'] for r in results]
    energies = [r['final_energy'] for r in results]
    drops = [r['energy_drop'] for r in results]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1.bar(solvers, runtimes, color='skyblue')
    ax1.set_title("Runtime Comparison (s)")
    ax1.set_ylabel("Seconds")
    
    ax2.bar(solvers, energies, color='salmon')
    ax2.set_title("Final Energy (Lower is Better)")
    ax2.set_ylabel("Energy")
    
    ax3.bar(solvers, drops, color='lightgreen')
    ax3.set_title("Energy Improvement (Init -> Final)")
    ax3.set_ylabel("Energy Drop")
    
    plt.tight_layout()
    plt.savefig("visualization/benchmark_plot.png")
    plt.close()
