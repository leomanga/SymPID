import time
import signal
import pprint
from sympy import symbols, EXRAW, latex, Poly
from sympy.abc import s
from sympy.polys.specialpolys import random_poly

from pid_stabilizer import (
    calculate_pid_stability_conditions_itertools,
    calculate_pid_stability_conditions_backtracking,
    build_closed_loop_characteristic_poly
)
from plotting import (
    plot_stability_region,
    plot_hurwitz,
    plot_hurwitz_subplots,
    plot_hurwitz_3d,
    plot_hermite_biehler_3d
)


def benchmark_scaling_by_degree(min_deg=3, max_deg=10, runs_per_degree=3, fixed_N_deg=None, timeout_hurwitz=10):
    """Benchmarks the scaling of different stability calculation methods by polynomial degree."""
    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException()

    signal.signal(signal.SIGALRM, timeout_handler)

    s_sym = symbols("s")
    
    results = {
        "itertools": {},
        "backtracking": {},
        "hurwitz_exraw": {},
        "hurwitz_fraction_free": {}
    }
    
    hurwitz_exraw_active = True
    hurwitz_fraction_free_active = True
    
    print(f"--- START BENCHMARK SCALING (Degrees from {min_deg} to {max_deg}) ---")
    
    for d in range(min_deg, max_deg + 1):
        print(f"\n[Denominator Degree: {d}]")
        
        deg_N = fixed_N_deg if fixed_N_deg is not None else d - 1
        
        times_itertools = []
        times_backtracking = []
        times_hurwitz_exraw = []
        times_hurwitz_fraction_free = []
        
        for run in range(runs_per_degree):
            D = random_poly(s_sym, d, -30, 30, polys=True)
            N = random_poly(s_sym, deg_N, -30, 30, polys=True)
            
            # --- ITERTOOLS ---
            start = time.perf_counter()
            calculate_pid_stability_conditions_itertools(N, D, 1)
            end = time.perf_counter()
            times_itertools.append(end - start)
            
            # --- BACKTRACKING ---
            start = time.perf_counter()
            calculate_pid_stability_conditions_backtracking(N, D, 1)
            end = time.perf_counter()
            times_backtracking.append(end - start)
            
            # --- HURWITZ EXRAW ---
            if hurwitz_exraw_active:
                signal.alarm(timeout_hurwitz)
                try:
                    start = time.perf_counter()
                    build_closed_loop_characteristic_poly(N, D, 1)[0].set_domain(EXRAW).hurwitz_conditions()
                    end = time.perf_counter()
                    times_hurwitz_exraw.append(end - start)
                except TimeoutException:
                    print(f"  -> Hurwitz EXRAW timed out (> {timeout_hurwitz}s) at degree {d}. Disabled for subsequent tests.")
                    hurwitz_exraw_active = False
                finally:
                    signal.alarm(0)

            # --- HURWITZ FRACTION FREE ---
            if hurwitz_fraction_free_active:
                signal.alarm(timeout_hurwitz)
                try:
                    start = time.perf_counter()
                    build_closed_loop_characteristic_poly(N, D, 1)[0].hurwitz_conditions()
                    end = time.perf_counter()
                    times_hurwitz_fraction_free.append(end - start)
                except TimeoutException:
                    print(f"  -> Hurwitz Fraction-Free timed out (> {timeout_hurwitz}s) at degree {d}. Disabled for subsequent tests.")
                    hurwitz_fraction_free_active = False
                finally:
                    signal.alarm(0)
                    
        # Average calculation
        avg_iter = sum(times_itertools) / len(times_itertools)
        avg_back = sum(times_backtracking) / len(times_backtracking)
        
        results["itertools"][d] = avg_iter
        results["backtracking"][d] = avg_back
        
        print(f"  Itertools:             {avg_iter:.4f}s")
        print(f"  Backtracking:          {avg_back:.4f}s")
        
        if hurwitz_exraw_active and times_hurwitz_exraw:
            avg_hurw_exraw = sum(times_hurwitz_exraw) / len(times_hurwitz_exraw)
            results["hurwitz_exraw"][d] = avg_hurw_exraw
            print(f"  Hurwitz EXRAW:         {avg_hurw_exraw:.4f}s")
        else:
            results["hurwitz_exraw"][d] = None
            if not hurwitz_exraw_active:
                print("  Hurwitz EXRAW:         Timeout / Disabled")

        if hurwitz_fraction_free_active and times_hurwitz_fraction_free:
            avg_hurw_ff = sum(times_hurwitz_fraction_free) / len(times_hurwitz_fraction_free)
            results["hurwitz_fraction_free"][d] = avg_hurw_ff
            print(f"  Hurwitz Fraction-Free: {avg_hurw_ff:.4f}s")
        else:
            results["hurwitz_fraction_free"][d] = None
            if not hurwitz_fraction_free_active:
                print("  Hurwitz Fraction-Free: Timeout / Disabled")
                
    print("\n--- FINAL RESULTS (Dictionary) ---")
    pprint.pprint(results)
    
    return results


if __name__ == "__main__":
    # Example usage / Plots showcase
    """
    D = Poly(s**5 + 8*s**4 + 32*s**3 + 46*s**2 + 46*s + 17)
    N = Poly(s**3 - 4*s**2 + s + 2)
    c = calculate_pid_stability_conditions_backtracking(N, D, 1)

    hz = build_closed_loop_characteristic_poly(N, D, 1)
    c2 = hz[0].hurwitz_conditions()

    for _1 in c:
        print("[")
        for _2 in _1:
            print(latex(_2))
        print("]")

    print(latex(c2))

    kp_range = (-10, 10)
    ki_range = (-1, 8)
    kd_range = (-8, 6)

    # 2D Plots (with Kp = 1)
    plot_stability_region(c, N, D, 1, savefile=False, resolution=500, ki_range=ki_range, kd_range=kd_range)

    kp, ki, kd = symbols("k_p k_i k_d")
    plt = plot_hurwitz(c2, N, D, 1, kp, hz[2], hz[3], hz[0], resolution=500, savefile=False, ki_range=ki_range, kd_range=kd_range)
    plt.show()

    plt2 = plot_hurwitz_subplots(c2, N, D, 1, kp, hz[2], hz[3], hz[0], resolution=500, savefile=False, ki_range=ki_range, kd_range=kd_range)
    plt2.show()

    # 3D Plots
    plt3 = plot_hurwitz_3d(N, D, 1, kp_range=kp_range, ki_range=ki_range, kd_range=kd_range, resolution=50, savefile=False)
    if plt3 is not None:
        plt3.show()

    plt4 = plot_hermite_biehler_3d(N, D, 1, kp_range=kp_range, ki_range=ki_range, kd_range=kd_range, resolution=50, savefile=False)
    if plt4 is not None:
        plt4.show()
    """
    
    # Run the benchmark as an example
    benchmark_scaling_by_degree(min_deg=3, max_deg=55, runs_per_degree=10, timeout_hurwitz=60, fixed_N_deg=None)
