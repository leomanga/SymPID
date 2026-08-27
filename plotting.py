import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sympy import symbols, lambdify

# Increase font size for better readability in documents (e.g., PDF)
plt.rcParams.update({
    'axes.titlesize': 18,
    'figure.titlesize': 25,
    'axes.labelsize': 16,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15
})

def plot_stability_region(conditions, N_poly, D_poly, plot_id, ki_range=(-10, 10), kd_range=(-10, 10),
                          resolution=200, savefile=True):
    """
    Plots the 2D stability region using the Hermite-Biehler method.
    
    Args:
        conditions (list): A list of stability condition strings.
        N_poly (Poly): The numerator polynomial of the plant.
        D_poly (Poly): The denominator polynomial of the plant.
        plot_id (int or str): Identifier for the saved plot filename.
        ki_range (tuple): Range for the integral gain ki.
        kd_range (tuple): Range for the derivative gain kd.
        resolution (int): Resolution of the meshgrid.
        savefile (bool): Whether to save the plot to disk.
    
    Returns:
        plt: The matplotlib pyplot instance.
    """
    if not conditions:
        return

    ki, kd = symbols("k_i k_d")

    # Create the grid for k_i and k_d
    ki_vals = np.linspace(ki_range[0], ki_range[1], resolution)
    kd_vals = np.linspace(kd_range[0], kd_range[1], resolution)
    KI, KD = np.meshgrid(ki_vals, kd_vals)

    stable_region = np.zeros_like(KI, dtype=bool)

    plt.figure(figsize=(10, 8))
    for conds_list in conditions:
        current_string_stable = np.ones_like(KI, dtype=bool)

        for cond in conds_list:
            cond_func = lambdify((ki, kd), cond, 'numpy')
            val = cond_func(KI, KD)
            if np.isscalar(val):
                val = np.full_like(KI, val, dtype=bool)
            current_string_stable &= (val > 0)

        stable_region |= current_string_stable

    plt.contourf(KI, KD, stable_region, levels=[0.5, 1.5], colors=['#2ca02c'], alpha=0.6)

    # Insert the polynomial expressions in the title for clarity
    plt.title(f'Method: Hermite-Biehler\nN(s) = {N_poly.as_expr()}\nD(s) = {D_poly.as_expr()}')
    plt.xlabel('$k_i$')
    plt.ylabel('$k_d$')

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    legend_elements = [Patch(facecolor='#2ca02c', alpha=0.6, label='Stable Region')]
    plt.legend(handles=legend_elements, loc='upper right')

    # Save the image to disk without pausing the program
    if savefile:
        filename = f"stability_PID_{plot_id}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()  # Crucial for cleaning up and freeing RAM

    return plt


def plot_hurwitz(conditions, N_poly, D_poly, plot_id, kp, ki, kd, kp_val=1, ki_range=(-10, 10),
                 kd_range=(-10, 10), resolution=200, other_mesh=None,
                 show_conditions=False, savefile=True):
    """
    Plots the 2D stability region using the Routh-Hurwitz method.
    
    Args:
        conditions (list): A list of Routh-Hurwitz conditions.
        N_poly (Poly): The numerator polynomial.
        D_poly (Poly): The denominator polynomial.
        plot_id (int or str): Identifier for the saved plot filename.
        kp (Symbol): Proportional gain symbol.
        ki (Symbol): Integral gain symbol.
        kd (Symbol): Derivative gain symbol.
        kp_val (float): Fixed value for proportional gain.
        ki_range (tuple): Range for the integral gain ki.
        kd_range (tuple): Range for the derivative gain kd.
        resolution (int): Resolution of the meshgrid.
        other_mesh (np.ndarray): Optional mesh to overlay.
        show_conditions (bool): Whether to plot individual conditions.
        savefile (bool): Whether to save the plot to disk.
        
    Returns:
        plt: The matplotlib pyplot instance.
    """
    if not conditions:
        return

    ki_vals = np.linspace(ki_range[0], ki_range[1], resolution)
    kd_vals = np.linspace(kd_range[0], kd_range[1], resolution)
    KI, KD = np.meshgrid(ki_vals, kd_vals)

    legend_elements = []
    legend_labels = []
    _, ax = plt.subplots(figsize=(10, 8))

    full_mask = np.ones_like(KI, dtype=bool)

    for i, c in enumerate(conditions):
        cond_func = lambdify((kp, ki, kd), c > 0, 'numpy')
        mask_single_cond = cond_func(kp_val, KI, KD)
        if np.isscalar(mask_single_cond):
            mask_single_cond = np.full_like(KI, mask_single_cond, dtype=bool)

        full_mask &= mask_single_cond

        if show_conditions:
            line_color = cm.hsv(i / len(conditions))
            ax.contour(KI, KD, mask_single_cond, levels=[0.5, 1.5],
                       colors=[line_color], linewidths=2)
            ax.contourf(KI, KD, mask_single_cond, levels=[0.5, 1.5],
                        colors=[line_color], alpha=0.1)

            legend_elements.append(Line2D([0], [0], color=line_color, lw=1))
            legend_labels.append(f"Condition {i}")

    if other_mesh is not None:
        ax.contourf(KI, KD, other_mesh, levels=[0.5, 1.5], colors=['black'], alpha=1)

    if show_conditions:
        ax.contourf(KI, KD, full_mask, levels=[0.5, 1.5], colors=['#1f77b4'], alpha=0.8)
        legend_elements.append(Patch(facecolor='#1f77b4', alpha=0.8, label='Stable Region'))
        legend_labels.append('Stable Region')
        ax.legend(handles=legend_elements, labels=legend_labels,
                  loc='upper center', ncol=5)
    else:
        ax.contourf(KI, KD, full_mask, levels=[0.5, 1.5], colors=['#1f77b4'], alpha=0.6)
        legend_elements = [Patch(facecolor='#1f77b4', alpha=0.6, label='Stable Region')]
        ax.legend(handles=legend_elements, loc='upper right')

    ax.set_xlabel('$k_i$')
    ax.set_ylabel('$k_d$')
    
    ax.set_title(f'Method: Routh\nN(s) = {N_poly.as_expr()}\nD(s) = {D_poly.as_expr()}')
        
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    print(f"Stability regions present in the plot: {full_mask.any()}")

    if savefile:
        filename = f"stability_PID_{plot_id}_hurwitz.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    return plt


def plot_hurwitz_subplots(conditions, N_poly, D_poly, plot_id, kp, ki, kd, kp_val=1, ki_range=(-10, 10),
                          kd_range=(-10, 10), resolution=200, savefile=True):
    """
    Plots the sub-conditions for the Routh-Hurwitz method in separate subplots.
    
    Args:
        conditions (list): A list of Routh-Hurwitz conditions.
        N_poly (Poly): The numerator polynomial.
        D_poly (Poly): The denominator polynomial.
        plot_id (int or str): Identifier for the saved plot filename.
        kp (Symbol): Proportional gain symbol.
        ki (Symbol): Integral gain symbol.
        kd (Symbol): Derivative gain symbol.
        kp_val (float): Fixed value for proportional gain.
        ki_range (tuple): Range for the integral gain ki.
        kd_range (tuple): Range for the derivative gain kd.
        resolution (int): Resolution of the meshgrid.
        savefile (bool): Whether to save the plot to disk.
        
    Returns:
        plt: The matplotlib pyplot instance.
    """
    if not conditions:
        return

    import math
    ki_vals = np.linspace(ki_range[0], ki_range[1], resolution)
    kd_vals = np.linspace(kd_range[0], kd_range[1], resolution)
    KI, KD = np.meshgrid(ki_vals, kd_vals)

    n_conds = len(conditions)
    cols = min(3, n_conds)
    rows = math.ceil(n_conds / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_conds == 1:
        axes = [axes]
    else:
        axes = np.atleast_1d(axes).flatten()

    for i, c in enumerate(conditions):
        cond_func = lambdify((kp, ki, kd), c > 0, 'numpy')
        mask_single_cond = cond_func(kp_val, KI, KD)
        if np.isscalar(mask_single_cond):
            mask_single_cond = np.full_like(KI, mask_single_cond, dtype=bool)

        ax = axes[i]
        ax.contourf(KI, KD, mask_single_cond, levels=[0.5, 1.5],
                    colors=['#1f77b4'], alpha=0.6)
        
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_xlabel('$k_i$')
        ax.set_ylabel('$k_d$')
        ax.set_title(f"Condition {i+1}")

    for j in range(n_conds, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'Method: Routh (Sub-conditions)\nN(s) = {N_poly.as_expr()}\nD(s) = {D_poly.as_expr()}')
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    if savefile:
        filename = f"stability_PID_{plot_id}_hurwitz_subplots.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    return plt


def plot_hurwitz_3d(N_poly, D_poly, plot_id, kp_range=(-10, 10), ki_range=(-10, 10),
                    kd_range=(-10, 10), resolution=50, elev=30, azim=45, savefile=True):
    """
    Plots the 3D stability region using the Routh-Hurwitz method.
    
    Args:
        N_poly (Poly): The numerator polynomial.
        D_poly (Poly): The denominator polynomial.
        plot_id (int or str): Identifier for the saved plot filename.
        kp_range (tuple): Range for the proportional gain kp.
        ki_range (tuple): Range for the integral gain ki.
        kd_range (tuple): Range for the derivative gain kd.
        resolution (int): Resolution of the 3D grid.
        elev (int): Elevation angle for the 3D plot viewpoint.
        azim (int): Azimuth angle for the 3D plot viewpoint.
        savefile (bool): Whether to save the plot to disk.
        
    Returns:
        plt or None: The matplotlib pyplot instance, or None if saved to file.
    """
    from pid_stabilizer import build_closed_loop_characteristic_poly
    from sympy import And

    hz = build_closed_loop_characteristic_poly(N_poly, D_poly, kp=None)
    conditions = hz[0].hurwitz_conditions()
    kp_sym, ki_sym, kd_sym = hz[1], hz[2], hz[3]

    if not conditions:
        return

    f_sym = And(*[c > 0 for c in conditions])
    f = lambdify((kp_sym, ki_sym, kd_sym), f_sym, 'numpy')

    kp_vals = np.linspace(kp_range[0], kp_range[1], resolution)
    ki_vals = np.linspace(ki_range[0], ki_range[1], resolution)
    kd_vals = np.linspace(kd_range[0], kd_range[1], resolution)
    
    KP, KI, KD = np.meshgrid(kp_vals, ki_vals, kd_vals, indexing='ij')

    mask = f(KP, KI, KD)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if np.any(mask):
        sc = ax.scatter(KP[mask], KI[mask], KD[mask], c=KP[mask], cmap='plasma', s=12, alpha=0.5)
        fig.colorbar(sc, ax=ax, label='$k_p$', pad=0.1)

    ax.set_title(f'3D Stability: Routh-Hurwitz\nN(s) = {N_poly.as_expr()}\nD(s) = {D_poly.as_expr()}')
    ax.set_xlabel('$k_p$')
    ax.set_ylabel('$k_i$')
    ax.set_zlabel('$k_d$')
    
    # Set the viewpoint
    ax.view_init(elev=elev, azim=azim)

    if savefile:
        plt.savefig(f"stability_PID_{plot_id}_hurwitz_3d.png", bbox_inches='tight')
        plt.close(fig)
        return None
    return plt


def plot_hermite_biehler_3d(N_poly, D_poly, plot_id, kp_range=(-10, 10), ki_range=(-10, 10),
                            kd_range=(-10, 10), resolution=50, elev=30, azim=45, savefile=True):
    """
    Plots the 3D stability region using the Hermite-Biehler method.
    
    Args:
        N_poly (Poly): The numerator polynomial.
        D_poly (Poly): The denominator polynomial.
        plot_id (int or str): Identifier for the saved plot filename.
        kp_range (tuple): Range for the proportional gain kp.
        ki_range (tuple): Range for the integral gain ki.
        kd_range (tuple): Range for the derivative gain kd.
        resolution (int): Resolution of the 3D grid.
        elev (int): Elevation angle for the 3D plot viewpoint.
        azim (int): Azimuth angle for the 3D plot viewpoint.
        savefile (bool): Whether to save the plot to disk.
        
    Returns:
        plt or None: The matplotlib pyplot instance, or None if saved to file.
    """
    from pid_stabilizer import calculate_pid_stability_conditions_backtracking
    from sympy import symbols, lambdify
    
    kp_vals = np.linspace(kp_range[0], kp_range[1], resolution)
    ki_vals = np.linspace(ki_range[0], ki_range[1], resolution)
    kd_vals = np.linspace(kd_range[0], kd_range[1], resolution)
    
    KI, KD = np.meshgrid(ki_vals, kd_vals)
    ki_sym, kd_sym = symbols("k_i k_d")
    
    valid_kp = []
    valid_ki = []
    valid_kd = []
    
    for kp_val in kp_vals:
        conds = calculate_pid_stability_conditions_backtracking(N_poly, D_poly, kp_val)
        if not conds:
            continue
            
        stable_region = np.zeros_like(KI, dtype=bool)
        for conds_list in conds:
            current_string_stable = np.ones_like(KI, dtype=bool)
            for cond in conds_list:
                cond_func = lambdify((ki_sym, kd_sym), cond, 'numpy')
                val = cond_func(KI, KD)
                if np.isscalar(val):
                    val = np.full_like(KI, val, dtype=bool)
                current_string_stable &= (val > 0)
            stable_region |= current_string_stable
            
        if np.any(stable_region):
            valid_ki.extend(KI[stable_region])
            valid_kd.extend(KD[stable_region])
            valid_kp.extend(np.full_like(KI[stable_region], kp_val))
            
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if valid_kp:
        sc = ax.scatter(valid_kp, valid_ki, valid_kd, c=valid_kp, cmap='plasma', s=12, alpha=0.5)
        fig.colorbar(sc, ax=ax, label='$k_p$', pad=0.1)

    ax.set_title(f'3D Stability: Hermite-Biehler\nN(s) = {N_poly.as_expr()}\nD(s) = {D_poly.as_expr()}')
    ax.set_xlabel('$k_p$')
    ax.set_ylabel('$k_i$')
    ax.set_zlabel('$k_d$')
    
    # Set the viewpoint
    ax.view_init(elev=elev, azim=azim)

    if savefile:
        plt.savefig(f"stability_PID_{plot_id}_hb_3d.png", bbox_inches='tight')
        plt.close(fig)
        return None
    return plt
