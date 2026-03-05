import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import corner

__all__ = [
    "plot_corner_probability",
    "plot_fisher_heatmap",
    "plot_correlation_matrix",
    "plot_uncertainty_ellipse",
    "plot_eigenvalue_spectrum",
    "plot_standard_errors",
    "plot_condition_number_breakdown",
    "create_fisher_dashboard",
    "local_sensitivity_analysis",
    "plot_sensitivity_heatmap",
]


def plot_corner_probability(tt, yy, highlight_pts=None, cmap="Blues_r", array1=None, array2=None, inlabels=None, cb_label=r'$-\ln P$'):
    """Corner plot of parameter samples coloured by a scalar (e.g. log-probability).

    Parameters
    ----------
    tt : np.ndarray, shape (n_samples, n_params)
        Parameter samples.
    yy : np.ndarray, shape (n_samples,)
        Scalar values used to colour the scatter points (e.g. ``-ln P``).
    highlight_pts : np.ndarray, optional
        Additional points to overplot in red.
    cmap : str, optional
        Matplotlib colormap name.  Defaults to ``"Blues_r"``.
    array1 : array-like, optional
        A single parameter vector to mark with vertical/horizontal red lines.
    array2 : array-like, optional
        A second parameter vector to mark in green.
    inlabels : list of str, optional
        Axis labels for each parameter.
    cb_label : str, optional
        Colorbar label.  Defaults to ``r'$-\\ln P$'``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ndim = tt.shape[1]
    fig = corner.corner(tt, c=yy, plot_datapoints=False, plot_density=False, plot_contours=False, 
                    labels=inlabels, label_kwargs={"fontsize":16, "rotation": 45, "ha": "right"})
    axes = np.array(fig.axes).reshape((ndim, ndim))
    cb_rng = [yy.min(), yy.max()]

    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            # im = ax.scatter(tt.T[xi], tt.T[yi], c=yy, s=5, cmap='coolwarm', norm=colors.LogNorm(vmin=min(cb_rng), vmax=max(cb_rng)), alpha=1.0)
            im = ax.scatter(tt.T[xi], tt.T[yi], c=yy, s=5, cmap=cmap, norm=colors.Normalize(vmin=min(cb_rng), vmax=max(cb_rng)), alpha=1.0)
            if highlight_pts is not None:
                ax.scatter(highlight_pts.T[xi], highlight_pts.T[yi], s=20, facecolor="r", edgecolor="none")
        
    # remove histograms on diagonal
    for i in range(ndim):
        ax = axes[i, i]
        fig.delaxes(ax)

    if array1 is not None: 
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(array1[xi], color="r")
                ax.axhline(array1[yi], color="r")
                ax.plot(array1[xi], array1[yi], "sr")
                
    if array2 is not None: 
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(array2[xi], color="g")
                ax.axhline(array2[yi], color="g")
                ax.plot(array2[xi], array2[yi], "sg")

    cb = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', anchor=(0,1), 
                        shrink=.5, pad=-0.1)
    cb.set_label(cb_label, fontsize=25, labelpad=-120)
    cb_ticks = np.linspace(cb_rng[0], cb_rng[1], 10)
    cb.set_ticks(cb_ticks)
    cb.set_ticklabels([f'{val:.1f}' for val in cb_ticks])
    cb.ax.tick_params(labelsize=18)
    
    return fig


def plot_fisher_heatmap(fisher_info, param_names=None, ax=None):
    """
    Plot Fisher Information Matrix as a heatmap.
    
    Parameters
    ----------
    fisher_info : np.ndarray
        Fisher Information Matrix
    param_names : list, optional
        Parameter names for axis labels
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    n_params = fisher_info.shape[0]
    if param_names is None:
        param_names = [f'$\\theta_{{{i+1}}}$' for i in range(n_params)]
    
    # Plot heatmap
    im = ax.imshow(fisher_info, cmap='YlOrRd', aspect='auto')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Fisher Information')
    
    # Set ticks and labels
    ax.set_xticks(range(n_params))
    ax.set_yticks(range(n_params))
    ax.set_xticklabels(param_names)
    ax.set_yticklabels(param_names)
    
    # Add values as text
    for i in range(n_params):
        for j in range(n_params):
            text = ax.text(j, i, f'{fisher_info[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Fisher Information Matrix', fontsize=20, fontweight='bold')
    ax.set_xlabel('Parameters', fontsize=18)
    ax.set_ylabel('Parameters', fontsize=18)
    
    return ax


def plot_correlation_matrix(fisher_info, param_names=None, ax=None):
    """
    Plot parameter correlation matrix derived from Fisher Information.
    
    Parameters
    ----------
    fisher_info : np.ndarray
        Fisher Information Matrix
    param_names : list, optional
        Parameter names for axis labels
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    n_params = fisher_info.shape[0]
    if param_names is None:
        param_names = [f'$\\theta_{{{i+1}}}$' for i in range(n_params)]
    
    # Compute correlation matrix
    cov = np.linalg.inv(fisher_info)
    std_devs = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_devs, std_devs)
    
    # Plot heatmap
    im = ax.imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Correlation')
    
    # Set ticks and labels
    ax.set_xticks(range(n_params))
    ax.set_yticks(range(n_params))
    ax.set_xticklabels(param_names)
    ax.set_yticklabels(param_names)
    
    # Add values as text
    for i in range(n_params):
        for j in range(n_params):
            color = "white" if abs(corr[i, j]) > 0.5 else "black"
            text = ax.text(j, i, f'{corr[i, j]:.2f}',
                          ha="center", va="center", color=color, fontsize=9)
    
    ax.set_title('Parameter Correlation Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Parameters')
    
    return ax


def plot_uncertainty_ellipse(fisher_info, center=None, param_names=None, 
                            n_std=2, ax=None, indices=(0, 1)):
    """
    Plot 2D uncertainty ellipse from Fisher Information.
    
    Parameters
    ----------
    fisher_info : np.ndarray
        Fisher Information Matrix
    center : np.ndarray, optional
        Center point for ellipse (parameter values)
    param_names : list, optional
        Parameter names for axis labels
    n_std : float, default=2
        Number of standard deviations for ellipse
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    indices : tuple, default=(0, 1)
        Which two parameters to plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Extract 2x2 submatrix
    i, j = indices
    FI_2d = fisher_info[np.ix_([i, j], [i, j])]
    
    # Get covariance matrix
    cov_2d = np.linalg.inv(FI_2d)
    
    if center is None:
        center = np.zeros(2)
    else:
        center = np.array([center[i], center[j]])
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
    
    # Compute ellipse parameters
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])
    
    # Create ellipse
    ellipse = Ellipse(center, width, height, angle=angle,
                     facecolor='lightblue', edgecolor='blue', 
                     linewidth=2, alpha=0.5, label=f'{n_std}$\\sigma$ confidence')
    ax.add_patch(ellipse)
    
    # Plot center
    ax.plot(center[0], center[1], 'ro', markersize=8, label='Parameter estimate')
    
    # Plot principal axes
    for k in range(2):
        # Direction of kth eigenvector
        direction = eigenvectors[:, k]
        length = n_std * np.sqrt(eigenvalues[k])
        
        ax.arrow(center[0], center[1], 
                length * direction[0], length * direction[1],
                head_width=0.05*length, head_length=0.1*length,
                fc='red', ec='red', linewidth=2, alpha=0.7)
        ax.arrow(center[0], center[1], 
                -length * direction[0], -length * direction[1],
                head_width=0.05*length, head_length=0.1*length,
                fc='red', ec='red', linewidth=2, alpha=0.7)
    
    # Set axis labels
    if param_names is None:
        param_names = [f'$\\theta_{{{k+1}}}$' for k in range(fisher_info.shape[0])]
    
    ax.set_xlabel(param_names[i], fontsize=12)
    ax.set_ylabel(param_names[j], fontsize=12)
    ax.set_title(f'Uncertainty Ellipse ({n_std}$\\sigma$)', fontsize=14, fontweight='bold')
    
    # Make aspect ratio equal
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set reasonable axis limits
    margin = max(width, height) * 0.3
    ax.set_xlim(center[0] - width/2 - margin, center[0] + width/2 + margin)
    ax.set_ylim(center[1] - height/2 - margin, center[1] + height/2 + margin)
    
    return ax


def plot_eigenvalue_spectrum(fisher_info, param_names=None, ax=None):
    """
    Plot eigenvalue spectrum of Fisher Information Matrix.
    Shows which parameter combinations are well/poorly constrained.
    
    Parameters
    ----------
    fisher_info : np.ndarray
        Fisher Information Matrix
    param_names : list, optional
        Parameter names
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    n_params = fisher_info.shape[0]
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(fisher_info)
    
    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Plot eigenvalues
    x = np.arange(1, n_params + 1)
    bars = ax.bar(x, eigenvalues, color='steelblue', edgecolor='black', linewidth=1.5)
    
    # Color code: red for small eigenvalues (poorly constrained)
    threshold = 0.1 * eigenvalues.max()
    for i, (bar, val) in enumerate(zip(bars, eigenvalues)):
        if val < threshold:
            bar.set_color('salmon')
    
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Eigenvalue (Information)', fontsize=12)
    ax.set_title('Fisher Information Eigenvalue Spectrum', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add log scale option if eigenvalues span many orders of magnitude
    if eigenvalues.max() / eigenvalues.min() > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Eigenvalue (Information) [log scale]', fontsize=12)
    
    # Add legend
    high_info = mpatches.Patch(color='steelblue', label='Well constrained')
    low_info = mpatches.Patch(color='salmon', label='Poorly constrained')
    ax.legend(handles=[high_info, low_info])
    
    # Print eigenvector information
    print("\nPrincipal Components:")
    print("-" * 60)
    if param_names is None:
        param_names = [f'$\\theta_{{{i+1}}}$' for i in range(n_params)]
    
    for i in range(n_params):
        print(f"PC{i+1} (λ={eigenvalues[i]:.2e}):")
        components = " + ".join([f"{eigenvectors[j, i]:+.2f}·{param_names[j]}" 
                                for j in range(n_params)])
        print(f"  {components}")
        if eigenvalues[i] < threshold:
            print(f"  ⚠️  WARNING: Poorly constrained direction!")
        print()
    
    return ax


def plot_standard_errors(fisher_info, param_names=None, ax=None):
    """
    Plot parameter standard errors from Fisher Information.
    
    Parameters
    ----------
    fisher_info : np.ndarray
        Fisher Information Matrix
    param_names : list, optional
        Parameter names
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    n_params = fisher_info.shape[0]
    if param_names is None:
        param_names = [f'$\\theta_{{{i+1}}}$' for i in range(n_params)]
    
    # Compute standard errors
    cov = np.linalg.inv(fisher_info)
    std_errors = np.sqrt(np.diag(cov))
    
    # Plot bars
    x = np.arange(n_params)
    bars = ax.bar(x, std_errors, color='coral', edgecolor='black', 
                  linewidth=1.5, alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, se) in enumerate(zip(bars, std_errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{se:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Parameters', fontsize=12)
    ax.set_ylabel('Standard Error', fontsize=12)
    ax.set_title('Parameter Uncertainty (Standard Errors)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(param_names)
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_condition_number_breakdown(fisher_info, param_names=None, ax=None):
    """
    Visualize condition number and its implications.
    
    Parameters
    ----------
    fisher_info : np.ndarray
        Fisher Information Matrix
    param_names : list, optional
        Parameter names
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(fisher_info)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    condition_number = eigenvalues[0] / eigenvalues[-1]
    
    # Create visualization
    n = len(eigenvalues)
    x = np.arange(n)
    
    ax.semilogy(x, eigenvalues, 'o-', linewidth=2, markersize=8, 
                color='steelblue', label='Eigenvalues')
    
    # Highlight max and min
    ax.semilogy(0, eigenvalues[0], 'go', markersize=12, 
                label=f'Max: {eigenvalues[0]:.2e}')
    ax.semilogy(n-1, eigenvalues[-1], 'ro', markersize=12, 
                label=f'Min: {eigenvalues[-1]:.2e}')
    
    # Add condition number annotation
    ax.axhline(eigenvalues[0], color='green', linestyle='--', alpha=0.3)
    ax.axhline(eigenvalues[-1], color='red', linestyle='--', alpha=0.3)
    
    # Annotate condition number
    mid_y = np.sqrt(eigenvalues[0] * eigenvalues[-1])
    ax.annotate(f'$\\kappa$ = {condition_number:.1f}', 
                xy=(n/2, mid_y), fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Add interpretation text
    if condition_number < 10:
        status = "Excellent"
        color = 'green'
    elif condition_number < 100:
        status = "Good"
        color = 'orange'
    elif condition_number < 1000:
        status = "Fair"
        color = 'darkorange'
    else:
        status = "Poor"
        color = 'red'
    
    ax.text(0.02, 0.98, f'Condition: {status}', 
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', color=color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Eigenvalue Index (sorted)', fontsize=12)
    ax.set_ylabel('Eigenvalue (log scale)', fontsize=12)
    ax.set_title('Condition Number Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def create_fisher_dashboard(fisher_info, param_names=None, center=None):
    """
    Create comprehensive dashboard with multiple Fisher Information visualizations.
    
    Parameters
    ----------
    fisher_info : np.ndarray
        Fisher Information Matrix
    param_names : list, optional
        Parameter names
    center : np.ndarray, optional
        Center point for ellipse plot
    """
    n_params = fisher_info.shape[0]
    
    if n_params == 2:
        fig = plt.figure(figsize=(16, 10))
        
        # 2x3 layout for 2D case
        ax1 = plt.subplot(2, 3, 1)
        plot_fisher_heatmap(fisher_info, param_names, ax1)
        
        ax2 = plt.subplot(2, 3, 2)
        plot_correlation_matrix(fisher_info, param_names, ax2)
        
        ax3 = plt.subplot(2, 3, 3)
        plot_standard_errors(fisher_info, param_names, ax3)
        
        ax4 = plt.subplot(2, 3, 4)
        plot_uncertainty_ellipse(fisher_info, center, param_names, n_std=2, ax=ax4)
        
        ax5 = plt.subplot(2, 3, 5)
        plot_eigenvalue_spectrum(fisher_info, param_names, ax5)
        
        ax6 = plt.subplot(2, 3, 6)
        plot_condition_number_breakdown(fisher_info, param_names, ax6)
        
    else:
        fig = plt.figure(figsize=(16, 12))
        
        # 2x3 layout for higher dimensions
        ax1 = plt.subplot(2, 3, 1)
        plot_fisher_heatmap(fisher_info, param_names, ax1)
        
        ax2 = plt.subplot(2, 3, 2)
        plot_correlation_matrix(fisher_info, param_names, ax2)
        
        ax3 = plt.subplot(2, 3, 3)
        plot_standard_errors(fisher_info, param_names, ax3)
        
        ax4 = plt.subplot(2, 3, 4)
        plot_eigenvalue_spectrum(fisher_info, param_names, ax4)
        
        ax5 = plt.subplot(2, 3, 5)
        plot_condition_number_breakdown(fisher_info, param_names, ax5)
        
        # For >2D, show 2D projection of first two parameters
        if n_params > 2:
            ax6 = plt.subplot(2, 3, 6)
            plot_uncertainty_ellipse(fisher_info, center, param_names, 
                                    n_std=2, ax=ax6, indices=(0, 1))
    
    plt.tight_layout()
    return fig


# ========================================================
# Local Sensitivity Analysis
# ========================================================

def local_sensitivity_analysis(vpm_model, theta_fiducial, inparams, outparams, delta_percent=1.0):
    """Perform local (gradient-based) sensitivity analysis around a fiducial point.

    Perturbs each input parameter individually by ±``delta_percent`` percent
    of its fiducial value and records the resulting percent change in every
    output parameter.

    Parameters
    ----------
    vpm_model : VplanetModel
        Initialised VPLanet model instance.
    theta_fiducial : np.ndarray, shape (n_params,)
        Fiducial (reference) parameter values in the units defined by
        ``vpm_model.inparams``.
    inparams : dict
        Input parameter dict (``{"body.dParam": unit, ...}``); used to
        extract the ordered list of parameter names.
    outparams : dict
        Output parameter dict; used to extract the ordered list of output
        names.
    delta_percent : float, optional
        Fractional perturbation size in percent.  Defaults to ``1.0``
        (i.e. ±1 %).

    Returns
    -------
    sensitivity_matrix : np.ndarray, shape (n_outputs, n_inputs, 2)
        Percent change in each output for each input perturbation.
        ``sensitivity_matrix[j, i, 0]`` is the response to
        ``+delta_percent`` applied to input ``i`` on output ``j``;
        index ``1`` corresponds to ``-delta_percent``.
    param_names : list of str
        Input parameter names (keys of ``inparams``).
    output_names : list of str
        Output parameter names (keys of ``outparams``).
    """
    
    param_names = list(inparams.keys())
    output_names = list(outparams.keys())
    n_params = len(param_names)
    n_outputs = len(output_names)
    
    # Get baseline outputs
    print("Computing baseline outputs...")
    baseline_outputs = vpm_model.run_model(theta_fiducial, remove=True)
    
    # Initialize sensitivity matrix [n_outputs x n_inputs x 2]
    sensitivity_matrix = np.zeros((n_outputs, n_params, 2))
    
    print(f"Performing sensitivity analysis for {n_params} parameters...")
    
    for i, param_name in enumerate(param_names):
        print(f"  Parameter {i+1}/{n_params}: {param_name}")
        
        # Create perturbed parameter sets
        theta_plus = theta_fiducial.copy()
        theta_minus = theta_fiducial.copy()
        
        # Apply ±delta_percent change
        delta = delta_percent / 100.0
        theta_plus[i] = theta_fiducial[i] * (1 + delta)
        theta_minus[i] = theta_fiducial[i] * (1 - delta)
        
        # Run models with perturbed parameters
        try:
            outputs_plus = vpm_model.run_model(theta_plus, remove=True)
            outputs_minus = vpm_model.run_model(theta_minus, remove=True)
            
            # Calculate percent changes for each output
            for j in range(n_outputs):
                if baseline_outputs[j] != 0:  # Avoid division by zero
                    pct_change_plus = ((outputs_plus[j] - baseline_outputs[j]) / baseline_outputs[j]) * 100
                    pct_change_minus = ((outputs_minus[j] - baseline_outputs[j]) / baseline_outputs[j]) * 100
                    
                    sensitivity_matrix[j, i, 0] = pct_change_plus
                    sensitivity_matrix[j, i, 1] = pct_change_minus
                else:
                    sensitivity_matrix[j, i, :] = np.nan
                    
        except Exception as e:
            print(f"    Error with parameter {param_name}: {e}")
            
            sensitivity_matrix[:, i, :] = np.nan
    
    return sensitivity_matrix, param_names, output_names


# ========================================================
# Plot Sensitivity Analysis Results
# ========================================================

def plot_sensitivity_heatmap(sensitivity_matrix, param_names, output_names, delta_percent=1.0, label_fs=16):
    """Plot local sensitivity analysis results as a four-panel heatmap figure.

    Produces a 2×2 grid showing: (1) average absolute sensitivity,
    (2) response to +delta perturbation, (3) response to −delta
    perturbation, and (4) a bar chart of the most influential input for
    each output.

    Parameters
    ----------
    sensitivity_matrix : np.ndarray, shape (n_outputs, n_inputs, 2)
        Output of :func:`local_sensitivity_analysis`.
    param_names : list of str
        Input parameter names (used for x-axis tick labels; the prefix
        ``"earth.d"`` is stripped automatically).
    output_names : list of str
        Output parameter names (used for y-axis tick labels; the prefix
        ``"final.earth."`` is stripped automatically).
    delta_percent : float, optional
        Perturbation size used in the analysis — shown in panel titles.
        Defaults to ``1.0``.
    label_fs : int, optional
        Base font size for axis labels and titles.  Defaults to ``16``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    
    # Clean parameter and output names for display
    param_labels = [name.replace('earth.d', '') for name in param_names]
    output_labels = [name.replace('final.earth.', '') for name in output_names]
    
    # Calculate average absolute sensitivity (average of |+delta| and |-delta|)
    avg_sensitivity = np.mean(np.abs(sensitivity_matrix), axis=2)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(20,16))
    
    # Plot 1: Average absolute sensitivity
    im1 = axes[0,0].imshow(avg_sensitivity, aspect='auto', cmap='Purples', 
                          interpolation='nearest', vmin=0)
    axes[0,0].set_title(r'Average Absolute Sensitivity (|$\Delta${delta_percent}\%|)'.format(delta_percent=delta_percent), fontsize=label_fs)
    axes[0,0].set_xlabel('Input Parameters', fontsize=label_fs)
    axes[0,0].set_ylabel('Output Parameters', fontsize=label_fs)
    axes[0,0].set_xticks(range(len(param_labels)))
    axes[0,0].set_xticklabels(param_labels, rotation=45, ha='right')
    axes[0,0].set_yticks(range(len(output_labels)))
    axes[0,0].set_yticklabels(output_labels)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=axes[0,0], shrink=0.8)
    cbar1.set_label('\% Change in Output', fontsize=label_fs)
    
    # Plot 2: +delta sensitivity  
    im2 = axes[0,1].imshow(sensitivity_matrix[:,:,0], aspect='auto', cmap='RdBu_r', 
                          interpolation='nearest', 
                          vmin=-np.max(np.abs(sensitivity_matrix)), 
                          vmax=np.max(np.abs(sensitivity_matrix)))
    axes[0,1].set_title(f'+{delta_percent}\% Parameter Change', fontsize=label_fs)
    axes[0,1].set_xlabel('Input Parameters', fontsize=label_fs)
    axes[0,1].set_ylabel('Output Parameters', fontsize=label_fs)
    axes[0,1].set_xticks(range(len(param_labels)))
    axes[0,1].set_xticklabels(param_labels, rotation=45, ha='right')
    axes[0,1].set_yticks(range(len(output_labels)))
    axes[0,1].set_yticklabels(output_labels)
    
    cbar2 = plt.colorbar(im2, ax=axes[0,1], shrink=0.8)
    cbar2.set_label('\% Change in Output', fontsize=label_fs)
    
    # Plot 3: -delta sensitivity
    im3 = axes[1,0].imshow(sensitivity_matrix[:,:,1], aspect='auto', cmap='RdBu', 
                          interpolation='nearest',
                          vmin=-np.max(np.abs(sensitivity_matrix)), 
                          vmax=np.max(np.abs(sensitivity_matrix)))
    axes[1,0].set_title(f'-{delta_percent}\% Parameter Change', fontsize=label_fs)
    axes[1,0].set_xlabel('Input Parameters', fontsize=label_fs)
    axes[1,0].set_ylabel('Output Parameters', fontsize=label_fs)
    axes[1,0].set_xticks(range(len(param_labels)))
    axes[1,0].set_xticklabels(param_labels, rotation=45, ha='right')
    axes[1,0].set_yticks(range(len(output_labels)))
    axes[1,0].set_yticklabels(output_labels)
    
    cbar3 = plt.colorbar(im3, ax=axes[1,0], shrink=0.8)
    cbar3.set_label('\% Change in Output', fontsize=label_fs)
    
    # Plot 4: Most sensitive parameter for each output
    most_sensitive_idx = np.argmax(avg_sensitivity, axis=1)
    most_sensitive_values = np.max(avg_sensitivity, axis=1)
    
    bars = axes[1,1].barh(range(len(output_labels)), most_sensitive_values)
    axes[1,1].set_title('Most Sensitive Input for Each Output', fontsize=label_fs)
    axes[1,1].set_xlabel('Maximum Sensitivity (\% change)', fontsize=label_fs)
    axes[1,1].set_yticks(range(len(output_labels)))
    axes[1,1].set_yticklabels(output_labels)
    
    # Add labels showing which parameter is most sensitive
    for i, (idx, val) in enumerate(zip(most_sensitive_idx, most_sensitive_values)):
        if not np.isnan(val):
            axes[1,1].text(val + 0.01*np.max(most_sensitive_values), i, 
                          param_labels[idx], va='center', fontsize=label_fs)
    
    plt.tight_layout()
    plt.show()
    
    return fig