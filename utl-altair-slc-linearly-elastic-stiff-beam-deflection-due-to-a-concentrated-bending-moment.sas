%let pgm=utl-altair-slc-linearly-elastic-stiff-beam-deflection-due-to-a-concentrated-bending-moment;

%stop_submission;

Altair slc linearly elastic stiff beam deflection due to a concentrated bending moment

Too long to post on list, see github
https://github.com/rogerjdeangelis/utl-altair-slc-linearly-elastic-stiff-beam-deflection-due-to-a-concentrated-bending-moment

Graphic Output
https://github.com/rogerjdeangelis/utl-altair-slc-linearly-elastic-stiff-beam-deflection-due-to-a-concentrated-bending-moment/blob/main/beam_analysis_report_20260218_140316.pdf

WHAT WE WANT
                                                                                Geneally
Ordered by Beam Material descending Absolute Max Deflection                     Deflection
                                                                                Decreases
                           Absolute                                             with increased rigidity
                              max        Beam       Cross-                      Flexural
                          deflection    length     section     Cross-section    rigidity
Obs    Beam material         (mm)         (m)     width (m)      height (m)      (kN·m²)

  1    Steel Beam C         0.64150       6.0        0.08           0.15         4500.00
  2    Wood Beam            0.36084       4.5        0.20           0.30         4500.00
  3    Copper Beam          0.16199       2.5        0.10           0.15         3093.75
  4    Aluminum Beam 2      0.15495       4.0        0.18           0.20         8280.00
  5    Steel Beam A         0.15035       5.0        0.10           0.20        13333.33
  6    Aluminum Beam        0.14347       3.0        0.15           0.18         5030.10
  7    Steel Beam E         0.11390       5.5        0.12           0.22        21296.00
  8    Composite Beam       0.06757       4.8        0.14           0.25        27343.75
  9    Steel Beam D         0.04911       3.5        0.15           0.20        20000.00
 10    Steel Beam B         0.04106       4.0        0.12           0.25        31250.00

GENERATED FILES:

  1  Input slc table beam.sas7bdat
  2. CSV data file: d:/csv/beam_analysis_results.csv
  3. PDF report: d:/pdf/beam_analysis_report_20260218_124547.pdf
  4  output slc table d:/wpswrxx/results_df_final.sas7bdat

PDF REPORT CONTAINS:

  - Title page with summary
  - Comparison plots across all beams
  - Detailed analysis of selected beam
  - Individual beam deflection profiles
  - Statistical analysis


Deepseek prompt

Python please provide a complete reproducible for linearly elastic stiff beam deflection due to a concentrated bending moment amd print
the max deflection

https://chat.deepseek.com/a/chat/s/c5de2620-56cb-47a2-b53b-a62379ce0d16


KEEP IN MIND:

For a simply supported beam with a concentrated moment at the center:

The beam deflects in an S-shape
One side deflects upward, the other side deflects downward
The maximum absolute deflection is what matters structurally

Prep

Folders

 d:/pdf
 d:/wpswrkx
 libname works "d:/wpswrkx"; /*--- ihave this in my autoexec ---*/

/*                   _
(_)_ __  _ __  _   _| |_
| | `_ \| `_ \| | | | __|
| | | | | |_) | |_| | |_
|_|_| |_| .__/ \__,_|\__|
        |_|
*/

proc datasets lib=workx kill nodetails nolist;
run;quit;

%utlfkil(d:/csv/beam_analysis_results.csv);

options validvarname=v7 ls=255 ps=255;
data workx.beam_inp;
label
  name     = "Beam"
  L        = "Length of the beam (m)"
  E        = "Young's modulus of the beam material (Pa)"
  b        = "Width of rectangular beam cross-section (m)"
  h        = "Height of rectangular beam cross-section (m)"
  material = "Beam material type"
  ;
informat
  name $15.
  L best32.
  E  best32.
  b  best32.
  h   best32.
  material $9.
 ;
input
  name & L E b h material;
cards4;
Steel Beam A  5 200000000000 0.1 0.2 Steel
Steel Beam B  4 200000000000 0.12 0.25 Steel
Aluminum Beam  3 69000000000 0.15 0.18 Aluminum
Wood Beam  4.5 10000000000 0.2 0.3 Wood
Steel Beam C  6 200000000000 0.08 0.15 Steel
Steel Beam D  3.5 200000000000 0.15 0.2 Steel
Copper Beam  2.5 110000000000 0.1 0.15 Copper
Steel Beam E  5.5 200000000000 0.12 0.22 Steel
Aluminum Beam 2  4 69000000000 0.18 0.2 Aluminum
Composite Beam  4.8 150000000000 0.14 0.25 Composite
;;;;
run;quit;

proc print data=workx.beam_inp label;
run;

/**************************************************************************************************************************/
/*                     WORKX.BEAM_INP total obs=10                                                                        */
/*                                                                                                                        */
/*                                    Young's         Width of        Height of                                           */
/*                                  modulus of      rectangular      rectangular                                          */
/*                    Length of      the beam           beam             beam        Beam                                 */
/*                     the beam      material      cross-section    cross-section    material                             */
/* NAME                  (m)           (Pa)             (m)              (m)         type                                 */
/*                                                                                                                        */
/* Steel Beam A          5.0       200000000000         0.10             0.20        Steel                                */
/* Steel Beam B          4.0       200000000000         0.12             0.25        Steel                                */
/* Aluminum Beam         3.0        69000000000         0.15             0.18        Aluminum                             */
/* Wood Beam             4.5        10000000000         0.20             0.30        Wood                                 */
/* Steel Beam C          6.0       200000000000         0.08             0.15        Steel                                */
/* Steel Beam D          3.5       200000000000         0.15             0.20        Steel                                */
/* Copper Beam           2.5       110000000000         0.10             0.15        Copper                               */
/* Steel Beam E          5.5       200000000000         0.12             0.22        Steel                                */
/* Aluminum Beam 2       4.0        69000000000         0.18             0.20        Aluminum                             */
/* Composite Beam        4.8       150000000000         0.14             0.25        Composite                            */
/*                                                                                                                        */
/* Middle Observation(5 ) of table = WORKX.BEAM_INP - Total Obs 10                                                        */
/*                                                                                                                        */
/*  -- CHARACTER --                                                                                                       */
/* Variable          Typ    Value               Label                                                                     */
/*                                                                                                                        */
/* material           C9    Steel               Beam material type                                                        */
/* name               C15   Steel Beam C        Beam                                                                      */
/*                                                                                                                        */
/*                                                                                                                        */
/*  -- NUMERIC --                                                                                                         */
/* L                  N8               6        Length of the beam (m)                                                    */
/* E                  N8    200000000000        Young's modulus of the beam material (Pa)                                 */
/* b                  N8            0.08        Width of rectangular beam cross-section (m)                               */
/* h                  N8            0.15        Height of rectangular beam cross-section (m)                              */
/**************************************************************************************************************************/

/*
 _ __  _ __ ___   ___ ___  ___ ___
| `_ \| `__/ _ \ / __/ _ \/ __/ __|
| |_) | | | (_) | (_|  __/\__ \__ \
| .__/|_|  \___/ \___\___||___/___/
|_|
*/

options set=PYTHONHOME "D:\py314";
proc python;
submit;
import pyreadstat as ps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import fsolve
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime

# Set plotting style for better visualization
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.grid'] = True
# Use non-interactive backend for batch processing
matplotlib.use('Agg')

def beam_deflection_concentrated_moment(L, E, I, M0, x_moment, n_points=1000):
    """
    Calculate deflection of a beam with a concentrated bending moment.
    """
    # Create spatial discretization
    x = np.linspace(0, L, n_points)

    # Initialize arrays
    moment = np.zeros_like(x)

    # Calculate reactions for simply supported beam
    R1 = -M0 / L  # Reaction at left support

    # Bending moment distribution
    for i, xi in enumerate(x):
        if xi <= x_moment:
            moment[i] = R1 * xi
        else:
            moment[i] = R1 * xi + M0

    # Calculate slope using integration of M/(EI)
    M_over_EI = moment / (E * I)
    slope_integral = cumulative_trapezoid(M_over_EI, x, initial=0)

    # For simply supported beam, find theta0 such that deflection is zero at supports
    def get_deflection(theta0):
        slope = theta0 + slope_integral
        deflection = cumulative_trapezoid(slope, x, initial=0)
        return deflection

    def objective(theta0):
        deflection = get_deflection(theta0)
        return deflection[-1]  # Deflection at x=L should be 0

    # Solve for theta0 with better initial guess
    theta0_initial_guess = -M0 * L / (6 * E * I)  # Better initial guess
    try:
        theta0_solution = fsolve(objective, theta0_initial_guess, full_output=False)[0]
    except:
        theta0_solution = theta0_initial_guess  # Fallback to initial guess if fsolve fails

    # Calculate final slope and deflection
    slope = theta0_solution + slope_integral
    deflection = cumulative_trapezoid(slope, x, initial=0)

    # Find maximum deflection
    max_deflection_idx = np.argmax(np.abs(deflection))
    max_deflection = deflection[max_deflection_idx]
    max_deflection_location = x[max_deflection_idx]

    return x, deflection, slope, moment, max_deflection, max_deflection_location

def analytical_max_deflection(L, E, I, M0, x_moment):
    """
    Calculate analytical maximum deflection for a simply supported beam
    with concentrated moment.
    """
    a = x_moment
    b = L - a

    # For verification, when the moment is at the center
    if np.isclose(a, L/2):
        max_deflection_analytical = M0 * L**2 / (16 * E * I)
    else:
        # General formula for max deflection
        max_deflection_analytical = abs(M0 * a * b * (a - b)) / (3 * E * I * L**2) * np.sqrt(3)

    return max_deflection_analytical

def analyze_multiple_beams(beam_df, M0, x_moment_rel=0.5, n_points=1000):
    """
    Analyze multiple beams from a DataFrame
    """
    results = []

    for idx, row in beam_df.iterrows():
        # Extract beam properties
        L = row['L']
        E = row['E']
        b = row['b']
        h = row['h']

        # Calculate moment of inertia
        I = b * h**3 / 12

        # Get beam name if available, otherwise use index
        beam_name = row.get('name', f'Beam_{idx+1}')

        # Calculate moment location
        x_moment = x_moment_rel * L

        # Perform analysis
        try:
            x, deflection, slope, moment, max_def, max_def_loc = beam_deflection_concentrated_moment(
                L, E, I, M0, x_moment, n_points
            )

            # Calculate analytical max deflection
            max_def_analytical = analytical_max_deflection(L, E, I, M0, x_moment)

            # Store results
            results.append({
                'name': beam_name,
                'L_m': L,
                'E_GPa': E/1e9,
                'b_m': b,
                'h_m': h,
                'I_m4': I,
                'EI_kNm2': E*I/1e3,
                'x_moment_m': x_moment,
                'max_deflection_mm': max_def * 1000,
                'max_deflection_loc_m': max_def_loc,
                'max_deflection_analytical_mm': max_def_analytical * 1000,
                'max_slope_mrad': np.max(np.abs(slope)) * 1000,
                'max_moment_kNm': np.max(np.abs(moment)) / 1000
            })
        except Exception as e:
            print(f"Error analyzing beam {beam_name}: {e}")
            continue

    return pd.DataFrame(results)

def save_plots_to_pdf(beam_df, results_df, M0, x_moment_rel, filename=None):
    """
    Save all plots to a PDF file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"d:/pdf/beam_analysis_report_{timestamp}.pdf"

    # Create output directory if it doesn't exist
    output_dir = "beam_analysis_reports"
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)

    with PdfPages(filepath) as pdf:

        # Page 1: Title and Summary
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Beam Deflection Analysis Report', fontsize=16, fontweight='bold', y=0.95)

        # Add analysis information
        plt.figtext(0.1, 0.85, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                   fontsize=10, fontfamily='monospace')
        plt.figtext(0.1, 0.8, f"Concentrated Moment (M0): {M0} N·m", fontsize=10)
        plt.figtext(0.1, 0.75, f"Relative Moment Location: {x_moment_rel}", fontsize=10)

        # Create summary table
        plt.figtext(0.1, 0.65, "Analysis Summary:", fontsize=12, fontweight='bold')

        # Create a table of key results
        table_data = []
        headers = ['Beam Name', 'L (m)', 'I (m4)', 'EI (kN·m²)', 'Max Defl (mm)']

        for idx, row in results_df.head(5).iterrows():
            table_data.append([
                row['name'][:20],
                f"{row['L_m']:.2f}",
                f"{row['I_m4']:.6f}",
                f"{row['EI_kNm2']:.1f}",
                f"{row['max_deflection_mm']:.3f}"
            ])

        # Create table
        table_ax = fig.add_axes([0.1, 0.3, 0.8, 0.3])
        table_ax.axis('off')
        table = table_ax.table(cellText=table_data, colLabels=headers,
                              loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Add statistics
        avg_defl = results_df['max_deflection_mm'].mean()
        max_defl = results_df['max_deflection_mm'].max()
        min_defl = results_df['max_deflection_mm'].min()

        plt.figtext(0.1, 0.2, f"Statistics:", fontsize=12, fontweight='bold')
        plt.figtext(0.1, 0.15, f"  Average Deflection: {avg_defl:.3f} mm", fontsize=10)
        plt.figtext(0.1, 0.12, f"  Maximum Deflection: {max_defl:.3f} mm", fontsize=10)
        plt.figtext(0.1, 0.09, f"  Minimum Deflection: {min_defl:.3f} mm", fontsize=10)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Comparison Plots
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Beam Analysis Results Comparison', fontsize=14, fontweight='bold')

        # Plot 1: Maximum deflection comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(results_df['name'], results_df['max_deflection_mm'], color='steelblue')
        ax1.set_xlabel('Beam')
        ax1.set_ylabel('Max Deflection (mm)')
        ax1.set_title('Maximum Deflection by Beam')
        ax1.tick_params(axis='x', rotation=45)

        # Plot 2: Flexural rigidity vs deflection
        ax2 = axes[0, 1]
        scatter = ax2.scatter(results_df['EI_kNm2'], results_df['max_deflection_mm'],
                             c=range(len(results_df)), cmap='viridis', s=100)
        ax2.set_xlabel('Flexural Rigidity EI (kN·m²)')
        ax2.set_ylabel('Max Deflection (mm)')
        ax2.set_title('Deflection vs Flexural Rigidity')
        plt.colorbar(scatter, ax=ax2, label='Beam Index')

        # Plot 3: Cross-section dimensions
        ax3 = axes[1, 0]
        width = 0.35
        x_pos = np.arange(len(results_df))
        ax3.bar(x_pos - width/2, results_df['b_m']*1000, width, label='Width (mm)', color='lightcoral')
        ax3.bar(x_pos + width/2, results_df['h_m']*1000, width, label='Height (mm)', color='lightblue')
        ax3.set_xlabel('Beam')
        ax3.set_ylabel('Dimension (mm)')
        ax3.set_title('Cross-section Dimensions')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(results_df['name'], rotation=45, ha='right')
        ax3.legend()

        # Plot 4: Numerical vs Analytical
        ax4 = axes[1, 1]
        x_pos = np.arange(len(results_df))
        width = 0.35
        ax4.bar(x_pos - width/2, results_df['max_deflection_mm'], width, label='Numerical', color='forestgreen')
        ax4.bar(x_pos + width/2, results_df['max_deflection_analytical_mm'], width, label='Analytical', color='goldenrod')
        ax4.set_xlabel('Beam')
        ax4.set_ylabel('Max Deflection (mm)')
        ax4.set_title('Numerical vs Analytical Results')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(results_df['name'], rotation=45, ha='right')
        ax4.legend()

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Detailed Analysis for Selected Beam
        selected_beam = beam_df.iloc[0]
        L_selected = selected_beam['L']
        E_selected = selected_beam['E']
        b_selected = selected_beam['b']
        h_selected = selected_beam['h']
        I_selected = b_selected * h_selected**3 / 12

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(f'Detailed Analysis: {selected_beam["name"]}', fontsize=14, fontweight='bold')

        # Plot for center moment
        x_moment_center = x_moment_rel * L_selected
        x, deflection, slope, moment, max_def, max_loc = beam_deflection_concentrated_moment(
            L_selected, E_selected, I_selected, M0, x_moment_center
        )

        # Deflection plot
        ax1 = axes[0, 0]
        ax1.plot(x, deflection*1000, 'b-', linewidth=2)
        ax1.axvline(x=x_moment_center, color='r', linestyle='--', alpha=0.7)
        ax1.scatter([max_loc], [max_def*1000], color='g', s=100, zorder=5)
        ax1.set_xlabel('Position (m)')
        ax1.set_ylabel('Deflection (mm)')
        ax1.set_title(f'Deflection (Moment at x={x_moment_center:.2f}m)')
        ax1.grid(True, alpha=0.3)

        # Slope plot
        ax2 = axes[0, 1]
        ax2.plot(x, slope*1000, 'g-', linewidth=2)
        ax2.axvline(x=x_moment_center, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Position (m)')
        ax2.set_ylabel('Slope (mrad)')
        ax2.set_title('Beam Slope')
        ax2.grid(True, alpha=0.3)

        # Moment plot
        ax3 = axes[1, 0]
        ax3.plot(x, moment/1000, 'm-', linewidth=2)
        ax3.axvline(x=x_moment_center, color='r', linestyle='--', alpha=0.7)
        ax3.fill_between(x, moment/1000, alpha=0.3, color='m')
        ax3.set_xlabel('Position (m)')
        ax3.set_ylabel('Bending Moment (kN·m)')
        ax3.set_title('Bending Moment Diagram')
        ax3.grid(True, alpha=0.3)

        # Deflection at different moment locations
        ax4 = axes[1, 1]
        moment_locations = np.linspace(0.1, 0.9, 5)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(moment_locations)))

        for x_rel, color in zip(moment_locations, colors):
            x_m = x_rel * L_selected
            x_d, defl_d, _, _, _, _ = beam_deflection_concentrated_moment(
                L_selected, E_selected, I_selected, M0, x_m
            )
            ax4.plot(x_d, defl_d*1000, color=color, linewidth=1.5,
                    label=f'x/L={x_rel:.1f}')

        ax4.set_xlabel('Position (m)')
        ax4.set_ylabel('Deflection (mm)')
        ax4.set_title('Deflection for Different Moment Locations')
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Individual Beam Analyses
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Individual Beam Deflection Profiles', fontsize=14, fontweight='bold')

        # Create subplots for each beam (max 6 beams per page)
        n_beams = min(len(beam_df), 6)
        for i in range(n_beams):
            ax = plt.subplot(2, 3, i+1)

            beam = beam_df.iloc[i]
            L_b = beam['L']
            E_b = beam['E']
            b_b = beam['b']
            h_b = beam['h']
            I_b = b_b * h_b**3 / 12
            beam_name = beam.get('name', f'Beam_{i+1}')

            x_m = x_moment_rel * L_b
            x_b, defl_b, _, _, max_d, max_loc = beam_deflection_concentrated_moment(
                L_b, E_b, I_b, M0, x_m
            )

            ax.plot(x_b, defl_b*1000, 'b-', linewidth=2)
            ax.axvline(x=x_m, color='r', linestyle='--', alpha=0.5)
            ax.scatter([max_loc], [max_d*1000], color='g', s=50, zorder=5)
            ax.set_xlabel('Position (m)')
            ax.set_ylabel('Defl (mm)')
            ax.set_title(f'{beam_name[:20]}')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=8)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: Statistical Analysis
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Statistical Analysis', fontsize=14, fontweight='bold')

        # Histogram of deflections
        ax1 = axes[0, 0]
        ax1.hist(results_df['max_deflection_mm'], bins=10, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Max Deflection (mm)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Maximum Deflections')
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2 = axes[0, 1]
        ax2.boxplot(results_df['max_deflection_mm'])
        ax2.set_ylabel('Max Deflection (mm)')
        ax2.set_title('Deflection Box Plot')
        ax2.set_xticklabels(['All Beams'])
        ax2.grid(True, alpha=0.3)

        # Scatter plot of dimensions vs deflection
        ax3 = axes[1, 0]
        area = results_df['b_m'] * results_df['h_m'] * 1e6  # Cross-sectional area in mm²
        scatter = ax3.scatter(area, results_df['max_deflection_mm'],
                            c=results_df['EI_kNm2'], s=100, cmap='viridis')
        ax3.set_xlabel('Cross-sectional Area (mm²)')
        ax3.set_ylabel('Max Deflection (mm)')
        ax3.set_title('Deflection vs Cross-sectional Area')
        plt.colorbar(scatter, ax=ax3, label='EI (kN·m²)')
        ax3.grid(True, alpha=0.3)

        # Pie chart of materials (if material column exists)
        ax4 = axes[1, 1]
        if 'material' in beam_df.columns:
            material_counts = beam_df['material'].value_counts()
            ax4.pie(material_counts.values, labels=material_counts.index, autopct='%1.1f%%')
            ax4.set_title('Material Distribution')
        else:
            ax4.text(0.5, 0.5, 'Material data not available',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Material Distribution')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Save metadata
        d = pdf.infodict()
        d['Title'] = 'Beam Deflection Analysis Report'
        d['Author'] = 'Python Beam Analysis Script'
        d['Subject'] = f'Analysis of {len(beam_df)} beams under concentrated moment M0={M0} N·m'
        d['Keywords'] = 'beam deflection structural analysis moment'
        d['CreationDate'] = datetime.now()
        d['ModDate'] = datetime.now()

    print(f"\nPDF report saved to: {filepath}")
    return filepath

# Main execution - Modified for Altair SLC environment
print("=" * 60)
print("BEAM DEFLECTION ANALYSIS WITH PDF REPORT GENERATION")
print("=" * 60)

# Create sample input DataFrame
print("\nCreating sample input DataFrame...")

beam_inp,meta = ps.read_sas7bdat('d:/wpswrkx/beam_inp.sas7bdat')
print(beam_inp)

# Save to parquet for potential later use
beam_inp.to_parquet('d:/wpswrkx/beam_inp.parquet', engine='pyarrow')

print(f"Created DataFrame with {len(beam_inp)} beams")
print(f"Columns: {list(beam_inp.columns)}")

# Define loading parameters
M0 = 10000  # Concentrated moment (N·m)
x_moment_rel = 0.5  # Moment at center

# Analyze all beams
print("\nAnalyzing all beams...")
results_df = analyze_multiple_beams(beam_inp, M0, x_moment_rel)

print("\nAnalysis complete. Results summary:")
print(results_df[['name', 'max_deflection_mm', 'EI_kNm2']].to_string(index=False))

# Save results to CSV
csv_filename = 'd:/csv/beam_analysis_results.csv'
results_df.to_csv(csv_filename, index=False)
print(f"\nResults saved to '{csv_filename}'")

# Generate PDF report
print("\nGenerating PDF report...")
pdf_file = save_plots_to_pdf(beam_inp, results_df, M0, x_moment_rel)

print("\n" + "=" * 60)
print("PROCESS COMPLETE")
print("=" * 60)
print(f"\nGenerated files:")
print(f"  1. CSV data file: {csv_filename}")
print(f"  2. PDF report: {pdf_file}")
print(f"\nPDF report contains:")
print("  - Title page with summary")
print("  - Comparison plots across all beams")
print("  - Detailed analysis of selected beam")
print("  - Individual beam deflection profiles")
print("  - Statistical analysis")

# Skip interactive display in Altair SLC environment
print("\nSkipping interactive plot display (not available in Altair SLC)")
print("All results have been saved to files.")

results_df.to_parquet('d:/wpswrkx/results_df.parquet', engine='pyarrow')

print("\nDone!")
endsubmit;
run;


/*--- USE PYTHON 310 FOR CREATING SAS DATASETS ONLY ---*/

options validvarname=v7; /*--- very important pyhton is case sensitive ---*/
options set=PYTHONHOME "D:\py310";
proc python;
submit;
import pyarrow
import pandas as pd
# beam_inp = pd.read_parquet('d:/wpswrkx/beam_inp.parquet', engine='pyarrow')
results_df = pd.read_parquet('d:/wpswrkx/results_df.parquet', engine='pyarrow')
# print(beam_inp)
print(results_df)
endsubmit;
import python=results_df data=workx.results_df;
run;quit;

/*--- LIST OUTPUT ---*/

data workx.results_df_final;
  label
    name = "Beam material"
    abs_max_deflection_mm = "Absolute max deflection (mm)"
    L_m = "Beam length (m)"
    b_m = "Cross-section width (m)"
    h_m = "Cross-section height (m)"
    E_GPa = "Young's modulus (GPa)"
    I_m4 = "Moment of inertia (m4)"
    EI_kNm2 = "Flexural rigidity (kN·m²)"
    x_moment_m = "Moment application location (m)"
    max_deflection_mm = "Maximum deflection (mm)"
    max_deflection_loc_m = "Location of max deflection (m)"
    max_deflection_analytical_mm = "Analytical max deflection (mm)"
    max_slope_mrad = "Maximum slope (milliradians)"
    max_moment_kNm = "Maximum bending moment (kN·m)"
   ;
  set workx.results_df;
    abs_max_deflection_mm = abs(max_deflection_mm);
run;

proc sort data=workx.results_df_final;
by descending abs_max_deflection_mm;
run;quit;

proc print data=workx.results_df_final label;
Title "Ordered by Beam Material descending Absolute Max Deflection";
var name abs_max_deflection_mm L_m b_m h_m EI_kNm2;
run;quit;

/**************************************************************************************************************************/
/*  CSV d:/csv/beam_analysis_results.csv                                                                                  */
/* PDF d:/pdf/beam_analysis_report_20260218_124547.pdf                                                                    */
/* INPUT: d:/wpswrkx/beam_inp.sas7bdat                                                                                    */
/* OUTPUT: d:/wpswrkx/results_df_final.sas7bdat                                                                           */
/*                                                                                                                        */
/* d:/wpswrkx/results_df_final.sas7bdat                                             Geneally                              */
/* Ordered By Dscending Absolute Max Deflection                     Deflection                                            */
/*                                                                                 Decreases                              */
/*                            Absolute                                             with increased rigidity                */
/*                               max        Beam       Cross-                      Flexural                               */
/*                           deflection    length     section     Cross-section    rigidity                               */
/* Obs    Beam material         (mm)         (m)     width (m)      height (m)      (kN·m²)                               */
/*                                                                                                                        */
/*   1    Steel Beam C         0.64150       6.0        0.08           0.15         4500.00                               */
/*   2    Wood Beam            0.36084       4.5        0.20           0.30         4500.00                               */
/*   3    Copper Beam          0.16199       2.5        0.10           0.15         3093.75                               */
/*   4    Aluminum Beam 2      0.15495       4.0        0.18           0.20         8280.00                               */
/*   5    Steel Beam A         0.15035       5.0        0.10           0.20        13333.33                               */
/*   6    Aluminum Beam        0.14347       3.0        0.15           0.18         5030.10                               */
/*   7    Steel Beam E         0.11390       5.5        0.12           0.22        21296.00                               */
/*   8    Composite Beam       0.06757       4.8        0.14           0.25        27343.75                               */
/*   9    Steel Beam D         0.04911       3.5        0.15           0.20        20000.00                               */
/*  10    Steel Beam B         0.04106       4.0        0.12           0.25        31250.00                               */
/*                                                                                                                        */
/*  Middle Observation(5 ) of table = workx.results_df_final - Total Obs 10                                               */
/*                                                                                                                        */
/*   -- CHARACTER --                                                                                                      */
/*  Variable                        Type   Value           Label                                                          */
/*                                                                                                                        */
/*  name                             C15   Steel Beam A    Beam material                                                  */
/*                                                                                                                        */
/*                                                                                                                        */
/*   -- NUMERIC --                                                                                                        */
/*  abs_max_deflection_mm            N8    0.1503505508    Absolute max deflection (mm)                                   */
/*  L_m                              N8               5    Beam length (m)                                                */
/*  b_m                              N8             0.1    Cross-section width (m)                                        */
/*  h_m                              N8             0.2    Cross-section height (m)                                       */
/*  I_m4                             N8    0.0000666667    Moment of inertia (m4)                                         */
/*  EI_kNm2                          N8    13333.333333    Flexural rigidity (kN·m²)                                      */
/*  x_moment_m                       N8             2.5    Moment application location (m)                                */
/*  max_deflection_mm                N8    -0.150350551    Maximum deflection (mm)                                        */
/*  E_GPa                            N8             200    Young's modulus (GPa)                                          */
/*  max_deflection_loc_m             N8    3.5585585586    Location of max deflection (m)                                 */
/*  max_deflection_analytical_mm     N8        1.171875    Analytical max deflection (mm)                                 */
/*  max_slope_mrad                   N8    0.3115621878    Maximum slope (milliradians)                                   */
/*  max_moment_kNm                   N8     4.994994995    Maximum bending moment (kN·m)                                  */
/**************************************************************************************************************************/

/*
| | ___   __ _
| |/ _ \ / _` |
| | (_) | (_| |
|_|\___/ \__, |
         |___/
*/

1                                          Altair SLC    14:03 Wednesday, February 18, 2026

NOTE: Copyright 2002-2025 World Programming, an Altair Company
NOTE: Altair SLC 2026 (05.26.01.00.000758)
      Licensed to Roger DeAngelis
NOTE: This session is executing on the X64_WIN11PRO platform and is running in 64 bit mode

NOTE: AUTOEXEC processing beginning; file is C:\wpsoto\autoexec.sas
NOTE: AUTOEXEC source line
1       +  ï»¿ods _all_ close;
           ^
ERROR: Expected a statement keyword : found "?"
NOTE: Library workx assigned as follows:
      Engine:        SAS7BDAT
      Physical Name: d:\wpswrkx

NOTE: Library slchelp assigned as follows:
      Engine:        WPD
      Physical Name: C:\Progra~1\Altair\SLC\2026\sashelp

NOTE: Library worksas assigned as follows:
      Engine:        SAS7BDAT
      Physical Name: d:\worksas

NOTE: Library workwpd assigned as follows:
      Engine:        WPD
      Physical Name: d:\workwpd


LOG:  14:03:14
NOTE: 1 record was written to file PRINT

NOTE: The data step took :
      real time : 0.031
      cpu time  : 0.015


NOTE: AUTOEXEC processing completed

1          options set=PYTHONHOME "D:\py314";
2         proc python;
3         submit;
4         import pyreadstat as ps
5         import numpy as np
6         import pandas as pd
7         import matplotlib.pyplot as plt
8         from scipy.integrate import cumulative_trapezoid
9         from scipy.optimize import fsolve
10        import matplotlib
11        from matplotlib.backends.backend_pdf import PdfPages
12        import os
13        from datetime import datetime
14
15        # Set plotting style for better visualization
16        matplotlib.rcParams['font.size'] = 12
17        matplotlib.rcParams['axes.grid'] = True
18        # Use non-interactive backend for batch processing
19        matplotlib.use('Agg')
20
21        def beam_deflection_concentrated_moment(L, E, I, M0, x_moment, n_points=1000):
22            """
23            Calculate deflection of a beam with a concentrated bending moment.
24            """
25            # Create spatial discretization
26            x = np.linspace(0, L, n_points)
27
28            # Initialize arrays
29            moment = np.zeros_like(x)
30
31            # Calculate reactions for simply supported beam
32            R1 = -M0 / L  # Reaction at left support
33
34            # Bending moment distribution
35            for i, xi in enumerate(x):
36                if xi <= x_moment:
37                    moment[i] = R1 * xi
38                else:
39                    moment[i] = R1 * xi + M0
40
41            # Calculate slope using integration of M/(EI)
42            M_over_EI = moment / (E * I)
43            slope_integral = cumulative_trapezoid(M_over_EI, x, initial=0)
44
45            # For simply supported beam, find theta0 such that deflection is zero at supports
46            def get_deflection(theta0):
47                slope = theta0 + slope_integral
48                deflection = cumulative_trapezoid(slope, x, initial=0)
49                return deflection
50
51            def objective(theta0):
52                deflection = get_deflection(theta0)
53                return deflection[-1]  # Deflection at x=L should be 0
54
55            # Solve for theta0 with better initial guess
56            theta0_initial_guess = -M0 * L / (6 * E * I)  # Better initial guess
57            try:
58                theta0_solution = fsolve(objective, theta0_initial_guess, full_output=False)[0]
59            except:
60                theta0_solution = theta0_initial_guess  # Fallback to initial guess if fsolve fails
61
62            # Calculate final slope and deflection
63            slope = theta0_solution + slope_integral
64            deflection = cumulative_trapezoid(slope, x, initial=0)
65
66            # Find maximum deflection
67            max_deflection_idx = np.argmax(np.abs(deflection))
68            max_deflection = deflection[max_deflection_idx]
69            max_deflection_location = x[max_deflection_idx]
70
71            return x, deflection, slope, moment, max_deflection, max_deflection_location
72
73        def analytical_max_deflection(L, E, I, M0, x_moment):
74            """
75            Calculate analytical maximum deflection for a simply supported beam
76            with concentrated moment.
77            """
78            a = x_moment
79            b = L - a
80
81            # For verification, when the moment is at the center
82            if np.isclose(a, L/2):
83                max_deflection_analytical = M0 * L**2 / (16 * E * I)
84            else:
85                # General formula for max deflection
86                max_deflection_analytical = abs(M0 * a * b * (a - b)) / (3 * E * I * L**2) * np.sqrt(3)
87
88            return max_deflection_analytical
89
90        def analyze_multiple_beams(beam_df, M0, x_moment_rel=0.5, n_points=1000):
91            """
92            Analyze multiple beams from a DataFrame
93            """
94            results = []
95
96            for idx, row in beam_df.iterrows():
97                # Extract beam properties
98                L = row['L']
99                E = row['E']
100               b = row['b']
101               h = row['h']
102
103               # Calculate moment of inertia
104               I = b * h**3 / 12
105
106               # Get beam name if available, otherwise use index
107               beam_name = row.get('name', f'Beam_{idx+1}')
108
109               # Calculate moment location
110               x_moment = x_moment_rel * L
111
112               # Perform analysis
113               try:
114                   x, deflection, slope, moment, max_def, max_def_loc = beam_deflection_concentrated_moment(
115                       L, E, I, M0, x_moment, n_points
116                   )
117
118                   # Calculate analytical max deflection
119                   max_def_analytical = analytical_max_deflection(L, E, I, M0, x_moment)
120
121                   # Store results
122                   results.append({
123                       'name': beam_name,
124                       'L_m': L,
125                       'E_GPa': E/1e9,
126                       'b_m': b,
127                       'h_m': h,
128                       'I_m4': I,
129                       'EI_kNm2': E*I/1e3,
130                       'x_moment_m': x_moment,
131                       'max_deflection_mm': max_def * 1000,
132                       'max_deflection_loc_m': max_def_loc,
133                       'max_deflection_analytical_mm': max_def_analytical * 1000,
134                       'max_slope_mrad': np.max(np.abs(slope)) * 1000,
135                       'max_moment_kNm': np.max(np.abs(moment)) / 1000
136                   })
137               except Exception as e:
138                   print(f"Error analyzing beam {beam_name}: {e}")
139                   continue
140
141           return pd.DataFrame(results)
142
143       def save_plots_to_pdf(beam_df, results_df, M0, x_moment_rel, filename=None):
144           """
145           Save all plots to a PDF file
146           """
147           if filename is None:
148               timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
149               filename = f"d:/pdf/beam_analysis_report_{timestamp}.pdf"
150
151           # Create output directory if it doesn't exist
152           output_dir = "beam_analysis_reports"
153           os.makedirs(output_dir, exist_ok=True)
154
155           filepath = os.path.join(output_dir, filename)
156
157           with PdfPages(filepath) as pdf:
158
159               # Page 1: Title and Summary
160               fig = plt.figure(figsize=(11, 8.5))
161               fig.suptitle('Beam Deflection Analysis Report', fontsize=16, fontweight='bold', y=0.95)
162
163               # Add analysis information
164               plt.figtext(0.1, 0.85, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
165                          fontsize=10, fontfamily='monospace')
166               plt.figtext(0.1, 0.8, f"Concentrated Moment (M0): {M0} NÂ·m", fontsize=10)
167               plt.figtext(0.1, 0.75, f"Relative Moment Location: {x_moment_rel}", fontsize=10)
168
169               # Create summary table
170               plt.figtext(0.1, 0.65, "Analysis Summary:", fontsize=12, fontweight='bold')
171
172               # Create a table of key results
173               table_data = []
174               headers = ['Beam Name', 'L (m)', 'I (m4)', 'EI (kNÂ·mÂ²)', 'Max Defl (mm)']
175
176               for idx, row in results_df.head(5).iterrows():
177                   table_data.append([
178                       row['name'][:20],
179                       f"{row['L_m']:.2f}",
180                       f"{row['I_m4']:.6f}",
181                       f"{row['EI_kNm2']:.1f}",
182                       f"{row['max_deflection_mm']:.3f}"
183                   ])
184
185               # Create table
186               table_ax = fig.add_axes([0.1, 0.3, 0.8, 0.3])
187               table_ax.axis('off')
188               table = table_ax.table(cellText=table_data, colLabels=headers,
189                                     loc='center', cellLoc='left')
190               table.auto_set_font_size(False)
191               table.set_fontsize(10)
192               table.scale(1, 1.5)
193
194               # Add statistics
195               avg_defl = results_df['max_deflection_mm'].mean()
196               max_defl = results_df['max_deflection_mm'].max()
197               min_defl = results_df['max_deflection_mm'].min()
198
199               plt.figtext(0.1, 0.2, f"Statistics:", fontsize=12, fontweight='bold')
200               plt.figtext(0.1, 0.15, f"  Average Deflection: {avg_defl:.3f} mm", fontsize=10)
201               plt.figtext(0.1, 0.12, f"  Maximum Deflection: {max_defl:.3f} mm", fontsize=10)
202               plt.figtext(0.1, 0.09, f"  Minimum Deflection: {min_defl:.3f} mm", fontsize=10)
203
204               pdf.savefig(fig)
205               plt.close(fig)
206
207               # Page 2: Comparison Plots
208               fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
209               fig.suptitle('Beam Analysis Results Comparison', fontsize=14, fontweight='bold')
210
211               # Plot 1: Maximum deflection comparison
212               ax1 = axes[0, 0]
213               bars1 = ax1.bar(results_df['name'], results_df['max_deflection_mm'], color='steelblue')
214               ax1.set_xlabel('Beam')
215               ax1.set_ylabel('Max Deflection (mm)')
216               ax1.set_title('Maximum Deflection by Beam')
217               ax1.tick_params(axis='x', rotation=45)
218
219               # Plot 2: Flexural rigidity vs deflection
220               ax2 = axes[0, 1]
221               scatter = ax2.scatter(results_df['EI_kNm2'], results_df['max_deflection_mm'],
222                                    c=range(len(results_df)), cmap='viridis', s=100)
223               ax2.set_xlabel('Flexural Rigidity EI (kNÂ·mÂ²)')
224               ax2.set_ylabel('Max Deflection (mm)')
225               ax2.set_title('Deflection vs Flexural Rigidity')
226               plt.colorbar(scatter, ax=ax2, label='Beam Index')
227
228               # Plot 3: Cross-section dimensions
229               ax3 = axes[1, 0]
230               width = 0.35
231               x_pos = np.arange(len(results_df))
232               ax3.bar(x_pos - width/2, results_df['b_m']*1000, width, label='Width (mm)', color='lightcoral')
233               ax3.bar(x_pos + width/2, results_df['h_m']*1000, width, label='Height (mm)', color='lightblue')
234               ax3.set_xlabel('Beam')
235               ax3.set_ylabel('Dimension (mm)')
236               ax3.set_title('Cross-section Dimensions')
237               ax3.set_xticks(x_pos)
238               ax3.set_xticklabels(results_df['name'], rotation=45, ha='right')
239               ax3.legend()
240
241               # Plot 4: Numerical vs Analytical
242               ax4 = axes[1, 1]
243               x_pos = np.arange(len(results_df))
244               width = 0.35
245               ax4.bar(x_pos - width/2, results_df['max_deflection_mm'], width, label='Numerical', color='forestgreen')
246               ax4.bar(x_pos + width/2, results_df['max_deflection_analytical_mm'], width, label='Analytical', color='goldenrod')
247               ax4.set_xlabel('Beam')
248               ax4.set_ylabel('Max Deflection (mm)')
249               ax4.set_title('Numerical vs Analytical Results')
250               ax4.set_xticks(x_pos)
251               ax4.set_xticklabels(results_df['name'], rotation=45, ha='right')
252               ax4.legend()
253
254               plt.tight_layout()
255               pdf.savefig(fig)
256               plt.close(fig)
257
258               # Page 3: Detailed Analysis for Selected Beam
259               selected_beam = beam_df.iloc[0]
260               L_selected = selected_beam['L']
261               E_selected = selected_beam['E']
262               b_selected = selected_beam['b']
263               h_selected = selected_beam['h']
264               I_selected = b_selected * h_selected**3 / 12
265
266               fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
267               fig.suptitle(f'Detailed Analysis: {selected_beam["name"]}', fontsize=14, fontweight='bold')
268
269               # Plot for center moment
270               x_moment_center = x_moment_rel * L_selected
271               x, deflection, slope, moment, max_def, max_loc = beam_deflection_concentrated_moment(
272                   L_selected, E_selected, I_selected, M0, x_moment_center
273               )
274
275               # Deflection plot
276               ax1 = axes[0, 0]
277               ax1.plot(x, deflection*1000, 'b-', linewidth=2)
278               ax1.axvline(x=x_moment_center, color='r', linestyle='--', alpha=0.7)
279               ax1.scatter([max_loc], [max_def*1000], color='g', s=100, zorder=5)
280               ax1.set_xlabel('Position (m)')
281               ax1.set_ylabel('Deflection (mm)')
282               ax1.set_title(f'Deflection (Moment at x={x_moment_center:.2f}m)')
283               ax1.grid(True, alpha=0.3)
284
285               # Slope plot
286               ax2 = axes[0, 1]
287               ax2.plot(x, slope*1000, 'g-', linewidth=2)
288               ax2.axvline(x=x_moment_center, color='r', linestyle='--', alpha=0.7)
289               ax2.set_xlabel('Position (m)')
290               ax2.set_ylabel('Slope (mrad)')
291               ax2.set_title('Beam Slope')
292               ax2.grid(True, alpha=0.3)
293
294               # Moment plot
295               ax3 = axes[1, 0]
296               ax3.plot(x, moment/1000, 'm-', linewidth=2)
297               ax3.axvline(x=x_moment_center, color='r', linestyle='--', alpha=0.7)
298               ax3.fill_between(x, moment/1000, alpha=0.3, color='m')
299               ax3.set_xlabel('Position (m)')
300               ax3.set_ylabel('Bending Moment (kNÂ·m)')
301               ax3.set_title('Bending Moment Diagram')
302               ax3.grid(True, alpha=0.3)
303
304               # Deflection at different moment locations
305               ax4 = axes[1, 1]
306               moment_locations = np.linspace(0.1, 0.9, 5)
307               colors = plt.cm.rainbow(np.linspace(0, 1, len(moment_locations)))
308
309               for x_rel, color in zip(moment_locations, colors):
310                   x_m = x_rel * L_selected
311                   x_d, defl_d, _, _, _, _ = beam_deflection_concentrated_moment(
312                       L_selected, E_selected, I_selected, M0, x_m
313                   )
314                   ax4.plot(x_d, defl_d*1000, color=color, linewidth=1.5,
315                           label=f'x/L={x_rel:.1f}')
316
317               ax4.set_xlabel('Position (m)')
318               ax4.set_ylabel('Deflection (mm)')
319               ax4.set_title('Deflection for Different Moment Locations')
320               ax4.legend(loc='best', fontsize=8)
321               ax4.grid(True, alpha=0.3)
322
323               plt.tight_layout()
324               pdf.savefig(fig)
325               plt.close(fig)
326
327               # Page 4: Individual Beam Analyses
328               fig = plt.figure(figsize=(11, 8.5))
329               fig.suptitle('Individual Beam Deflection Profiles', fontsize=14, fontweight='bold')
330
331               # Create subplots for each beam (max 6 beams per page)
332               n_beams = min(len(beam_df), 6)
333               for i in range(n_beams):
334                   ax = plt.subplot(2, 3, i+1)
335
336                   beam = beam_df.iloc[i]
337                   L_b = beam['L']
338                   E_b = beam['E']
339                   b_b = beam['b']
340                   h_b = beam['h']
341                   I_b = b_b * h_b**3 / 12
342                   beam_name = beam.get('name', f'Beam_{i+1}')
343
344                   x_m = x_moment_rel * L_b
345                   x_b, defl_b, _, _, max_d, max_loc = beam_deflection_concentrated_moment(
346                       L_b, E_b, I_b, M0, x_m
347                   )
348
349                   ax.plot(x_b, defl_b*1000, 'b-', linewidth=2)
350                   ax.axvline(x=x_m, color='r', linestyle='--', alpha=0.5)
351                   ax.scatter([max_loc], [max_d*1000], color='g', s=50, zorder=5)
352                   ax.set_xlabel('Position (m)')
353                   ax.set_ylabel('Defl (mm)')
354                   ax.set_title(f'{beam_name[:20]}')
355                   ax.grid(True, alpha=0.3)
356                   ax.tick_params(axis='both', labelsize=8)
357
358               plt.tight_layout()
359               pdf.savefig(fig)
360               plt.close(fig)
361
362               # Page 5: Statistical Analysis
363               fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
364               fig.suptitle('Statistical Analysis', fontsize=14, fontweight='bold')
365
366               # Histogram of deflections
367               ax1 = axes[0, 0]
368               ax1.hist(results_df['max_deflection_mm'], bins=10, color='skyblue', edgecolor='black')
369               ax1.set_xlabel('Max Deflection (mm)')
370               ax1.set_ylabel('Frequency')
371               ax1.set_title('Distribution of Maximum Deflections')
372               ax1.grid(True, alpha=0.3)
373
374               # Box plot
375               ax2 = axes[0, 1]
376               ax2.boxplot(results_df['max_deflection_mm'])
377               ax2.set_ylabel('Max Deflection (mm)')
378               ax2.set_title('Deflection Box Plot')
379               ax2.set_xticklabels(['All Beams'])
380               ax2.grid(True, alpha=0.3)
381
382               # Scatter plot of dimensions vs deflection
383               ax3 = axes[1, 0]
384               area = results_df['b_m'] * results_df['h_m'] * 1e6  # Cross-sectional area in mmÂ²
385               scatter = ax3.scatter(area, results_df['max_deflection_mm'],
386                                   c=results_df['EI_kNm2'], s=100, cmap='viridis')
387               ax3.set_xlabel('Cross-sectional Area (mmÂ²)')
388               ax3.set_ylabel('Max Deflection (mm)')
389               ax3.set_title('Deflection vs Cross-sectional Area')
390               plt.colorbar(scatter, ax=ax3, label='EI (kNÂ·mÂ²)')
391               ax3.grid(True, alpha=0.3)
392
393               # Pie chart of materials (if material column exists)
394               ax4 = axes[1, 1]
395               if 'material' in beam_df.columns:
396                   material_counts = beam_df['material'].value_counts()
397                   ax4.pie(material_counts.values, labels=material_counts.index, autopct='%1.1f%%')
398                   ax4.set_title('Material Distribution')
399               else:
400                   ax4.text(0.5, 0.5, 'Material data not available',
401                           ha='center', va='center', transform=ax4.transAxes)
402                   ax4.set_title('Material Distribution')
403
404               plt.tight_layout()
405               pdf.savefig(fig)
406               plt.close(fig)
407
408               # Save metadata
409               d = pdf.infodict()
410               d['Title'] = 'Beam Deflection Analysis Report'
411               d['Author'] = 'Python Beam Analysis Script'
412               d['Subject'] = f'Analysis of {len(beam_df)} beams under concentrated moment M0={M0} NÂ·m'
413               d['Keywords'] = 'beam deflection structural analysis moment'
414               d['CreationDate'] = datetime.now()
415               d['ModDate'] = datetime.now()
416
417           print(f"\nPDF report saved to: {filepath}")
418           return filepath
419
420       # Main execution - Modified for Altair SLC environment
421       print("=" * 60)
422       print("BEAM DEFLECTION ANALYSIS WITH PDF REPORT GENERATION")
423       print("=" * 60)
424
425       # Create sample input DataFrame
426       print("\nCreating sample input DataFrame...")
427
428       beam_inp,meta = ps.read_sas7bdat('d:/wpswrkx/beam_inp.sas7bdat')
429       print(beam_inp)
430
431       # Save to parquet for potential later use
432       beam_inp.to_parquet('d:/wpswrkx/beam_inp.parquet', engine='pyarrow')
433
434       print(f"Created DataFrame with {len(beam_inp)} beams")
435       print(f"Columns: {list(beam_inp.columns)}")
436
437       # Define loading parameters
438       M0 = 10000  # Concentrated moment (NÂ·m)
439       x_moment_rel = 0.5  # Moment at center
440
441       # Analyze all beams
442       print("\nAnalyzing all beams...")
443       results_df = analyze_multiple_beams(beam_inp, M0, x_moment_rel)
444
445       print("\nAnalysis complete. Results summary:")
446       print(results_df[['name', 'max_deflection_mm', 'EI_kNm2']].to_string(index=False))
447
448       # Save results to CSV
449       csv_filename = 'd:/csv/beam_analysis_results.csv'
450       results_df.to_csv(csv_filename, index=False)
451       print(f"\nResults saved to '{csv_filename}'")
452
453       # Generate PDF report
454       print("\nGenerating PDF report...")
455       pdf_file = save_plots_to_pdf(beam_inp, results_df, M0, x_moment_rel)
456
457       print("\n" + "=" * 60)
458       print("PROCESS COMPLETE")
459       print("=" * 60)
460       print(f"\nGenerated files:")
461       print(f"  1. CSV data file: {csv_filename}")

2                                                                                                                         Altair SLC

462       print(f"  2. PDF report: {pdf_file}")
463       print(f"\nPDF report contains:")
464       print("  - Title page with summary")
465       print("  - Comparison plots across all beams")
466       print("  - Detailed analysis of selected beam")
467       print("  - Individual beam deflection profiles")
468       print("  - Statistical analysis")
469
470       # Skip interactive display in Altair SLC environment
471       print("\nSkipping interactive plot display (not available in Altair SLC)")
472       print("All results have been saved to files.")
473
474       results_df.to_parquet('d:/wpswrkx/results_df.parquet', engine='pyarrow')
475
476       print("\nDone!")
477       endsubmit;

NOTE: Submitting statements to Python:


478       run;
NOTE: Procedure python step took :
      real time : 5.469
      cpu time  : 0.015


479
480
481       /*--- USE PYTHON 310 FOR CREATING SAS DATASETS ONLY ---*/
482
483       options validvarname=v7; /*--- very important pyhton is case sensitive ---*/
484       options set=PYTHONHOME "D:\py310";
485       proc python;
486       submit;
487       import pyarrow
488       import pandas as pd
489       # beam_inp = pd.read_parquet('d:/wpswrkx/beam_inp.parquet', engine='pyarrow')
490       results_df = pd.read_parquet('d:/wpswrkx/results_df.parquet', engine='pyarrow')
491       # print(beam_inp)
492       print(results_df)
493       endsubmit;

NOTE: Submitting statements to Python:


494       import python=results_df data=workx.results_df;
NOTE: Creating data set 'WORKX.results_df' from Python data frame 'results_df'
NOTE: Data set "WORKX.results_df" has 10 observation(s) and 13 variable(s)

495       run;quit;
NOTE: Procedure python step took :
      real time : 0.821
      cpu time  : 0.031


496
497       /*--- LIST OUTPUT ---*/
498
499       data workx.results_df_final;
500         label
501           name = "Beam material"
502           abs_max_deflection_mm = "Absolute max deflection (mm)"
503           L_m = "Beam length (m)"
504           b_m = "Cross-section width (m)"
505           h_m = "Cross-section height (m)"
506           E_GPa = "Young's modulus (GPa)"
507           I_m4 = "Moment of inertia (m4)"
508           EI_kNm2 = "Flexural rigidity (kNÂ·mÂ²)"
509           x_moment_m = "Moment application location (m)"
510           max_deflection_mm = "Maximum deflection (mm)"
511           max_deflection_loc_m = "Location of max deflection (m)"
512           max_deflection_analytical_mm = "Analytical max deflection (mm)"
513           max_slope_mrad = "Maximum slope (milliradians)"
514           max_moment_kNm = "Maximum bending moment (kNÂ·m)"
515          ;
516         set workx.results_df;
517           abs_max_deflection_mm = abs(max_deflection_mm);
518       run;

NOTE: 10 observations were read from "WORKX.results_df"
NOTE: Data set "WORKX.results_df_final" has 10 observation(s) and 14 variable(s)
NOTE: The data step took :
      real time : 0.016
      cpu time  : 0.015


519
520       proc sort data=workx.results_df_final;
521       by descending abs_max_deflection_mm;
522       run;quit;
NOTE: Automatically set SORTSIZE to 10240MiB
NOTE: 10 observations were read from "WORKX.results_df_final"
NOTE: Data set "WORKX.results_df_final" has 10 observation(s) and 14 variable(s)
NOTE: Procedure sort step took :
      real time : 0.016
      cpu time  : 0.015


523
524       proc print data=workx.results_df_final label;
525       Title "Ordered by Beam Material descending Absolute Max Deflection";
526       var name abs_max_deflection_mm L_m b_m h_m EI_kNm2;
527       run;quit;
NOTE: 10 observations were read from "WORKX.results_df_final"
NOTE: Procedure print step took :
      real time : 0.016
      cpu time  : 0.000


ERROR: Error printed on page 1

NOTE: Submitted statements took :
      real time : 6.432
      cpu time  : 0.156


/*              _
  ___ _ __   __| |
 / _ \ `_ \ / _` |
|  __/ | | | (_| |
 \___|_| |_|\__,_|

*/
