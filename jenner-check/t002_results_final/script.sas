/* Final "LIST OUTPUT" stage of the repo's main program, run standalone.

   In the original, results_df is produced by the proc-python analysis
   step (numpy/scipy under Altair SLC) and read back into SAS. Here we
   stand in a small fixed results_df built from the deflection values the
   repo itself documents, so the downstream SAS logic - the abs() derived
   column, the descending sort, and the labeled proc print - runs on its
   own. That block (data results_df_final / proc sort / proc print) is
   the author's verbatim; only the results_df seed is substituted. */

options validvarname=v7;

/* Stand-in for the Python-produced results_df (13 vars); the deflection
   magnitudes are the repo's own documented output values. */
data results_df;
  length name $15;
  input name & max_deflection_mm L_m E_GPa b_m h_m EI_kNm2;
  x_moment_m = L_m/2;
  I_m4 = b_m*h_m**3/12;
  max_deflection_loc_m = .;
  max_deflection_analytical_mm = .;
  max_slope_mrad = .;
  max_moment_kNm = .;
cards4;
Steel Beam A  -0.15035 5.0 200 0.10 0.20 13333.33
Steel Beam B  -0.04106 4.0 200 0.12 0.25 31250.00
Aluminum Beam  0.14347 3.0 69 0.15 0.18 5030.10
Wood Beam  0.36084 4.5 10 0.20 0.30 4500.00
Steel Beam C  0.64150 6.0 200 0.08 0.15 4500.00
Steel Beam D  -0.04911 3.5 200 0.15 0.20 20000.00
Copper Beam  0.16199 2.5 110 0.10 0.15 3093.75
Steel Beam E  -0.11390 5.5 200 0.12 0.22 21296.00
Aluminum Beam 2  0.15495 4.0 69 0.18 0.20 8280.00
Composite Beam  -0.06757 4.8 150 0.14 0.25 27343.75
;;;;
run;

/*--- LIST OUTPUT ---*/

data results_df_final;
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
  set results_df;
    abs_max_deflection_mm = abs(max_deflection_mm);
run;

proc sort data=results_df_final;
by descending abs_max_deflection_mm;
run;quit;

proc print data=results_df_final label;
Title "Ordered by Beam Material descending Absolute Max Deflection";
var name abs_max_deflection_mm L_m b_m h_m EI_kNm2;
run;quit;
