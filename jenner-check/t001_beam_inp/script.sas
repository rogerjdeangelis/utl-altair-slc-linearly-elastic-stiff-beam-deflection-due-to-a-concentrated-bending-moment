/* Input beam table from the repo's main program, run standalone.
   The only change from the original is the library: workx (an autoexec
   libname pointing at d:/wpswrkx) is dropped so the table lands in WORK. */

options validvarname=v7 ls=255 ps=255;
data beam_inp;
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

proc print data=beam_inp label;
run;
