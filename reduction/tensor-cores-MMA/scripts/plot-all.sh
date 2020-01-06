#!/bin/bash
GPU=${1}

#for rconf and pconf, adjust the yranges manually
RECY1=0.26; RECY2=0.31
SINGLEY1=0.20; SINGLEY2=0.3
SPLITY1=0.2 SPLITY2=0.3

# (1) rconf and pconf
gnuplot -c scripts/plot-rconf.gp ${GPU} normal recurrence ${RECY1} ${RECY2}
gnuplot -c scripts/plot-rconf.gp ${GPU} normal singlepass ${SINGLEY1} ${SINGLEY2}
gnuplot -c scripts/plot-pconf.gp ${GPU} normal split ${SPLITY1} ${SPLITY2}



# (2) variants
gnuplot -c scripts/plot-variants-runtime.gp ${GPU} normal
gnuplot -c scripts/plot-variants-speedup.gp ${GPU} normal
gnuplot -c scripts/plot-variants-error.gp ${GPU} normal

gnuplot -c scripts/plot-variants-runtime.gp ${GPU} uniform
gnuplot -c scripts/plot-variants-speedup.gp ${GPU} uniform
gnuplot -c scripts/plot-variants-error.gp ${GPU} uniform
 
 

 
# (3) comparison with CUB
gnuplot -c scripts/plot-comparison-runtime.gp ${GPU} normal
gnuplot -c scripts/plot-comparison-speedup.gp ${GPU} normal
gnuplot -c scripts/plot-comparison-error.gp ${GPU} normal
gnuplot -c scripts/plot-comparison-beps.gp ${GPU} normal
# 
gnuplot -c scripts/plot-comparison-runtime.gp ${GPU} uniform
#gnuplot -c scripts/plot-comparison-speedup.gp ${GPU} uniform
#gnuplot -c scripts/plot-comparison-error.gp ${GPU} uniform
#gnuplot -c scripts/plot-comparison-beps.gp ${GPU} uniform
