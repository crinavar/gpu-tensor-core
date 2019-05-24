reset
set   autoscale                        # scale axes automatically
set output 'plots/reduction-plot-speedup-B256.eps'
set title 'Reduction Speedup, Titan V'
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set xlabel 'N'
set ylabel 'S'
plot    'data/data_B256.dat' using 1:2 title 'shuffle' with linespoints
