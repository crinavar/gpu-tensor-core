reset

gpu  = ARG1
dist = ARG2

print "GPU: Titan ",gpu," dist: ",dist

out = '../plots/plot_cub_error_Titan'.gpu.'_'.dist.'.eps'
title = "Error Tensor Reduction (Titan ".gpu.")\n".dist." Distribution"

set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Courier" 24
set output out
set title title

set ytics mirror
unset ytics
#set xtics (1024, 8192, 16384, 24576, 321024)
#set y2tics 0.2
#set link y2
set ylabel '% Error' rotate by 90 offset -1

set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically

set xlabel 'N x 10^{6} Datasize'
set font "Courier, 20"
set pointsize   0.5
set xtics format "%1.0s"
#set xtics (8192 ,8388608, 16777216)
set key Left top left reverse samplen 3.0 font "Courier,18" spacing 1 

set style line 1 lt 1 lc rgb '#0072bd' pt 5  pi -4 lw 2 ps 1 # blue   
set style line 2 lt 1 lc rgb '#d95319' pt 7  pi -4 lw 2 ps 1 # orange
set style line 7 lt 1 lc rgb '#edb120' pt 6  pi -4 lw 2 ps 1 # yellow
set style line 5 lt 1 lc rgb '#7e2f8e' pt 11 pi -4 lw 2 ps 1 # purple
set style line 4 lt 1 lc rgb '#77ac30' pt 13 pi -4 lw 2 ps 1 # green
set style line 6 lt 1 lc rgb '#4dbeee' pt 4  pi -4 lw 2 ps 1 # light-blue
set style line 3 lt 1 lc rgb '#a2142f' pt 8  pi -4 lw 2 ps 1 # red

set key left top Left title 'Methods:' font "Courier,20"
#set log y

plot    '../data/cub_titan'.gpu.'_'.dist.'.dat' using 1:9 title "TC-Block" with lp ls 3,\
        '../data/cub_titan'.gpu.'_'.dist.'.dat' using 1:17 title "CUB16" with lp ls 6,\
        '../data/cub_titan'.gpu.'_'.dist.'.dat' using 1:25 title "CUB32" with lp ls 7


