reset
gpu  = ARG1
dist = ARG2

print "GPU: Titan ",gpu," dist: ",dist

out = '../plots/plot_split_Titan'.gpu.'_'.dist.'.eps'
title = "Runtime Mixed Reduction\n".dist." Distribution\nTitan ".gpu 

set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Courier" 24
set output out
set title title

set ytics mirror
unset ytics
#set xtics (1024, 8192, 16384, 24576, 321024)
#set y2tics 0.2
#set link y2
set ylabel 'Time (Second)' rotate by 90 offset -1

set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically

set xlabel '% Tensor Cores'
set font "Courier, 20"
#set pointsize   0.5
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically

set key Left top left reverse samplen 3.0 font "Courier,18" spacing 1 

set style line 1 dashtype 1 pt 7 lw 1.0 lc rgb "#2d905d"
set style line 2 dashtype 1 pt 9 lw 1.0 lc rgb "magenta"
set style line 3 dashtype 1 pt 5 lw 1.0 lc rgb "red"
set style line 4 dashtype 1 pt 2 lw 1.0 lc rgb "#2271b3"
set style line 5 dashtype 1 pt 1 lw 1.0 lc rgb "#228B22"
set style line 6 dashtype 1 pt 6 lw 1.0 lc rgb "black"
set style line 7 dashtype 1 pt 3 lw 1.0 lc rgb "#1E90FF"
set key right top Left title 'Method' font "Courier, 20"
set log y

plot    '../data/pconf_titan'.gpu.'_'.dist.'.dat' using 2:3 title "B32" with line ls 1,\
        '../data/pconf_titan'.gpu.'_'.dist.'.dat' using 2:11 title "B128" with line ls 2,\
        '../data/pconf_titan'.gpu.'_'.dist.'.dat' using 2:19 title "B512" with line ls 3,\
        '../data/pconf_titan'.gpu.'_'.dist.'.dat' using 2:27 title "B1024" with line ls 4



