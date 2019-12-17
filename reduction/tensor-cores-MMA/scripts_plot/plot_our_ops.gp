reset
gpu  = ARG1
dist = ARG2

print "GPU: Titan ",gpu," dist: ",dist

out = '../plots/plot_our_elem_Titan'.gpu.'_'.dist.'.eps'
title = "Billons elements per Second\n".dist." Distribution\nTitan ".gpu 

set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Courier" 24
set output out
set title title

set ytics mirror
unset ytics
#set xtics (1024, 8192, 16384, 24576, 321024)
#set y2tics 0.2
#set link y2
set ylabel 'Operation per second' rotate by 90 offset -1

set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically

set xlabel 'N x 10^{6} Datasize'
set font "Courier, 20"
set pointsize   0.5
set xtics format "%1.0s"
#set xtics (8192 ,8388608, 16777216)
set key Left top left reverse samplen 3.0 font "Courier,18" spacing 1 

set style line 1 dashtype 1 pt 7 lw 1.0 lc rgb "#2d905d"
set style line 2 dashtype 1 pt 9 lw 1.0 lc rgb "magenta"
set style line 3 dashtype 1 pt 5 lw 1.0 lc rgb "#1E90FF"
set style line 4 dashtype 1 pt 2 lw 1.0 lc rgb "red"
set style line 5 dashtype 1 pt 1 lw 1.0 lc rgb "yellow"
set style line 6 dashtype 1 pt 6 lw 1.0 lc rgb "black"
set style line 7 dashtype 1 pt 3 lw 1.0 lc rgb "green"

set key right bottom Left title 'Methods:' font "Courier,20"
#set log y

plot    '../data/our_titan'.gpu.'_'.dist.'.dat' using 1:($1/$2) title "Shuffle" with line ls 1,\
        '../data/our_titan'.gpu.'_'.dist.'.dat' using 1:($10 <= 0 ? 1/0 : $1/$10 ) title "Theory" with line ls 2,\
        '../data/our_titan'.gpu.'_'.dist.'.dat' using 1:($1/$18) title "Tensor-Shuffle" with line ls 3,\
        '../data/our_titan'.gpu.'_'.dist.'.dat' using 1:($1/$26) title "Mixed" with line ls 4,\
        '../data/our_titan'.gpu.'_'.dist.'.dat' using 1:($1/$34) title "Chain R8" with line ls 5



