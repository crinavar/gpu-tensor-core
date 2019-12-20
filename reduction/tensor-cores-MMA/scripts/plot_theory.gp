reset

#if (!exists(ARG1) || !exists(ARG2)){
#    print "run as : gnuplot -c plot_pur_time   GPU    DISTRIBUTION"
#    exit
#}

gpu  = ARG1
dist = ARG2

print "GPU: Titan ",gpu," dist: ",dist

out = '../plots/plot_theory_Titan'.gpu.'_'.dist.'.eps'
title = "Runtime Theory Reduction\n".dist." Distribution\nTitan ".gpu 

set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Courier" 24
set output out
set title title

set ytics mirror
unset ytics
#set xtics (1024, 8192, 16384, 24576, 321024)
#set y2tics 0.2
#set link y2
set ylabel 'Time (Seconds)' rotate by 90 offset -1

set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically

set xlabel 'N x 10^{3} Datasize'
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

set key left top Left title 'Methods:' font "Courier,20"
#set log y

plot    '../data/theory_titan'.gpu.'_'.dist.'.dat' using 1:2 title "B32" with line ls 1,\
        '../data/theory_titan'.gpu.'_'.dist.'.dat' using 1:10 title "B128" with line ls 2,\
        '../data/theory_titan'.gpu.'_'.dist.'.dat' using 1:18 title "B256" with line ls 3,\
        '../data/theory_titan'.gpu.'_'.dist.'.dat' using 1:26 title "B512" with line ls 4,\
        '../data/theory_titan'.gpu.'_'.dist.'.dat' using 1:34 title "B1024" with line ls 6



