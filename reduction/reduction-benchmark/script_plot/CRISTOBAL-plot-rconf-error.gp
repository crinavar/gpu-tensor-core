reset
gpu  = ARG1
dist = ARG2

print "GPU: Titan ",gpu," dist: ",dist

out = '../plots/JOURNAL-rconf-chained-error-'.gpu.'_'.dist.'.eps'
title = "Error Fraction (".gpu.")\nChained MMAs, N {/Symbol \273} 1.4Bi, ".dist." dist." 

set autoscale                        # scale axes automatically
set term postscript eps color blacktext "Courier" 24
set output out
set title title

#set log y
#set yrange [4.8:4.9]
set ytics mirror
set xtics (0, 4, 16, 32, 48, 64, 80, 96, 112, 128)
set ylabel 'Error [0,1]' rotate by 90 offset -1
set xlabel '#R'
set font "Courier, 20"

set key Left top left reverse samplen 3.0 font "Courier,18" spacing 1 
set style line 1 lt 1 lc rgb 'black' dt 1 pt 5  pi -6 lw 2 ps 1 # blue   
set style line 2 lt 1 lc rgb '#d95319' dt 2 pt 7  pi -6 lw 2 ps 1 # orange
set style line 7 lt 1 lc rgb '#edb120' dt 3 pt 6  pi -6 lw 2 ps 1 # yellow
set style line 4 lt 1 lc rgb '#7e2f8e' dt 4 pt 11 pi -6 lw 2 ps 1 # purple
set style line 5 lt 1 lc rgb '#77ac30' pt 13 pi -6 lw 2 ps 1 # green
set style line 6 lt 1 lc rgb '#4dbeee' pt 4  pi -6 lw 2 ps 1 # light-blue
set style line 3 lt 1 lc rgb '#a2142f' pt 8  pi -6 lw 2 ps 1 # red

set key right top Left font "Courier, 20"

plot    '../data/CRISTOBAL-rconf-'.gpu.'_'.dist.'_B32.dat' using 3:11 title "B32, B128, B512, B1024" with lines ls 1,\



