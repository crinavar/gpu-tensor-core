reset
set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Courier" 24
set output '../plots/runtime_TitanRTX_uniform.eps'
set title "Runtime Reduction Arithmetic\nUniform Distribution\nTitan RTX"

set ytics mirror
unset ytics
#set xtics (1024, 8192, 16384, 24576, 321024)
#set y2tics 0.2
#set link y2
set ylabel 'Time [s]' rotate by 90 offset -1

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
set style line 3 dashtype 1 pt 5 lw 1.0 lc rgb "red"
set style line 4 dashtype 1 pt 2 lw 1.0 lc rgb "#2271b3"
set style line 5 dashtype 1 pt 1 lw 1.0 lc rgb "#228B22"
set style line 6 dashtype 1 pt 6 lw 1.0 lc rgb "black"
set style line 7 dashtype 1 pt 3 lw 1.0 lc rgb "#1E90FF"

set key left top Left title 'Methods:' font "Courier,20"
#set log y

plot    '../data/data_titanRTX_uniform.dat' using 1:2 title "TC-Block" with line ls 3,\
        '../data/data_titanRTX_uniform.dat' using 1:10 title "CUB16" with line ls 4,\
        '../data/data_titanRTX_uniform.dat' using 1:18 title "CUB32" with line ls 5,\
        '../data/data_titanRTX_uniform.dat' using 1:26 title "Shuffle" with line ls 6



