reset

gpu  = ARG1
dist = ARG2

print "plot-comparison-runtime.gp ---> GPU: ",gpu," dist: ",dist

out     = 'plots/comparison-runtime-'.gpu.'-'.dist.'.eps'
mytitle = "Runtime vs CUB Library (".gpu.")\n".dist." Distribution\n "

set autoscale # scale axes automatically
set term postscript eps color blacktext "Courier" 24
set output out
set title mytitle

set ylabel 'Time [ms]' rotate by 90 offset 0
set log y
#set format y "%1.0s*10^{%T}"
#set yrange [0.01:1.0]
#set ytics (0.01, 0.1, 0.35, 0.7)

set xlabel 'n x 10^{6}'
set font "Courier, 20"
set pointsize   0.5
set xtics format "%1.0s"

set style line 1 lt 1 lc rgb 'forest-green' dt 1    pt 5    pi -6   lw 2 # green   
set style line 2 lt 2 lc rgb 'black'       dt 2    pt 2    pi -6   lw 2 # orange
set style line 3 lt 3 lc rgb 'web-blue'     dt 6    pt 6    pi -6   lw 2 # blue
set style line 4 lt 4 lc rgb 'red'          dt 5    pt 11   pi -6   lw 2 # purple
set style line 5 lt 1 lc rgb '#77ac30'              pt 13   pi -6   lw 2 # green
set style line 6 lt 1 lc rgb '#4dbeee'              pt 4    pi -6   lw 2 # light-blue
set style line 7 lt 1 lc rgb '#a2142f'              pt 8    pi -6   lw 2 # red

set key Left bot right reverse samplen 3.0 font "Courier,18" spacing 1 

singlepass = 'data/alg-singlepass-'.gpu.'-'.dist.'-B128.dat'
cub16 = 'data/alg-CUB-FP16-'.gpu.'-'.dist.'.dat'
cub32 = 'data/alg-CUB-FP32-'.gpu.'-'.dist.'.dat'

plot    cub32       using 1:2 title "CUB (float)"    with lp ls 2,\
        cub16       using 1:2 title "CUB (half)"   with lp ls 3,\
	singlepass  using 1:5 title "single-pass"   with lp ls 1
