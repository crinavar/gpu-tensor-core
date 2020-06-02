reset
if (ARGC != 3){
    print "run as : gnuplot -c  GPU-MODEL  CPU-MODEL DISTRIBUTION"
    exit
}

cpu  = ARG1
gpu  = ARG2
dist = ARG3

print "plot-power.gp\n CPU = ",cpu,"\n GPU = ",gpu,"\n dist: ",dist
out = 'plots/power-'.cpu.'-'.gpu.'-'.dist.'.eps'
mytitle = "Power Consumption (".gpu." | ".cpu.")\nn = 400M,  repeats = 1000"

set autoscale                        # scale axes automatically
set term postscript eps color blacktext "Courier" 24
set output out
set title mytitle

set ytics mirror
set ylabel 'W' rotate by 0 offset 1
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set xrange [0:46]
#set log x

set xlabel 'time [s]'
set font "Courier, 20"
set pointsize   0.5
#set xtics format "%1.0s"
set key right top Left  font "Courier, 18"

set style line 1 lt 1 lc rgb 'forest-green' dt "-.-"    pt 5    pi -6   lw 2 # green   
set style line 2 lt 2 lc rgb 'black'        dt 1        pt 2    pi -6   lw 2 # orange
set style line 3 lt 3 lc rgb 'web-blue'     dt "."      pt 6    pi -6   lw 2 # blue
set style line 4 lt 4 lc rgb 'red'          dt 5        pt 11   pi -6   lw 2 # purple

# variables
single_pass_data    = 'data/power-'.gpu.'-single-pass.dat'
cub16_data          = 'data/power-'.gpu.'-CUB-half.dat'
cub32_data          = 'data/power-'.gpu.'-CUB-float.dat'
omp_data            = 'data/power-'.cpu.'-omp-nt10-double.dat'

#print "warp_shuffle_data: ".warp_shuffle_data
#print "split_data: ".split_data
#print "recurrence_data: ".recurrence_data
#print "single_pass_data: ".single_pass_data

plot\
        cub16_data          using ($1/100):2 title "CUB (half)"             with l   ls 3,\
        cub32_data          using ($1/105):2 title "CUB (float)"            with l   ls 1,\
        omp_data            using (($1-98)/100):2 title "OpenMP nt=10"     with l   ls 4,\
        single_pass_data    using (($1-118)/100):2 title "single-pass"      with l   ls 2

print "done!\n\n"
