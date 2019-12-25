#!/bin/bash

gnuplot -c plot_our_time.gp V Normal
gnuplot -c plot_our_time.gp V Uniform
gnuplot -c plot_block.gp V Normal
gnuplot -c plot_block.gp V Uniform
gnuplot -c plot_chained.gp V Normal
gnuplot -c plot_chained.gp V Uniform
gnuplot -c plot_split.gp V Normal
gnuplot -c plot_split.gp V Uniform
gnuplot -c plot_theory.gp V Normal
gnuplot -c plot_theory.gp V Uniform

gnuplot -c plot_our_error.gp V Normal
gnuplot -c plot_our_error.gp V Uniform
gnuplot -c plot_our_ops.gp V Normal
gnuplot -c plot_our_ops.gp V Uniform