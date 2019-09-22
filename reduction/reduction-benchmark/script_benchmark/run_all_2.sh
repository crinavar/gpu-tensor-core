#!/bin/bash
#cd ..

FECHA_INICIO=$(date +%s)
./benchmark-all.sh 1 $((2**13)) $((2**24)) $((2**13)) 12 0 40 4 prog data_titanV
FECHA_FIN=$(date +%s)
script_1=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_1 " >> data/time2.dat


FECHA_INICIO=$(date +%s)
./benchmark-all.sh 1 $((2**13)) $((2**24)) $((2**13)) 12 1 40 4 prog data_titanV
FECHA_FIN=$(date +%s)
script_2=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_2 " >> data/time2.dat

echo "$script_1"
echo "$script_2"
