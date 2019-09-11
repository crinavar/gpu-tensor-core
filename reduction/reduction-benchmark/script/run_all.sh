#!/bin/bash
cd ..

FECHA_INICIO=$(date +%s)
./benchmark-all.sh 0 $((2**13)) $((2**24)) $((2**13)) 12 0 30 3 prog data_tintanRTX
FECHA_FIN=$(date +%s)
script_1=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_1 " >> data/time.dat


FECHA_INICIO=$(date +%s)
./benchmark-all.sh 0 $((2**13)) $((2**24)) $((2**13)) 12 1 30 3 prog data_titanRTX
FECHA_FIN=$(date +%s)
script_2=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_2 " >> data/time.dat


FECHA_INICIO=$(date +%s)
./benchmark-all.sh 1 $((2**13)) $((2**24)) $((2**13)) 12 0 30 3 prog data_titanV
FECHA_FIN=$(date +%s)
script_3=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_3 " >> data/time.dat


FECHA_INICIO=$(date +%s)
./benchmark-all.sh 1 $((2**13)) $((2**24)) $((2**13)) 12 1 30 3 prog data_titan_V
FECHA_FIN=$(date +%s)
script_4=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_4 " >> data/time.dat


echo "$script_1"
echo "$script_2"
echo "$script_3"
echo "$script_4"
