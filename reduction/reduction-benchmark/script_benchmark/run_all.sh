#!/bin/bash

DEV=$1
GPU=("titanRTX" "titanV")

cd /home/rcarrasco/tensor/gpu-tensor-core/reduction/reduction-benchmark/script_benchmark

FECHA_INICIO=$(date +%s)
./pconf.sh ${DEV} 92252160 127 0 10 3 prog pconf_${GPU[${DEV}]}
FECHA_FIN=$(date +%s)
script_time=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_time " >> ../data/time.dat

cd /home/rcarrasco/tensor/gpu-tensor-core/reduction/reduction-benchmark/script_benchmark

FECHA_INICIO=$(date +%s)
./pconf.sh ${DEV} 92252160 127 1 10 3 prog pconf_${GPU[${DEV}]}
FECHA_FIN=$(date +%s)
script_time=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_time " >> ../data/time.dat


cd /home/rcarrasco/tensor/gpu-tensor-core/reduction/reduction-benchmark/script_benchmark

FECHA_INICIO=$(date +%s)
./rconf.sh ${DEV} 92252160 127 0 10 3 prog rconf_${GPU[${DEV}]}
FECHA_FIN=$(date +%s)
script_time=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_time " >> ../data/time.dat

cd /home/rcarrasco/tensor/gpu-tensor-core/reduction/reduction-benchmark/script_benchmark

FECHA_INICIO=$(date +%s)
./rconf.sh ${DEV} 92252160 127 1 10 3 prog rconf_${GPU[${DEV}]}
FECHA_FIN=$(date +%s)
script_time=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_time " >> ../data/time.dat

cd /home/rcarrasco/tensor/gpu-tensor-core/reduction/reduction-benchmark/script_benchmark

FECHA_INICIO=$(date +%s)
./benchmark-theory.sh ${DEV} $((2**13)) $((2**19)) $((2**13)) 0 127 10 3 prog theory_${GPU[${DEV}]}
FECHA_FIN=$(date +%s)
script_time=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_time " >> ../data/time.dat

cd /home/rcarrasco/tensor/gpu-tensor-core/reduction/reduction-benchmark/script_benchmark

FECHA_INICIO=$(date +%s)
./benchmark-theory.sh ${DEV} $((2**13)) $((2**19)) $((2**13)) 1 127 10 3 prog theory_${GPU[${DEV}]}
FECHA_FIN=$(date +%s)
script_time=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_time " >> ../data/time.dat

cd /home/rcarrasco/tensor/gpu-tensor-core/reduction/reduction-benchmark/script_benchmark

FECHA_INICIO=$(date +%s)
./benchmark-block.sh ${DEV} $((2**19)) $((2**27)) $((2**19)) 0 127 10 3 prog block_${GPU[${DEV}]}
FECHA_FIN=$(date +%s)
script_time=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_time " >> ../data/time.dat

cd /home/rcarrasco/tensor/gpu-tensor-core/reduction/reduction-benchmark/script_benchmark

FECHA_INICIO=$(date +%s)
./benchmark-block.sh ${DEV} $((2**19)) $((2**27)) $((2**19)) 1 127 10 3 prog block_${GPU[${DEV}]}
FECHA_FIN=$(date +%s)
script_time=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_time " >> ../data/time.dat

cd /home/rcarrasco/tensor/gpu-tensor-core/reduction/reduction-benchmark/script_benchmark

FECHA_INICIO=$(date +%s)
./benchmark-our.sh ${DEV} $((2**19)) $((2**27)) $((2**19)) 0 127 10 3 prog our_${GPU[${DEV}]} 
FECHA_FIN=$(date +%s)
script_time=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_time " >> ../data/time.dat

cd /home/rcarrasco/tensor/gpu-tensor-core/reduction/reduction-benchmark/script_benchmark

FECHA_INICIO=$(date +%s)
./benchmark-our.sh ${DEV} $((2**19)) $((2**27)) $((2**19)) 1 127 10 3 prog our_${GPU[${DEV}]} 
FECHA_FIN=$(date +%s)
script_time=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_time " >> ../data/time.dat

