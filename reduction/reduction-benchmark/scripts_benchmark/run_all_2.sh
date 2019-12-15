#!/bin/bash
#cd ..

cd /home/rcarrasco/tensor/gpu-tensor-core/reduction/reduction-benchmark/script_benchmark
FECHA_INICIO=$(date +%s)
./benchmark-block.sh 0 $((2**17)) $((2**27)) $((2**17)) 0 127 10 3 prog block_titanRTX
FECHA_FIN=$(date +%s)
script_1=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_1 " >> ../data/time2.dat


cd /home/rcarrasco/tensor/gpu-tensor-core/reduction/reduction-benchmark/script_benchmark
FECHA_INICIO=$(date +%s)
./benchmark-block.sh 0 $((2**17)) $((2**27)) $((2**17)) 1 127 10 3 prog block_titanRTX
FECHA_FIN=$(date +%s)
script_2=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_2 " >> ../data/time2.dat

cd /home/rcarrasco/tensor/gpu-tensor-core/reduction/reduction-benchmark/script_benchmark
FECHA_INICIO=$(date +%s)
./benchmark-theory.sh 0 $((2**13)) $((2**19)) $((2**13)) 0 127 10 3 prog theory_titanRTX
FECHA_FIN=$(date +%s)
script_1=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_1 " >> ../data/time2.dat

cd /home/rcarrasco/tensor/gpu-tensor-core/reduction/reduction-benchmark/script_benchmark
FECHA_INICIO=$(date +%s)
./benchmark-theory.sh 0 $((2**13)) $((2**19)) $((2**13)) 1 127 10 3 prog theory_titanRTX
FECHA_FIN=$(date +%s)
script_1=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_1 " >> ../data/time2.dat

cd /home/rcarrasco/tensor/gpu-tensor-core/reduction/reduction-benchmark/script_benchmark

FECHA_INICIO=$(date +%s)
./benchmark-our.sh 0 $((2**17)) $((2**27)) $((2**17)) 0 127 10 3 prog our_titanRTX 
FECHA_FIN=$(date +%s)
script_time=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_time " >> ../data/time2.dat

cd /home/rcarrasco/tensor/gpu-tensor-core/reduction/reduction-benchmark/script_benchmark

FECHA_INICIO=$(date +%s)
./benchmark-our.sh 0 $((2**17)) $((2**27)) $((2**17)) 1 127 10 3 prog our_titanRTX
FECHA_FIN=$(date +%s)
script_time=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_time " >> ../data/time2.dat

cd /home/rcarrasco/tensor/gpu-tensor-core/reduction/reduction-benchmark/script_benchmark
FECHA_INICIO=$(date +%s)
./benchmark-all.sh 0 123469824 $((2**27)) $((2**17)) 127 0 10 3 prog cub_titanRTX
FECHA_FIN=$(date +%s)
script_1=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_1 " >> ../data/time2.dat


cd /home/rcarrasco/tensor/gpu-tensor-core/reduction/reduction-benchmark/script_benchmark
FECHA_INICIO=$(date +%s)
./benchmark-all.sh 0 $((2**17)) $((2**27)) $((2**17)) 127 1 10 3 prog cub_titanRTX
FECHA_FIN=$(date +%s)
script_2=$(( $FECHA_FIN - $FECHA_INICIO ))
echo "$script_2 " >> ../data/time2.dat

echo ""
echo ""
cat ../data/time2.dat


