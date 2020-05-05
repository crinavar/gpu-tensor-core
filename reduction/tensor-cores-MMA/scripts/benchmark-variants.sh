#!/bin/bash
if [ "$#" -ne 12 ]; then
    echo "run as ${0} DEV NUMTHREADS ARCH STARTN ENDN DN DIST SEED KREPEATS SAMPLES BINARY OUTFILE"
    exit;
fi
DEV=$1
OMPTHREADS=$2
ARCH=$3
STARTN=$4
ENDN=$5
DN=$6
DIST=$7
SEED=$8
KREPEATS=${9}
SAMPLES=${10}
BINARY=${11}
OUTFILE=${12}
METHODS=("warpshuffle" "recurrence" "singlepass" "split" "omp-float" "omp-double")
NM=$((${#METHODS[@]}-1))

# these values are for TITAN-RTX
#R=("1" "4" "4" "1" "1" "1")
#FS=("0" "0" "0" "0.5" "0" "0")
#BSIZE=("1024" "128" "32" "32" "-1" "-1")

# these values are for TESLA-V100
R=("1" "5" "4" "1" "1" "1")
FS=("0" "0" "0" "0.5" "0" "0")
BSIZE=("1024" "32" "128" "512" "-1" "-1")

DISTRIBUTION=("normal" "uniform")

#for i in 4 5;
for i in {0..${NM}};
do
    echo "[EXECUTE] scripts/benchmark-alg.sh ${DEV} ${OMPTHREADS}    ${STARTN} ${ENDN} ${DN}     ${BSIZE[$i]} ${BSIZE[$i]} 1     ${ARCH} ${R[$i]} ${FS[$i]} ${DIST} ${SEED}      ${KREPEATS} ${SAMPLES} ${BINARY} ${i} ${OUTFILE}"
    scripts/benchmark-alg.sh                 ${DEV} ${OMPTHREADS}    ${STARTN} ${ENDN} ${DN}     ${BSIZE[$i]} ${BSIZE[$i]} 1     ${ARCH} ${R[$i]} ${FS[$i]} ${DIST} ${SEED}      ${KREPEATS} ${SAMPLES} ${BINARY} ${i} ${OUTFILE}
done
