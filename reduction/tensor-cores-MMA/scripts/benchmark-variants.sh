#!/bin/bash
if [ "$#" -ne 11 ]; then
    echo "run as ${0} DEV ARCH STARTN ENDN DN DIST SEED KREPEATS SAMPLES BINARY OUTFILE"
    exit;
fi
DEV=$1
ARCH=$2
STARTN=$3
ENDN=$4
DN=$5
DIST=$6
SEED=$7
KREPEATS=${8}
SAMPLES=${9}
BINARY=${10}
OUTFILE=${11}
METHODS=("warpshuffle" "recurrence" "singlepass" "split")
NM=$((${#METHODS[@]}-1))
R=("1" "4" "4" "1")
FS=("0" "0" "0" "0.5")
BSIZE=("1024" "128" "32" "32")
DISTRIBUTION=("Normal" "Uniform")
TMEAN[0]=0
TVAR[0]=0
TSTDEV[0]=0
TSTERR[0]=0

for i in 0 1 2 3;
do
    echo "[EXECUTE] scripts/benchmark-alg.sh ${DEV}     ${STARTN} ${ENDN} ${DN}      ${BSIZE[$i]} ${BSIZE[$i]} 1     ${ARCH} ${R[$i]} ${FS[$i]} ${DIST} ${SEED}        ${KREPEATS} ${SAMPLES} ${BINARY} ${i} ${OUTFILE}"
    scripts/benchmark-alg.sh ${DEV} ${STARTN} ${ENDN} ${DN} ${BSIZE[$i]} ${BSIZE[$i]} 1 ${ARCH}  ${R[$i]}  ${FS[$i]} ${DIST} ${SEED} ${KREPEATS} ${SAMPLES} ${BINARY} ${i} ${OUTFILE}
done
