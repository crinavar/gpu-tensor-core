#!/bin/bash
if [ "$#" -ne 8 ]; then
    echo "run as ./benchmark-blockconf.sh    DEV    N  SEED   DIST   KREPEATS SAMPLES BINARY OUTFILE"
    exit;
fi
DEV=$1
N=$2
SEED=$3
DIST=$4
REPEAT=$5
SAMPLES=$6
BINARY=${7}
OUTFILE=${8}
METHODS=("tc_block" "cub_16" "cub_32" "shuffle")
DISTRIBUTION=("Normal" "Uniform")
NM=$((${#METHODS[@]}-1))
TMEAN[0]=0
TVAR[0]=0
TSTDEV[0]=0
TSTERR[0]=0

cd ../src
#for B in `seq ${STARTB} ${DB} ${ENDB}`;
for P in `seq 0 0.125 1`;
do
    #B=$((32*$i))
    echo -n "${N}   ${P}    " >> ../data/${OUTFILE}_${DISTRIBUTION[$DIST]}.dat
    for B in 32 128 512 1024;
    do
        #B=$((32*$i))
        echo "Compiling with BSIZE=$B and R=1"
        COMPILE=`make BSIZE=${B} R=1`
        echo ${COMPILE}
        M=0
        S=0
        x=0
        y=0
        z=0
        v=0
        w1=0
        y1=0
        z1=0
        v1=0
        w2=0
        y2=0
        z2=0
        v2=0
        for k in `seq 1 ${SAMPLES}`;
        do
            value=`./${BINARY} ${DEV}    ${N} $P ${SEED} ${REPEAT} ${DIST} 3`
            x="$(cut -d',' -f1 <<<"$value")"
            y="$(cut -d',' -f2 <<<"$value")"
            w="$(cut -d',' -f3 <<<"$value")"
            z="$(cut -d',' -f4 <<<"$value")"
            v="$(cut -d',' -f5 <<<"$value")"
            #echo "$value"
            #echo "$x"
            w1=$(echo "scale=10; $w1+$w" | bc)
            y1=$(echo "scale=10; $y1+$y" | bc)
            z1=$(echo "scale=10; $z1+$z" | bc)
            v1=$(echo "scale=10; $v1+$v" | bc)
            oldM=$M;
            #echo "hola ${x}"
            M=$(echo "scale=10;  $M+($x-$M)/$k"           | bc)
            S=$(echo "scale=10;  $S+($x-$M)*($x-${oldM})" | bc)
        done
        echo "done"
        MEAN=$M
        VAR=$(echo "scale=10; $S/(${SAMPLES}-1.0)"  | bc)
        STDEV=$(echo "scale=10; sqrt(${VAR})"       | bc)
        STERR=$(echo "scale=10; ${STDEV}/sqrt(${SAMPLES})" | bc)
        TMEAN[$q]=${MEAN}
        TVAR[$q]=${VAR}
        TSTDEV[$q]=${STDEV}
        TSTERR[$q]=${STERR}
        w2=$(echo "scale=10; $w1/$SAMPLES" | bc)
        y2=$(echo "scale=10; $y1/$SAMPLES" | bc)
        z2=$(echo "scale=10; $z1/$SAMPLES" | bc)
        v2=$(echo "scale=10; $v1/$SAMPLES" | bc)
        echo "---> B=${B} N=${N} P=${P} --> (MEAN, VAR, STDEV, STERR, SUM, CPUSUM, DIFF, %DIFF) -> (${TMEAN[$q]}[ms], ${TVAR[$q]}, ${TSTDEV[$q]}, ${TSTERR[$q]}, ${y2}, ${w2}, ${z2}, ${v2})"
        echo -n "${TMEAN[$q]} ${TVAR[$q]} ${TSTDEV[$q]} ${TSTERR[$q]} ${y} ${w} ${z} ${v}        " >> ../data/${OUTFILE}_${DISTRIBUTION[$DIST]}.dat
        echo " "
    done
    echo " " >> ../data/${OUTFILE}_${DISTRIBUTION[$DIST]}.dat
    echo " "
    echo " "
    echo " "
done 



