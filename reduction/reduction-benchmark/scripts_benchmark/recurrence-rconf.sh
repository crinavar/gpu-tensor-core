#!/bin/bash
if [ "$#" -ne 9 ]; then
    echo "run as ./benchmark-blockconf.sh    DEV  ARCH  N  SEED   DIST   KREPEATS SAMPLES BINARY OUTFILE"
    exit;
fi
DEV=$1
ARCH=$2
N=$3
SEED=$4
DIST=$5
REPEAT=$6
SAMPLES=$7
BINARY=${8}
OUTFILE=${9}
DISTRIBUTION=("normal" "uniform")
TMEAN[0]=0
TVAR[0]=0
TSTDEV[0]=0
TSTERR[0]=0
STARTB=32
DB=32
ENDDB=1024
ALG=4
for B in 32 128 512 1024;
do
    MYPATH=data/recurrence-rconf-${OUTFILE}_${DISTRIBUTION[$DIST]}_B${B}.dat
    for R in {1..128}
    do
        echo "Compiling with BSIZE=$B   R=${R}"
        COMPILE=`make BSIZE=${B} R=${R} ARCH=${ARCH}`
        echo ${COMPILE}
        #B=$((32*$i))
        echo -n "${N}  ${B}  ${R}  " >> ${MYPATH}
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
            echo "[BSIZE=$B, R=${R}, SAMPLE $k] ./${BINARY} ${DEV} ${N} 0 ${SEED} ${REPEAT} ${DIST} ${ALG}"
            value=`./${BINARY} ${DEV}    ${N} 0 ${SEED} ${REPEAT} ${DIST} ${ALG}`
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
        echo "---> B=${B} N=${N} R=${R} --> (MEAN, VAR, STDEV, STERR, SUM, CPUSUM, DIFF, %DIFF) -> (${TMEAN[$q]}[ms], ${TVAR[$q]}, ${TSTDEV[$q]}, ${TSTERR[$q]}, ${y2}, ${w2}, ${z2}, ${v2})"
        echo -n "${TMEAN[$q]} ${TVAR[$q]} ${TSTDEV[$q]} ${TSTERR[$q]} ${y} ${w} ${z} ${v}        " >> ${MYPATH} 
        echo " "
        echo " " >> ${MYPATH}
        echo " "
        echo " "
        echo " "
    done
done 
