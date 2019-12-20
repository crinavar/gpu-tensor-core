#!/bin/bash
if [ "$#" -ne 17 ]; then
    echo "run as ./benchmark-blockconf.sh DEV   N1 N2 DN    B1 B2 DB  ARCH R FS DIST SEED    KREPEATS SAMPLES BINARY ALG OUTFILE"
    exit;
fi
DEV=$1
STARTN=$2
ENDN=$3
DN=$4
STARTB=$5
ENDB=$6
DB=$7
ARCH=$8
R=$9
FS=${10}
DIST=${11}
SEED=${12}
REPEAT=${13}
SAMPLES=${14}
BINARY=${15}
ALG=${16}
OUTFILE=${17}
DISTRIBUTION=("Normal" "Uniform" "Constant")
ALGORITHMS=("warp-shuffle" "recurrence" "single-pass-R${R}" "split-FS${FS}" "recurrence-chained-R${R}")
TMEAN[0]=0
TVAR[0]=0
TSTDEV[0]=0
TSTERR[0]=0

for B in `seq ${STARTB} ${DB} ${ENDB}`;
do
    COMPILE="make BSIZE=${B} ARCH=${ARCH} R=${R}"
    echo "$COMPILE"
    C=`${COMPILE}`
    #C=`make BSIZE=$B ARCH=${ARCH} R=${R}`
    OUTPATH=data/${OUTFILE}_${ALGORITHMS[${ALG}]}_${DISTRIBUTION[$DIST]}_B${B}.dat
    echo -n "${N}  ${B}  ${R}  " >> ${OUTPATH}
    for N in `seq ${STARTN} ${DN} ${ENDN}`;
    do
        echo "[B=${B},R=${R},FS=${FS}]  ${N}"
        M=0, S=0, x=0, y=0, z=0, v=0, w1=0, y1=0, z1=0, v1=0, w2=0, y2=0, z2=0, v2=0
        for k in `seq 1 ${SAMPLES}`;
        do
            echo  "./${BINARY} ${DEV}    ${N} ${FS} ${REPEAT} ${SEED} ${DIST} ${ALG}"
            value=`./${BINARY} ${DEV}    ${N} ${FS} ${REPEAT} ${SEED} ${DIST} ${ALG}`
            echo "$value"
            x="$(cut -d',' -f1 <<<"$value")"
            y="$(cut -d',' -f2 <<<"$value")"
            w="$(cut -d',' -f3 <<<"$value")"
            z="$(cut -d',' -f4 <<<"$value")"
            v="$(cut -d',' -f5 <<<"$value")"
            w1=$(echo "scale=10; $w1+$w" | bc)
            y1=$(echo "scale=10; $y1+$y" | bc)
            z1=$(echo "scale=10; $z1+$z" | bc)
            v1=$(echo "scale=10; $v1+$v" | bc)
            oldM=$M;
            M=$(echo "scale=10;  $M+($x-$M)/$k"           | bc)
            S=$(echo "scale=10;  $S+($x-$M)*($x-${oldM})" | bc)
        done
        echo "done"
        MEAN=$M
        VAR=$(echo "scale=10; $S/(${SAMPLES}-1.0)"  | bc)
        STDEV=$(echo "scale=10; sqrt(${VAR})"       | bc)
        STERR=$(echo "scale=10; ${STDEV}/sqrt(${SAMPLES})" | bc)
        TMEAN[0]=${MEAN}
        TVAR[0]=${VAR}
        TSTDEV[0]=${STDEV}
        TSTERR[0]=${STERR}
        w2=$(echo "scale=10; $w1/$SAMPLES" | bc)
        y2=$(echo "scale=10; $y1/$SAMPLES" | bc)
        z2=$(echo "scale=10; $z1/$SAMPLES" | bc)
        v2=$(echo "scale=10; $v1/$SAMPLES" | bc)
        echo " "
        echo "---> (MEAN, VAR, STDEV, STERR, SUM, CPUSUM, #DIFF, %DIFF) -> (${TMEAN[0]}[ms], ${TVAR[0]}, ${TSTDEV[0]}, ${TSTERR[0]}, ${y2}, ${w2}, ${z2}, ${v2})"
        echo -n "${TMEAN[0]} ${TVAR[0]} ${TSTDEV[0]} ${TSTERR[0]} ${y} ${w} ${z} ${v}        " >> ${OUTPATH}
        echo " "
    done
    echo " " >> ${OUTPATH}
    echo " "
    echo " "
    echo " "
done 
echo " "
