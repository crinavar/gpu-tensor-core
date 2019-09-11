#!/bin/bash
if [ "$#" -ne 10 ]; then
    echo "run as ./benchmark-blockconf.sh    DEV    STARTN ENDN DN      SEE   DIST   KREPEATS SAMPLES BINARY OUTFILE"
    exit;
fi
DEV=$1
STARTN=$2
ENDN=$3
DN=$4
SEED=$5
DIST=$6
REPEAT=$7
SAMPLES=$8
BINARY=${9}
OUTFILE=${10}
METHODS=("tc_block" "cub_16" "cub_32" "shuffle")
DISTRIBUTION=("normal" "uniform")
NM=$((${#METHODS[@]}-1))
TMEAN[0]=0
TVAR[0]=0
TSTDEV[0]=0
TSTERR[0]=0
#seed=12

echo "Benchmarking for B=256"
#echo "Compiling with BSIZE=256"
LB=256
cd ..
#COMPILE=`make BSIZE=${LB} R=1`

for N in `seq ${STARTN} ${DN} ${ENDN}`;
do
    #echo "DEV=${DEV}  N=${N} B=${B}"
    echo -n "${N}   ${B}    " >> data/${OUTFILE}_${DISTRIBUTION[$DIST]}.dat
    for q in `seq 0 ${NM}`;
    do
        #echo $N
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
        # Chosen MAP
        #echo "./${BINARY} ${DEV}    ${N} 0 ${SEED} ${REPEAT} ${DIST} 2"
        #echo "./${BINARY}_cub16 ${DEV} ${N} ${DIST} ${SEED} ${REPEAT}"
        #echo "./${BINARY}_cub32 ${DEV} ${N} ${DIST} ${SEED} ${REPEAT}"
        #echo "./${BINARY} ${DEV}    ${N} 0 ${SEED} ${REPEAT} ${DIST} 0"
        for k in `seq 1 ${SAMPLES}`;
        do
            case "$q" in
                #case 1
                "0") value=`./${BINARY} ${DEV}    ${N} 0 ${SEED} ${REPEAT} ${DIST} 2`;;
                #case 2
                "1") value=`./${BINARY}_cub16 ${DEV} ${N} ${DIST} ${SEED} ${REPEAT}`;;
                #case 3
                "2") value=`./${BINARY}_cub32 ${DEV} ${N} ${DIST} ${SEED} ${REPEAT}`;;
                #case 4
                "3") value=`./${BINARY} ${DEV}    ${N} 0 ${SEED} ${REPEAT} ${DIST} 0`;;
            esac
            #echo "$value"
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
        TMEAN[$q]=${MEAN}
        TVAR[$q]=${VAR}
        TSTDEV[$q]=${STDEV}
        TSTERR[$q]=${STERR}
        w2=$(echo "scale=10; $w1/$SAMPLES" | bc)
        y2=$(echo "scale=10; $y1/$SAMPLES" | bc)
        z2=$(echo "scale=10; $z1/$SAMPLES" | bc)
        v2=$(echo "scale=10; $v1/$SAMPLES" | bc)
        echo "---> B=${B} N=${N} --> (MEAN, VAR, STDEV, STERR, SUM, CPUSUM,
        #DIFF, %DIFF) -> (${TMEAN[$q]}[ms], ${TVAR[$q]}, ${TSTDEV[$q]}, ${TSTERR[$q]}, ${y2}, ${w2}, ${z2}, ${v2})"
        echo -n "${TMEAN[$q]} ${TVAR[$q]} ${TSTDEV[$q]} ${TSTERR[$q]} ${y} ${w} ${z} ${v}        " >> data/${OUTFILE}_${DISTRIBUTION[$DIST]}.dat
        echo " "
    done
    echo " " >> data/${OUTFILE}_${DISTRIBUTION[$DIST]}.dat
    echo " "
    echo " "
    echo " "
done 
echo " "

