#!/bin/bash
if [ "$#" -ne 10 ]; then
    echo "run as ./benchmark-blockconf.sh DEV STARTN ENDN DN  DIST SEED  KREPEATS SAMPLES BINARY OUTFILE"
    exit;
fi
DEV=$1
STARTN=$2
ENDN=$3
DN=$4
DIST=$5
REPEAT=${7}
SAMPLES=${8}
BINARY=${9}
OUTFILE=${10}
METHODS=("shuffle" "tc_theory" "tc_blockshuffle" "tc_mixed" "tc_chain_8")
NM=$((${#METHODS[@]}-1))
met=("0" "1" "2" "3" "2")
chain=("1" "1" "1" "1" "4")
ns=("0" "0" "0" "0.5" "0")
BSIZE=("1024" "1024" "256" "512" "32")
TMEAN[0]=0
TVAR[0]=0
TSTDEV[0]=0
TSTERR[0]=0
SEED=$6
DISTRIBUTION=("Normal" "Uniform")

LB=256
cd ../src

for N in `seq ${STARTN} ${DN} ${ENDN}`;
do
    #echo "DEV=${DEV}  N=${N} B=${B}"
    echo -n "${N}   ${B}    " >> ../data/${OUTFILE}_${DISTRIBUTION[$DIST]}.dat
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
            COMPILE=`make BSIZE=${BSIZE[$q]} R=${chain[$q]}`
            echo "$COMPILE"
            for k in `seq 1 ${SAMPLES}`;
            do
                #echo "/${BINARY} ${DEV}    ${N} ${ns[$q]} ${SEED} ${REPEAT} ${DIST} ${met[$q]}"
                value=`./${BINARY} ${DEV}    ${N} ${ns[$q]} ${SEED} ${REPEAT} ${DIST} ${met[$q]}`
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
            echo " "
            echo "${METHODS[$q]}     -->     R=${chain[$q]} and  BSIZE=${BSIZE[$q]}"
            echo "---> B=${B} N=${N} --> (MEAN, VAR, STDEV, STERR, SUM, CPUSUM, #DIFF, %DIFF) 
                -> (${TMEAN[$q]}[ms], ${TVAR[$q]}, ${TSTDEV[$q]}, ${TSTERR[$q]}, ${y2}, ${w2}, ${z2}, ${v2})"
            echo -n "${TMEAN[$q]} ${TVAR[$q]} ${TSTDEV[$q]} ${TSTERR[$q]} ${y} ${w} ${z} ${v}        " >> ../data/${OUTFILE}_${DISTRIBUTION[$DIST]}.dat
            echo " "
        done
    echo " " >> ../data/${OUTFILE}_${DISTRIBUTION[$DIST]}.dat
    echo " "
    echo " "
    echo " "
done 
echo " "

