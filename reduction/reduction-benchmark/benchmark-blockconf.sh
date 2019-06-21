#!/bin/bash
if [ "$#" -ne 11 ]; then
    echo "run as ./benchmark-blockconf.sh    DEV     STARTB ENDB DB    STARTN ENDN DN     KREPEATS SAMPLES BINARY OUTFILE"
    exit;
fi
DEV=$1
STARTB=$2
ENDB=$3
DB=$4
STARTN=$5
ENDN=$6
DN=$7
REPEAT=$8
SAMPLES=$9
BINARY=${10}
OUTFILE=${11}
METHODS=("shuffle" "tc_theory" "tc_block" "tc_mixed" "tc_chain_8" "tc_chain_16" "tc_chain_32")
NM=$((${#METHODS[@]}-1))
met=("0" "1" "2" "3" "2" "2" "2")
chain=("1" "1" "1" "1" "8" "16" "32")
ns=("0" "0" "0" "0.5" "0" "0" "0")
TMEAN[0]=0
TVAR[0]=0
TSTDEV[0]=0
TSTERR[0]=0
seed=12
for B in `seq ${STARTB} ${DB} ${ENDB}`;
do
    echo "Benchmarking for B=${B}"
    echo "Compiling with BSIZE=$B"
    LB=$((${B}))
    #COMPILE=`make BSIZE=${LB} R=${chain[$q]}`
    #echo ${COMPILE}
    for N in `seq ${STARTN} ${DN} ${ENDN}`;
    do
        echo "DEV=${DEV}  N=${N} B=${B}"
        echo -n "${N}   ${B}    " >> data/${OUTFILE}_B${B}.dat
        #RPARAM=$(($N/${LB}))
        RPARAM=$(($N/${B}))
        for q in `seq 0 ${NM}`;
        do
            COMPILE=`make BSIZE=${LB} R=${chain[$q]}`
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
            # Chosen MAP
            echo "./${BINARY} ${DEV}    ${N} ${ns[$q]} ${seed} ${REPEAT} ${met[$q]}"
            echo -n "${METHODS[$q]} ($q) map (${SAMPLES} Samples)............."
            for k in `seq 1 ${SAMPLES}`;
            do
                value=`./${BINARY} ${DEV}    ${N} ${ns[$q]} $((${seed}*${k})) ${REPEAT} ${met[$q]}`
                #echo "./${BINARY} ${DEV}    ${N} ${ns[$q]} $((${seed}*${k})) ${REPEAT} ${met[$q]}"
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
            #echo "--------> $x1"
            echo "---> B=${B} N=${N} --> (MEAN, VAR, STDEV, STERR, SUM, CPUSUM,
            DIFF, %DIFF) -> (${TMEAN[$q]}[ms], ${TVAR[$q]}, ${TSTDEV[$q]}, ${TSTERR[$q]}, ${y2}, ${w2}, ${z2}, ${v2})"
            echo -n "${TMEAN[$q]} ${TVAR[$q]} ${TSTDEV[$q]} ${TSTERR[$q]} ${y} ${w} ${z} ${v}        " >> data/${OUTFILE}_B${B}.dat
            echo " "
        done
        echo " " >> data/${OUTFILE}_B${B}.dat
        echo " "
        echo " "
        echo " "
    done 
    echo " "
done 
