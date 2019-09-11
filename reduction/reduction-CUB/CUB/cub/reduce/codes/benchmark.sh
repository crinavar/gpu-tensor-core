#!/bin/bash
if [ "$#" -ne 6 ]; then
    echo "run as ./benchmark-blockconf.sh    STARTN ENDN DN     SAMPLES BINARY OUTFILE"
    exit;
fi

STARTN=$1
ENDN=$3
DN=$2
SAMPLES=$4
BINARY=${5}
OUTFILE=${6}
TMEAN[0]=0
TVAR[0]=0
TSTDEV[0]=0
TSTERR[0]=0

echo "run benchmark as STARTN: $STARTN, ENDN: $ENDN, DN: $DN, SAMPLE: $SAMPLES, BINARY: $BINARY, OUTFILE: $OUTFILE"
for N in `seq ${STARTN} ${DN} ${ENDN}`;
do
    x=0
    M=0
    S=0
    echo "./${BINARY} ${N}"
    for k in `seq 1 ${SAMPLES}`;
    do
        value=`./${BINARY} ${N}`
        oldM=$M;
        x="$(cut -d',' -f1 <<<"$value")"
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
    echo "$N, ${TMEAN[$q]}[ms], ${TVAR[$q]}, ${TSTDEV[$q]}, ${TSTERR[$q]}"
    echo "$N ${TMEAN[$q]} ${TVAR[$q]} ${TSTDEV[$q]} ${TSTERR[$q]}" >> data/${OUTFILE}.dat
done
echo " " >> data/${OUTFILE}.dat
echo " "
