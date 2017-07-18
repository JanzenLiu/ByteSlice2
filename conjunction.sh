DIR="./release"
TYPES="bits hbp avx2"
CONJ="naive std"

for T in $TYPES
do
    for C in $CONJ
    do
        $DIR/experiments/conjunction_options -t $T -c $C -o conjunction-$T-$C.data
    done
done

