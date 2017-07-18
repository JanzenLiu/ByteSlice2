DIR="./release"
TYPE_LIST="bits hbp avx2"
DISJ_LIST="std"
REPEAT=20

export OMP_NUM_THREADS=1
export GOMP_CPU_AFFINITY=0,1,2,3

for ctype in $TYPE_LIST
do
    for disj in $DISJ_LIST
    do
        ${DIR}/experiments/disjunction_options -r $REPEAT -t $ctype -c $disj -o disjunction-${ctype}-${disj}.data
    done
done


