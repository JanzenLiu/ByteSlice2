#ifndef _DISJUNCTION_H_
#define _DISJUNCTION_H_

#include    <vector>
#include    "common.h"
#include    "types.h"
#include    "column.h"
#include    "byteslice_column_block.h"
#include    "byte_mask_block.h"

namespace byteslice{

struct AtomPredicate{
    AtomPredicate(Column* col, Comparator cmp, WordUnit lit):
        column(col),
        comparator(cmp),
        literal(lit){
    }
    const Column* column;
    const Comparator comparator;
    const WordUnit literal;
};

/**
  * @brief Evaluate complex predicates as
  * "Conjunction of Disjunctions" in a pipeline manner.
  * @Warning The input columns must be ByteSlice type
  */
class Disjunction{
public:
    void AddPredicate(AtomPredicate predicate);
    void ExecuteBlockwise(BitVector* bitvector);
    void ExecuteColumnwise(BitVector* bitvector);
    void ExecuteNaive(BitVector* bitvector);
    void ExecuteStandard(BitVector* bitvector);


private:
    std::vector<AtomPredicate> conjunctions_;

};
    

}   //namespace
#endif
