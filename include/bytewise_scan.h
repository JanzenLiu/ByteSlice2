#ifndef BYTEWISE_SCAN_H
#define BYTEWISE_SCAN_H

#include    <vector>
#include    "common.h"
#include    "types.h"
#include    "column.h"
#include    "byteslice_column_block.h"

namespace byteslice{

struct BytewiseAtomPredicate{
    BytewiseAtomPredicate(Column* col, Comparator cmp, WordUnit lit):
    	// to add verification of column type...
        column(col),
        comparator(cmp),
        literal(lit),
        num_bytes(CEIL(col->bit_width(), 8)){
    }
    const Column* column;
    const Comparator comparator;
    const WordUnit literal;
    const size_t num_bytes;
};

struct ByteInColumn{
	ByteInColumn(size_t col_id, size_t byte_id):
		column_id(col_id),
		byte_id(byte_id){
	}
	const size_t column_id;
	const size_t byte_id;
}

/**
  * @brief Evaluate complex predicates as
  * "Conjunction of Byte Disjunctions" in a bytewise pipeline manner.
  * @Warning The input columns must be ByteSlice type
  */
class BytewiseScan{
public:
	void AddPredicate(BytewiseAtomPredicate predicate);
	BytewiseAtomPredicate GetPredicate(size_t pid);

private:
	std::vector<BytewiseAtomPredicate> conjunctions_;
	std::vector<ByteInColumn> sequence_;

};

}	//namespace
#endif