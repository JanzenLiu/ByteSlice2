#ifndef BYTEWISE_SCAN_H
#define BYTEWISE_SCAN_H

#include    <vector>
#include	<ctime>
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
	// should be constant theoretically, not set function here in order to use the shuffle function
	size_t column_id;
	size_t byte_id;
};

typedef std::vector<ByteInColumn> Sequence;

/**
  * @brief Evaluate complex predicates as
  * "Conjunction of Byte Disjunctions" in a bytewise pipeline manner.
  * @Warning The input columns must be ByteSlice type
  */
class BytewiseScan{
public:
	void AddPredicate(BytewiseAtomPredicate predicate);
	void SetSequence(const Sequence seq);
	bool ValidSequence(Sequence seq) const;
	Sequence NaturalSequence() const;
	Sequence RandomSequence() const;
	
	// accessor
	BytewiseAtomPredicate GetPredicate(size_t pid) const;
	Sequence sequence() const;
	size_t num_bytes_all() const;

private:
	std::vector<BytewiseAtomPredicate> conjunctions_;
	Sequence sequence_;
	size_t num_bytes_all_ = 0; //correct?
};

}	//namespace
#endif