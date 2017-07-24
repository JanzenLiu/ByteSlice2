#include    "include/pipeline_scan.h"
#include    <omp.h>
#include    <vector>

namespace byteslice{

void BytewiseScan::AddPredicate(BytewiseAtomPredicate predicate){
	assert(predicate.column->type() == ColumnType::kByteSlicePadRight);
    conjunctions_.push_back(predicate);
    for(size_t i = 0; i < predicate.num_bytes, i++){
    	sequence_.push_back(ByteInColumn(conjunctions_.size() - 1, i));
    }
}

BytewiseAtomPredicate BytewiseScan::GetPredicate(size_t, pid){
	return conjunctions_[pid];
}

}	//namespace