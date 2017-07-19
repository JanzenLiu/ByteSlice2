#include    "include/pipeline_scan.h"
#include    "include/byte_mask_block.h"
#include    <omp.h>
#include    <vector>

namespace byteslice{

void PipelineScan::AddPredicate(AtomPredicate predicate){
    //assert(conjunctions_.size() == 0 || predicate.column->num_tuples() == conjunctions_[0].column->num_tuples());
    assert(predicate.column->type() == ColumnType::kByteSlicePadRight);
    conjunctions_.push_back(predicate);
}

void PipelineScan::ExecuteBlockwise(BitVector* bitvector){
    size_t num_blocks = conjunctions_[0].column->GetNumBlocks();

#pragma omp parallel for schedule(dynamic)
    for(size_t block_id = 0; block_id < num_blocks; block_id++){
        size_t num = conjunctions_[0].column->GetBlock(block_id)->num_tuples();
        ByteMaskBlock* bmblk = new ByteMaskBlock(num);
        for(size_t pid = 0; pid < conjunctions_.size(); pid++){
            ColumnBlock* block = conjunctions_[pid].column->GetBlock(block_id);
            block->Scan(conjunctions_[pid].comparator,
                        conjunctions_[pid].literal,
                        bmblk,
                        (0 == pid)? Bitwise::kSet : Bitwise::kAnd);
        }
        bmblk->Condense(bitvector->GetBVBlock(block_id), Bitwise::kSet);
        delete bmblk;
    }
}

void PipelineScan::ExecuteColumnwise(BitVector* bitvector){
    const size_t num_blocks = conjunctions_[0].column->GetNumBlocks();
    std::vector<ByteMaskBlock*> bmvector;

    //allocate all byte mask blocks
    for(size_t block_id = 0; block_id < num_blocks; block_id++){
        size_t num = conjunctions_[0].column->GetBlock(block_id)->num_tuples();
        ByteMaskBlock* bmblk = new ByteMaskBlock(num);
        bmvector.push_back(bmblk);
    }

    //Scan in columnwise approach, but using ByteMask as intermediate.
    for(size_t pid = 0; pid < conjunctions_.size(); pid++){
        const Column* col = conjunctions_[pid].column;
        Comparator cmp = conjunctions_[pid].comparator;
        WordUnit lit = conjunctions_[pid].literal;
        //Whole-column scan
#       pragma omp parallel for schedule(dynamic)
        for(size_t block_id = 0; block_id < num_blocks; block_id++){
            col->GetBlock(block_id)->Scan(cmp,
                                          lit,
                                          bmvector[block_id],
                                          (0 == pid)? Bitwise::kSet : Bitwise::kAnd);

        }
    }

    //Condense into result
    for(size_t block_id = 0; block_id < num_blocks; block_id++){
        bmvector[block_id]->Condense(bitvector->GetBVBlock(block_id), Bitwise::kSet);
    }

    //free byte mask blocks
    for(size_t block_id = 0; block_id < num_blocks; block_id++){
        delete bmvector[block_id];
    }
}

void PipelineScan::ExecuteNaive(BitVector* bitvector){
    BitVector* bitvector2 = new BitVector(conjunctions_[0].column);

    //Execute predicates one by one
    for(size_t pid = 0; pid < conjunctions_.size(); pid++){
        //multi-thread is already contained in it
        conjunctions_[pid].column->Scan(conjunctions_[pid].comparator,
                                        conjunctions_[pid].literal,
                                        (0==pid? bitvector : bitvector2),
                                        Bitwise::kSet);

        //merge the two bitvectors
        if(0 != pid){
            bitvector->And(bitvector2);
        }
    }

    delete bitvector2;
}

void PipelineScan::ExecuteStandard(BitVector* bitvector){
    //Execute predicates one by one
    for(size_t pid = 0; pid < conjunctions_.size(); pid++){
        //multi-thread is already contained in it
        conjunctions_[pid].column->Scan(conjunctions_[pid].comparator,
                                        conjunctions_[pid].literal,
                                        bitvector,
                                        (0==pid? Bitwise::kSet : Bitwise::kAnd));

    }
}

void PipelineScan::ExecuteBytewiseNaive(BitVector* bitvector){
    size_t num_blocks = conjunctions_[0].column->GetNumBlocks();

#pragma omp parallel for schedule(dynamic)
    for(size_t block_id = 0; block_id < num_blocks; block_id++){
        size_t num = conjunctions_[0].column->GetBlock(block_id)->num_tuples();
        BitVectorBlock* bvblk = bitvector->GetBVBlock(block_id);
        for(size_t pid = 0; pid < conjunctions_.size(); pid++){
            ColumnBlock* block = conjunctions_[pid].column->GetBlock(block_id);
            for(size_t bid = 0; bid < CEIL(block->bit_width(), 8); bid++){
                block->ScanByte(conjunctions_[pid].comparator,
                                conjunctions_[pid].literal,
                                bid,
                                bvblk,
                                (0 == pid && 0 == bid)? Bitwise:kSet : Bitwise: kAnd);
            }
        }
    }
}

}   //namespace
