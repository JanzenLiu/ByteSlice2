#include    "include/bitslice_column_block.h"
#include    <cstdlib>
#include    <cstring>
#include    <algorithm>
#include    "include/avx-utility.h"

namespace byteslice{

#ifdef      NEARLYSTOP
#warning    "Early-stop is disabled in BitSliceColumnBlock!"
#endif
    
BitSliceColumnBlock::BitSliceColumnBlock(size_t num, size_t bit_width):
    ColumnBlock(ColumnType::kBitSlice, bit_width, num),
    num_bit_groups_(CEIL(bit_width, kNumBitsPerGroup)){

    assert(num <= kNumTuplesPerBlock);

    for(size_t i = 0; i < bit_width_; i++){
        size_t ret = posix_memalign((void**)&data_[i], 32, kMemSizePerBitSlice);
        ret = ret;
        memset(data_[i], 0x0, kMemSizePerBitSlice);
    }
}

BitSliceColumnBlock::~BitSliceColumnBlock(){
    for(size_t i = 0; i < bit_width_; i++){
        free(data_[i]);
    }
}

bool BitSliceColumnBlock::Resize(size_t num){
    num_tuples_ = num;
    return true;
}

void BitSliceColumnBlock::SerToFile(SequentialWriteBinaryFile &file) const{
    file.Append(&num_tuples_, sizeof(num_tuples_));
    for(size_t i = 0; i < bit_width_; i++){
        file.Append(data_[i], kMemSizePerBitSlice);
    }
}

void BitSliceColumnBlock::DeserFromFile(const SequentialReadBinaryFile &file){
    file.Read(&num_tuples_, sizeof(num_tuples_));
    for(size_t i = 0; i < bit_width_; i++){
        file.Read(data_[i], kMemSizePerBitSlice);
    }
}

//Scan against literal
void BitSliceColumnBlock::Scan(Comparator comparator, WordUnit literal, 
        BitVectorBlock* bvblock, Bitwise bit_opt) const{
    assert(bvblock->num() == num_tuples_);
    switch(comparator){
        case Comparator::kLess:
            return ScanHelper1<Comparator::kLess>(literal, bvblock, bit_opt);
        case Comparator::kGreater:
            return ScanHelper1<Comparator::kGreater>(literal, bvblock, bit_opt);
        case Comparator::kLessEqual:
            return ScanHelper1<Comparator::kLessEqual>(literal, bvblock, bit_opt);
        case Comparator::kGreaterEqual:
            return ScanHelper1<Comparator::kGreaterEqual>(literal, bvblock, bit_opt);
        case Comparator::kEqual:
            return ScanHelper1<Comparator::kEqual>(literal, bvblock, bit_opt);
        case Comparator::kInequal:
            return ScanHelper1<Comparator::kInequal>(literal, bvblock, bit_opt);
    }
}

template <Comparator CMP>
void BitSliceColumnBlock::ScanHelper1(WordUnit literal, 
        BitVectorBlock* bvblock, Bitwise bit_opt) const{
    switch(bit_opt){
        case Bitwise::kSet:
            return ScanHelper2<CMP, Bitwise::kSet>(literal, bvblock);
        case Bitwise::kAnd:
            return ScanHelper2<CMP, Bitwise::kAnd>(literal, bvblock);
        case Bitwise::kOr:
            return ScanHelper2<CMP, Bitwise::kOr>(literal, bvblock);
    }
}

template <Comparator CMP, Bitwise OPT>
void BitSliceColumnBlock::ScanHelper2(WordUnit literal, BitVectorBlock* bvblock) const{
    //prepare literals
    //AvxUnit* lit256 = new AvxUnit[bit_width_];
    AvxUnit* lit256;
    size_t count = posix_memalign((void**)&lit256, 32, sizeof(AvxUnit)*bit_width_);
    count = count;
    for(ssize_t bit_id = bit_width_-1; bit_id >= 0; bit_id--){
        lit256[bit_id] = avx_set1<WordUnit>(0ULL - (literal & 1ULL));
        literal >>= 1;
    }

    //for each segment
    for(size_t offset = 0; offset < num_tuples_; offset += kNumAvxBits){
        size_t word_pos = offset / kNumWordBits;

        AvxUnit m_lt, m_gt, m_eq;
        m_gt = avx_zero();
        m_lt = avx_zero();

        switch(OPT){
            case Bitwise::kSet:
                //m_eq = avx_set1<WordUnit>(-1ULL);
                m_eq = avx_ones();
                break;
            case Bitwise::kAnd:
                m_eq = bvblock->GetAvxUnit(word_pos);
                break;
            case Bitwise::kOr:
                m_eq = avx_not(bvblock->GetAvxUnit(word_pos));
                break;
        }

        //for each group
        for(size_t group_id=0; group_id < num_bit_groups_; group_id++){
#ifndef     NEARLYSTOP
            if(avx_iszero(m_eq)){   //prune
                break;
            }
#endif
            //for each bit in this group
            for(size_t bit_id = group_id*kNumBitsPerGroup; 
                    bit_id < std::min(group_id*kNumBitsPerGroup+kNumBitsPerGroup, bit_width_); 
                    bit_id++){
                AvxUnit data = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[bit_id]+word_pos));
                ScanKernel<CMP>(data, lit256[bit_id], m_lt, m_gt, m_eq);
            }
        }

        //combine the result
        AvxUnit x;
        switch(CMP){
            case Comparator::kEqual:
                x = m_eq;
                break;
            case Comparator::kInequal:
                x = avx_not(m_eq);
                break;
            case Comparator::kLess:
                x = m_lt;
                break;
            case Comparator::kGreater:
                x = m_gt;
                break;
            case Comparator::kLessEqual:
                x = avx_or(m_lt, m_eq);
                break;
            case Comparator::kGreaterEqual:
                x = avx_or(m_gt, m_eq);
                break;
        }

        //combine into bit vector
        AvxUnit y;
        switch(OPT){
            case Bitwise::kSet:
                y = x;
                break;
            case Bitwise::kAnd:
                y = avx_and(x, bvblock->GetAvxUnit(word_pos));
                break;
            case Bitwise::kOr:
                y = avx_or(x, bvblock->GetAvxUnit(word_pos));
                break;
        }
        bvblock->SetAvxUnit(y, word_pos);

    }
    bvblock->ClearTail();

    //delete lit256;
    free(lit256);
}

//Scan against other block
void BitSliceColumnBlock::Scan(Comparator comparator, const ColumnBlock* other_block,
        BitVectorBlock* bvblock, Bitwise bit_opt) const{
    assert(bvblock->num() == num_tuples_);
    assert(other_block->type() == type_);
    assert(other_block->num_tuples() == num_tuples_);
    assert(other_block->bit_width() == bit_width_);
    const BitSliceColumnBlock* block2 = static_cast<const BitSliceColumnBlock*>(other_block);
    
    //multiplexing
    switch(comparator){
        case Comparator::kLess:
            return ScanHelper1<Comparator::kLess>(block2, bvblock, bit_opt);
        case Comparator::kGreater:
            return ScanHelper1<Comparator::kGreater>(block2, bvblock, bit_opt);
        case Comparator::kLessEqual:
            return ScanHelper1<Comparator::kLessEqual>(block2, bvblock, bit_opt);
        case Comparator::kGreaterEqual:
            return ScanHelper1<Comparator::kGreaterEqual>(block2, bvblock, bit_opt);
        case Comparator::kEqual:
            return ScanHelper1<Comparator::kEqual>(block2, bvblock, bit_opt);
        case Comparator::kInequal:
            return ScanHelper1<Comparator::kInequal>(block2, bvblock, bit_opt);
    }
}

template <Comparator CMP>
void BitSliceColumnBlock::ScanHelper1(const BitSliceColumnBlock* other_block,
        BitVectorBlock* bvblock, Bitwise bit_opt) const{
    switch(bit_opt){
        case Bitwise::kSet:
            return ScanHelper2<CMP, Bitwise::kSet>(other_block, bvblock);
        case Bitwise::kAnd:
            return ScanHelper2<CMP, Bitwise::kAnd>(other_block, bvblock);
        case Bitwise::kOr:
            return ScanHelper2<CMP, Bitwise::kOr>(other_block, bvblock);
    }
}

template <Comparator CMP, Bitwise OPT>
void BitSliceColumnBlock::ScanHelper2(const BitSliceColumnBlock* other_block,
        BitVectorBlock* bvblock) const{

    //for each segment
    for(size_t offset = 0; offset < num_tuples_; offset += kNumAvxBits){
        size_t word_pos = offset / kNumWordBits;

        AvxUnit m_lt, m_gt, m_eq;
        m_gt = avx_zero();
        m_lt = avx_zero();

        switch(OPT){
            case Bitwise::kSet:
                m_eq = avx_set1<WordUnit>(-1ULL);
                break;
            case Bitwise::kAnd:
                m_eq = bvblock->GetAvxUnit(word_pos);
                break;
            case Bitwise::kOr:
                m_eq = avx_not(bvblock->GetAvxUnit(word_pos));
                break;
        }

        //for each group
        for(size_t group_id=0; group_id < num_bit_groups_; group_id++){
            if(avx_iszero(m_eq)){   //prune
                break;
            }
            //for each bit in this group
            for(size_t bit_id = group_id*kNumBitsPerGroup; 
                    bit_id < std::min(group_id*kNumBitsPerGroup+kNumBitsPerGroup, bit_width_); 
                    bit_id++){
                AvxUnit data = 
                    _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[bit_id]+word_pos));
                AvxUnit other_data
                    = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(other_block->data_[bit_id]+word_pos));
                ScanKernel<CMP>(data, other_data, m_lt, m_gt, m_eq);
            }
        }

        //combine the result
        AvxUnit x;
        switch(CMP){
            case Comparator::kEqual:
                x = m_eq;
                break;
            case Comparator::kInequal:
                x = avx_not(m_eq);
                break;
            case Comparator::kLess:
                x = m_lt;
                break;
            case Comparator::kGreater:
                x = m_gt;
                break;
            case Comparator::kLessEqual:
                x = avx_or(m_lt, m_eq);
                break;
            case Comparator::kGreaterEqual:
                x = avx_or(m_gt, m_eq);
                break;
        }

        //combine into bit vector
        AvxUnit y;
        switch(OPT){
            case Bitwise::kSet:
                y = x;
                break;
            case Bitwise::kAnd:
                y = avx_and(x, bvblock->GetAvxUnit(word_pos));
                break;
            case Bitwise::kOr:
                y = avx_or(x, bvblock->GetAvxUnit(word_pos));
                break;
        }
        bvblock->SetAvxUnit(y, word_pos);

    }
    bvblock->ClearTail();
}


//ScanKernel: update intermidiate masks based on two data
template <Comparator CMP>
inline void BitSliceColumnBlock::ScanKernel(const AvxUnit &data, const AvxUnit &other,
        AvxUnit &m_lt, AvxUnit &m_gt, AvxUnit &m_eq) const{

    switch(CMP){
        case Comparator::kEqual:
        case Comparator::kInequal:
            m_eq = avx_andnot(avx_xor(data, other), m_eq);
            break;
        case Comparator::kLess:
        case Comparator::kLessEqual:
            m_lt = avx_or(m_lt, avx_and(m_eq, avx_andnot(data, other)));
            m_eq = avx_andnot(avx_xor(data, other), m_eq);
            break;
        case Comparator::kGreater:
        case Comparator::kGreaterEqual:
            m_gt = avx_or(m_gt, avx_and(m_eq,avx_andnot(other, data)));
            m_eq = avx_andnot(avx_xor(data, other), m_eq);
            break;
    }
    return;
}

void BitSliceColumnBlock::BulkLoadArray(const WordUnit* codes, size_t num,
        size_t start_pos){
    assert(start_pos+num <= num_tuples_);
    for(size_t i = 0; i < num; i++){
        SetTuple(start_pos+i, codes[i]);
    }
}

}   //namespace

