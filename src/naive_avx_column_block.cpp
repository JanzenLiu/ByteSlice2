#include    "include/naive_avx_column_block.h"
#include    <cstdlib>
#include    <cstdint>
#include    <cstring>
#include    "include/avx-utility.h"

namespace byteslice{

template <typename DTYPE>
NaiveAvxColumnBlock<DTYPE>::NaiveAvxColumnBlock(size_t num):
    ColumnBlock(ColumnType::kNaiveAvx, sizeof(DTYPE)*8, num){
    assert(num <= kNumTuplesPerBlock);
    size_t ret = posix_memalign((void**)&data_, 32, sizeof(DTYPE)*kNumTuplesPerBlock);
    ret = ret;
    memset(data_, 0x0, sizeof(DTYPE)*kNumTuplesPerBlock);

}

template <typename DTYPE>
NaiveAvxColumnBlock<DTYPE>::~NaiveAvxColumnBlock(){
    free(data_);
}

template <typename DTYPE>
bool NaiveAvxColumnBlock<DTYPE>::Resize(size_t num){
    num_tuples_ = num;
    return true;
}

template <typename DTYPE>
void NaiveAvxColumnBlock<DTYPE>::SerToFile(SequentialWriteBinaryFile &file) const{
    file.Append(&num_tuples_, sizeof(num_tuples_));
    file.Append(data_, sizeof(DTYPE)*kNumTuplesPerBlock);
}

template <typename DTYPE>
void NaiveAvxColumnBlock<DTYPE>::DeserFromFile(const SequentialReadBinaryFile &file){
    file.Read(&num_tuples_, sizeof(num_tuples_));
    file.Read(data_, sizeof(DTYPE)*kNumTuplesPerBlock);
}

//Scan against a literal
template <typename DTYPE>
void NaiveAvxColumnBlock<DTYPE>::Scan(Comparator comparator, WordUnit literal, 
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

template <typename DTYPE>
template <Comparator CMP>
void NaiveAvxColumnBlock<DTYPE>::ScanHelper1(WordUnit literal, BitVectorBlock* bvblock, 
        Bitwise bit_opt) const{
    switch(bit_opt){
        case Bitwise::kSet:
            return ScanHelper2<CMP, Bitwise::kSet>(literal, bvblock);
        case Bitwise::kAnd:
            return ScanHelper2<CMP, Bitwise::kAnd>(literal, bvblock);
        case Bitwise::kOr:
            return ScanHelper2<CMP, Bitwise::kOr>(literal, bvblock);
    }
}

template <typename DTYPE>
template <Comparator CMP, Bitwise OPT>
void NaiveAvxColumnBlock<DTYPE>::ScanHelper2(WordUnit literal, BitVectorBlock* bvblock) const{
    //Do the real work here
    DTYPE lit = FLIP<DTYPE>(static_cast<DTYPE>(literal));
    AvxUnit lit256 = avx_set1<DTYPE>(lit);
    for(size_t offset = 0; offset < num_tuples_; offset += kNumAvxBits){
        AvxUnit segment_result = _mm256_setzero_si256();
        ScanIteration<CMP>(segment_result, offset, lit256);
        AvxUnit x = segment_result;
        size_t avx_pos_in_bitvector = offset/kNumWordBits;
        switch(OPT){
            case Bitwise::kSet:
                break;
            case Bitwise::kAnd:
                x = avx_and(x, bvblock->GetAvxUnit(avx_pos_in_bitvector));
                break;
            case Bitwise::kOr:
                x = avx_or(x, bvblock->GetAvxUnit(avx_pos_in_bitvector));
                break;
        }
        bvblock->SetAvxUnit(x, avx_pos_in_bitvector);
    }
    bvblock->ClearTail();

}

/**
  Each iteration compute the scan mask for one segment.
  One segment contains 256 items.
  Output bit vector is one __m256i mask.
*/
template <typename DTYPE>
template <Comparator CMP>
void NaiveAvxColumnBlock<DTYPE>::ScanIteration(AvxUnit &result,
        const size_t &offset, const AvxUnit &lit256) const{

    const size_t leap = sizeof(AvxUnit)/sizeof(DTYPE);
    WordUnit buffer[4] = {0,0,0,0};
    for(size_t i = 0; i < kNumAvxBits; i += leap){
        AvxUnit data256 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_+offset+i));
        ScanKernel<CMP>(data256, lit256, buffer, i);
    }

    result = _mm256_set_epi64x(buffer[3], buffer[2], buffer[1], buffer[0]);
}

//Scan against another block
template <typename DTYPE>
void NaiveAvxColumnBlock<DTYPE>::Scan(Comparator comparator, const ColumnBlock* other_block,
        BitVectorBlock* bvblock, Bitwise bit_opt) const{
    assert(other_block->type() == type_);
    assert(other_block->bit_width() == bit_width_);
    assert(other_block->num_tuples() == num_tuples_);
    assert(bvblock->num() == num_tuples_);
    
    switch(comparator){
        case Comparator::kLess:
            return ScanHelper1<Comparator::kLess>(other_block, bvblock, bit_opt);
        case Comparator::kGreater:
            return ScanHelper1<Comparator::kGreater>(other_block, bvblock, bit_opt);
        case Comparator::kLessEqual:
            return ScanHelper1<Comparator::kLessEqual>(other_block, bvblock, bit_opt);
        case Comparator::kGreaterEqual:
            return ScanHelper1<Comparator::kGreaterEqual>(other_block, bvblock, bit_opt);
        case Comparator::kEqual:
            return ScanHelper1<Comparator::kEqual>(other_block, bvblock, bit_opt);
        case Comparator::kInequal:
            return ScanHelper1<Comparator::kInequal>(other_block, bvblock, bit_opt);
    }

}

template <typename DTYPE>
template <Comparator CMP>
void NaiveAvxColumnBlock<DTYPE>::ScanHelper1(const ColumnBlock* other_block,
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

template <typename DTYPE>
template <Comparator CMP, Bitwise OPT>
void NaiveAvxColumnBlock<DTYPE>::ScanHelper2(const ColumnBlock* other_block,
        BitVectorBlock* bvblock) const{
    //Do the real work here
    for(size_t offset = 0; offset < num_tuples_; offset += kNumAvxBits){
        AvxUnit segment_result = _mm256_setzero_si256();
        ScanIteration<CMP>(segment_result, offset, other_block);
        AvxUnit x = segment_result;
        size_t avx_pos_in_bitvector = offset/kNumWordBits;
        switch(OPT){
            case Bitwise::kSet:
                break;
            case Bitwise::kAnd:
                x = avx_and(x, bvblock->GetAvxUnit(avx_pos_in_bitvector));
                break;
            case Bitwise::kOr:
                x = avx_or(x, bvblock->GetAvxUnit(avx_pos_in_bitvector));
                break;
        }
        bvblock->SetAvxUnit(x, avx_pos_in_bitvector);
    }
    bvblock->ClearTail();

}

template <typename DTYPE>
template <Comparator CMP>
void NaiveAvxColumnBlock<DTYPE>::ScanIteration(AvxUnit &result, const size_t &offset,
        const ColumnBlock* other_block) const{
    const size_t leap = sizeof(AvxUnit)/sizeof(DTYPE);
    WordUnit buffer[4] = {0,0,0,0};
    for(size_t i = 0; i < kNumAvxBits; i += leap){
        AvxUnit data256 
            = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_+offset+i));
        AvxUnit other256 
            = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(static_cast<const NaiveAvxColumnBlock*>(other_block)->data_+offset+i));
        ScanKernel<CMP>(data256, other256, buffer, i);
    }

    result = _mm256_set_epi64x(buffer[3], buffer[2], buffer[1], buffer[0]);
}


template <typename DTYPE>
template <Comparator CMP>
void NaiveAvxColumnBlock<DTYPE>::ScanKernel(const AvxUnit &data, const AvxUnit &other,
        WordUnit (&buffer)[4], const size_t &index) const{

    AvxUnit m_lt, m_gt, m_eq;
    switch(CMP){
        case Comparator::kLessEqual:
            m_eq = avx_cmpeq<DTYPE>(data, other);
        case Comparator::kLess:
            m_lt = avx_cmplt<DTYPE>(data, other);
            break;
        case Comparator::kGreaterEqual:
            m_eq = avx_cmpeq<DTYPE>(data, other);
        case Comparator::kGreater:
            m_gt = avx_cmpgt<DTYPE>(data, other);
            break;
        case Comparator::kEqual:
        case Comparator::kInequal:
            m_eq = avx_cmpeq<DTYPE>(data, other);
            break;
    }
    AvxUnit mask;
    switch(CMP){
        case Comparator::kLessEqual:
            mask = avx_or(m_lt, m_eq);
            break;
        case Comparator::kLess:
            mask = m_lt;
            break;
        case Comparator::kGreaterEqual:
            mask = avx_or(m_gt, m_eq);
            break;
        case Comparator::kGreater:
            mask = m_gt;
            break;
        case Comparator::kEqual:
            mask = m_eq;
            break;
        case Comparator::kInequal:
            mask = avx_not(m_eq);
            break;
    }

    //put the movemask into the result
    int mmask;
    switch(sizeof(DTYPE)){
        case 1:
            {
                mmask = _mm256_movemask_epi8(mask);
                break;
            }
        case 2:
            {
                mmask = movemask_epi16(mask);
                break;
            }
        case 4:
            {
                mmask = _mm256_movemask_ps(reinterpret_cast<__m256>(mask));
                break;
            }
        case 8:
            {   
                mmask = _mm256_movemask_pd(reinterpret_cast<__m256d>(mask));
                break;
            }
    }

    buffer[index/kNumWordBits] |=
        (static_cast<WordUnit>(static_cast<uint32_t>(mmask)) << (index%kNumWordBits));

}

template <typename DTYPE>
void NaiveAvxColumnBlock<DTYPE>::BulkLoadArray(const WordUnit* codes, size_t num, 
        size_t start_pos){
    assert(start_pos + num <= num_tuples_);
    for(size_t i = 0; i < num; i++){
        data_[start_pos+i] = FLIP<DTYPE>(static_cast<DTYPE>(codes[i]));
    }

}

//explict specialization
template class NaiveAvxColumnBlock<uint8_t>;
template class NaiveAvxColumnBlock<uint16_t>;
template class NaiveAvxColumnBlock<uint32_t>;
template class NaiveAvxColumnBlock<uint64_t>;

}   //namespace
