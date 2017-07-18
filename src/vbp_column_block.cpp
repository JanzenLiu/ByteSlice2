#include    "include/vbp_column_block.h"
#include    <cstdlib>
#include    <cstring>
#include    <algorithm>
#include    "include/avx-utility.h"

namespace byteslice{

template <size_t BIT_WIDTH>
VbpColumnBlock<BIT_WIDTH>::VbpColumnBlock(size_t num):
    ColumnBlock(ColumnType::kVbp, BIT_WIDTH, num){

    assert(num <= kNumTuplesPerBlock);

    //Compute bit group helper
    for(size_t gid = 0; gid < kNumBitGroups; gid++){
        bitgroup_helper_[gid] = std::min(kNumBitsPerGroup, BIT_WIDTH - gid * kNumBitsPerGroup);
    }

    //Allocate data space
    for(size_t gid = 0; gid < kNumBitGroups; gid++){
        size_t size = 
            sizeof(AvxUnit) * bitgroup_helper_[gid] * CEIL(kNumTuplesPerBlock, kNumAvxBits);
        size_t ret =
            posix_memalign((void**)&data_[gid], 32, size);
        ret = ret;
        memset(data_[gid], 0x0, size);
    }
}

template <size_t BIT_WIDTH>
VbpColumnBlock<BIT_WIDTH>::~VbpColumnBlock(){
    for(size_t gid = 0; gid < kNumBitGroups; gid++){
        free(data_[gid]);
    }
}

template <size_t BIT_WIDTH>
bool VbpColumnBlock<BIT_WIDTH>::Resize(size_t num){
    num_tuples_ = num;
    return true;
}

template <size_t BIT_WIDTH>
void VbpColumnBlock<BIT_WIDTH>::SerToFile(SequentialWriteBinaryFile &file) const{
}

template <size_t BIT_WIDTH>
void VbpColumnBlock<BIT_WIDTH>::DeserFromFile(const SequentialReadBinaryFile &file){
}

//Scan against literal
template <size_t BIT_WIDTH>
void VbpColumnBlock<BIT_WIDTH>::Scan(Comparator comparator, WordUnit literal,
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

template <size_t BIT_WIDTH>
template <Comparator CMP>
void VbpColumnBlock<BIT_WIDTH>::ScanHelper1(WordUnit literal,
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

template <size_t BIT_WIDTH>
template <Comparator CMP, Bitwise OPT>
void VbpColumnBlock<BIT_WIDTH>::ScanHelper2(WordUnit literal, BitVectorBlock* bvblock) const{
    //Do real work here
    constexpr size_t stride = sizeof(AvxUnit) / sizeof(WordUnit);
    AvxUnit vliteral[BIT_WIDTH];
    for(size_t bit_id = 0; bit_id < BIT_WIDTH; bit_id++){
        WordUnit bit = 1ULL & (literal >> (BIT_WIDTH -1 - bit_id));
        vliteral[bit_id] = avx_set1<WordUnit>(-bit);
    }

    //For each segment containing 256 tuples
    for(size_t offset = 0; offset < num_tuples_; offset += kNumAvxBits){
        size_t segment_id = offset / kNumAvxBits;
        size_t word_pos = offset / kNumWordBits;

        AvxUnit m_lt, m_gt, m_eq;
        m_gt = avx_zero();
        m_lt = avx_zero();

        switch(OPT){
            case Bitwise::kSet:
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
        for(size_t gid = 0; gid < kNumBitGroups; gid++){
            if(avx_iszero(m_eq)){
                break;
            }
            size_t base_word_id = segment_id * bitgroup_helper_[gid] * stride;
            //for each bit in this group
            for(size_t j=0; j < bitgroup_helper_[gid]; j++){
                size_t bit_id = gid * kNumBitsPerGroup + j;
                AvxUnit data = _mm256_lddqu_si256(
                        reinterpret_cast<__m256i*>(data_[gid] + base_word_id + j*stride));
                ScanKernel<CMP>(data, vliteral[bit_id], m_lt, m_gt, m_eq);               
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

//Scan against other block
template <size_t BIT_WIDTH>
void VbpColumnBlock<BIT_WIDTH>::Scan(Comparator comparator, const ColumnBlock* other_block,
        BitVectorBlock* bvblock, Bitwise bit_opt) const{
    assert(bvblock->num() == num_tuples_);
    assert(other_block->num_tuples() == num_tuples_);
    assert(other_block->type() == type_);
    assert(other_block->bit_width() == bit_width_);
    
    const VbpColumnBlock<BIT_WIDTH>* block2 =
        static_cast<const VbpColumnBlock<BIT_WIDTH>*>(other_block);
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

template <size_t BIT_WIDTH>
template <Comparator CMP>
void VbpColumnBlock<BIT_WIDTH>::ScanHelper1(const VbpColumnBlock<BIT_WIDTH>* other_block,
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

template <size_t BIT_WIDTH>
template <Comparator CMP, Bitwise OPT>
void VbpColumnBlock<BIT_WIDTH>::ScanHelper2(const VbpColumnBlock<BIT_WIDTH>* other_block,
                                            BitVectorBlock* bvblock) const{

    constexpr size_t stride = sizeof(AvxUnit) / sizeof(WordUnit);
    //For each segment containing 256 tuples
    for(size_t offset = 0; offset < num_tuples_; offset += kNumAvxBits){
        size_t segment_id = offset / kNumAvxBits;
        size_t word_pos = offset / kNumWordBits;

        AvxUnit m_lt, m_gt, m_eq;
        m_gt = avx_zero();
        m_lt = avx_zero();

        switch(OPT){
            case Bitwise::kSet:
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
        for(size_t gid = 0; gid < kNumBitGroups; gid++){
            if(avx_iszero(m_eq)){
                break;
            }
            size_t base_word_id = segment_id * bitgroup_helper_[gid] * stride;
            //for each bit in this group
            for(size_t j=0; j < bitgroup_helper_[gid]; j++){
                //size_t bit_id = gid * kNumBitsPerGroup + j;
                AvxUnit data = _mm256_lddqu_si256(
                        reinterpret_cast<__m256i*>(data_[gid] + base_word_id + j*stride));
                AvxUnit other_data = _mm256_lddqu_si256(
                        reinterpret_cast<__m256i*>(other_block->data_[gid] + base_word_id + j*stride));
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
template <size_t BIT_WIDTH>
template <Comparator CMP>
inline void VbpColumnBlock<BIT_WIDTH>::ScanKernel(const AvxUnit &data, const AvxUnit &other,
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

template <size_t BIT_WIDTH>
void VbpColumnBlock<BIT_WIDTH>::BulkLoadArray(const WordUnit* codes,
        size_t num, size_t start_pos){
    assert(start_pos + num <= num_tuples_);
    for(size_t i = 0; i < num; i++){
        SetTuple(start_pos+i, codes[i]);
    }
}

//explict specialization
template class VbpColumnBlock<1>;
template class VbpColumnBlock<2>;
template class VbpColumnBlock<3>;
template class VbpColumnBlock<4>;
template class VbpColumnBlock<5>;
template class VbpColumnBlock<6>;
template class VbpColumnBlock<7>;
template class VbpColumnBlock<8>;
template class VbpColumnBlock<9>;
template class VbpColumnBlock<10>;
template class VbpColumnBlock<11>;
template class VbpColumnBlock<12>;
template class VbpColumnBlock<13>;
template class VbpColumnBlock<14>;
template class VbpColumnBlock<15>;
template class VbpColumnBlock<16>;
template class VbpColumnBlock<17>;
template class VbpColumnBlock<18>;
template class VbpColumnBlock<19>;
template class VbpColumnBlock<20>;
template class VbpColumnBlock<21>;
template class VbpColumnBlock<22>;
template class VbpColumnBlock<23>;
template class VbpColumnBlock<24>;
template class VbpColumnBlock<25>;
template class VbpColumnBlock<26>;
template class VbpColumnBlock<27>;
template class VbpColumnBlock<28>;
template class VbpColumnBlock<29>;
template class VbpColumnBlock<30>;
template class VbpColumnBlock<31>;
template class VbpColumnBlock<32>;
}   //namespace
