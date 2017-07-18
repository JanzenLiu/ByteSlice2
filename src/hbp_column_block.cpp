#include    "include/hbp_column_block.h"
#include    <cstdlib>
#include    <cstring>
#include    "include/avx-utility.h"

namespace byteslice{

static constexpr size_t kPrefetchDistance = 512;

template <size_t BIT_WIDTH>
HbpColumnBlock<BIT_WIDTH>::HbpColumnBlock(size_t num):
    ColumnBlock(ColumnType::kHbp, BIT_WIDTH, num),
    num_segments_(CEIL(num, kNumCodesPerSegment)){

    assert(num <= kNumTuplesPerBlock);

    for(size_t i=0; i < kNumWordsPerSegment; i++){
        size_t ret = posix_memalign((void**)&data_[i], 32, kMemSizePerWordId);
        ret = ret;
        memset(data_[i], 0x0, kMemSizePerWordId);
    }

    //pre-compute helper
    for(size_t id=0; id < kNumCodesPerSegment; id++){
        seek_helpers_[id].word_id_in_segment = id % kNumWordsPerSegment;
        seek_helpers_[id].shift_in_word = (id / kNumWordsPerSegment) * (BIT_WIDTH + 1);
    }
}

template <size_t BIT_WIDTH>
bool HbpColumnBlock<BIT_WIDTH>::Resize(size_t num){
    num_tuples_ = num;
    num_segments_ = CEIL(num_tuples_, kNumCodesPerSegment);
    return true;
}

template <size_t BIT_WIDTH>
HbpColumnBlock<BIT_WIDTH>::~HbpColumnBlock(){
    for(size_t i = 0; i < kNumWordsPerSegment; i++){
        free(data_[i]);
    }
}

template <size_t BIT_WIDTH>
void HbpColumnBlock<BIT_WIDTH>::SerToFile(SequentialWriteBinaryFile &file) const{
    file.Append(&num_tuples_, sizeof(num_tuples_));

    for(size_t i=0; i < kNumWordsPerSegment; i++){
        file.Append(data_[i], kMemSizePerWordId);
    }
}

template <size_t BIT_WIDTH>
void HbpColumnBlock<BIT_WIDTH>::DeserFromFile(const SequentialReadBinaryFile &file){
    file.Read(&num_tuples_, sizeof(num_tuples_));
    num_segments_ = CEIL(num_tuples_, kNumCodesPerSegment);

    for(size_t i=0; i < kNumWordsPerSegment; i++){
        file.Read(data_[i], kMemSizePerWordId);
    }
}

//Scan against literal
template <size_t BIT_WIDTH>
void HbpColumnBlock<BIT_WIDTH>::Scan(Comparator comparator, WordUnit literal,
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
void HbpColumnBlock<BIT_WIDTH>::ScanHelper1(WordUnit literal,
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
void HbpColumnBlock<BIT_WIDTH>::ScanHelper2(WordUnit literal, BitVectorBlock* bvblock) const{
    //prepare auxiliary masks
    //mask_base = 0^k 1 ... 0^k 1
    //mask_delimiter = 1 0^k ... 1 0^k
    //mask_complement = 0 1^k ... 0 1^k
    AvxUnit mask_base, mask_delimiter, mask_complement;
    WordUnit m64_base = 0ULL;
    for(size_t i=0; i < kNumCodesPerWord; i++){
        m64_base = (m64_base << (BIT_WIDTH+1)) | 1ULL;
    }
    mask_base = avx_set1<WordUnit>(m64_base);
    mask_delimiter = avx_set1<WordUnit>(m64_base << BIT_WIDTH);
    mask_complement = avx_set1<WordUnit>((m64_base << BIT_WIDTH) - m64_base);
    //prepare literal mask
    literal &= kCodeMask;
    WordUnit m64_literal = literal * m64_base;
    AvxUnit mask_literal = avx_set1<WordUnit>(m64_literal);

    WordUnit working_unit = 0ULL;
    size_t cursor_word_id = 0;
    size_t cursor_offset = 0;

    //iterate four segments a time
    for(size_t offset=0; offset < num_segments_; offset += (kNumAvxBits/kNumWordBits)){
        AvxUnit segment_result = avx_zero();
        //for each word in segment
        for(size_t word_id=0; word_id < kNumWordsPerSegment; word_id++){

#           ifndef NPREFETCH
#           warning "Prefetch Enabled in HbpColumnBlock"
            __builtin_prefetch(data_[word_id] + offset + (kPrefetchDistance / 64));
#           endif

            AvxUnit word_result = avx_zero();
            AvxUnit data = 
                _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[word_id]+offset));
            ScanKernel<CMP>(word_result, data, mask_literal, 
                                mask_base, mask_delimiter, mask_complement);
            //merge word_result into segment_result
            segment_result = 
                avx_or(segment_result, _mm256_srli_epi64(word_result, BIT_WIDTH - word_id));
        }
        //Append segment result to bit_vector
        WordUnit *buffer = reinterpret_cast<WordUnit*>(&segment_result);
        for(size_t i=0; i < std::min(kNumAvxBits/kNumWordBits, num_segments_ - offset); i++){
            AppendBitVector<OPT>(working_unit, buffer[i], 
                    bvblock, cursor_word_id, cursor_offset);
        }
    }
    //Write the last half-full working unit
    FlushBitVector<OPT>(working_unit, bvblock, cursor_word_id, cursor_offset);
    bvblock->ClearTail();
}

//Scan against other block
template <size_t BIT_WIDTH>
void HbpColumnBlock<BIT_WIDTH>::Scan(Comparator comparator, const ColumnBlock* other_block,
        BitVectorBlock* bvblock, Bitwise bit_opt) const{
    assert(bvblock->num() == num_tuples_);
    assert(other_block->num_tuples() == num_tuples_);
    assert(other_block->type() == type_);
    assert(other_block->bit_width() == bit_width_);
    
    const HbpColumnBlock<BIT_WIDTH>* block2 =
        static_cast<const HbpColumnBlock<BIT_WIDTH>*>(other_block);
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
void HbpColumnBlock<BIT_WIDTH>::ScanHelper1(const HbpColumnBlock<BIT_WIDTH>* other_block,
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
void HbpColumnBlock<BIT_WIDTH>::ScanHelper2(const HbpColumnBlock<BIT_WIDTH>* other_block,
                                            BitVectorBlock* bvblock) const{

    //prepare auxiliary masks
    //mask_base = 0^k 1 ... 0^k 1
    //mask_delimiter = 1 0^k ... 1 0^k
    //mask_complement = 0 1^k ... 0 1^k
    AvxUnit mask_base, mask_delimiter, mask_complement;
    WordUnit m64_base = 0ULL;
    for(size_t i=0; i < kNumCodesPerWord; i++){
        m64_base = (m64_base << (BIT_WIDTH+1)) | 1ULL;
    }
    mask_base = avx_set1<WordUnit>(m64_base);
    mask_delimiter = avx_set1<WordUnit>(m64_base << BIT_WIDTH);
    mask_complement = avx_set1<WordUnit>((m64_base << BIT_WIDTH) - m64_base);

    WordUnit working_unit = 0ULL;
    size_t cursor_word_id = 0;
    size_t cursor_offset = 0;

    //iterate four segments a time
    for(size_t offset=0; offset < num_segments_; offset += (kNumAvxBits/kNumWordBits)){
        AvxUnit segment_result = avx_zero();
        //for each word in segment
        for(size_t word_id=0; word_id < kNumWordsPerSegment; word_id++){
            AvxUnit word_result = avx_zero();
            AvxUnit data = 
                _mm256_lddqu_si256(reinterpret_cast<__m256i*>(data_[word_id]+offset));
            AvxUnit other_data =
                _mm256_lddqu_si256(reinterpret_cast<__m256i*>(other_block->data_[word_id]+offset));
            ScanKernel<CMP>(word_result, data, other_data, 
                                mask_base, mask_delimiter, mask_complement);
            //merge word_result into segment_result
            segment_result = 
                avx_or(segment_result, _mm256_srli_epi64(word_result, BIT_WIDTH - word_id));
        }
        //Append segment result to bit_vector
        WordUnit *buffer = reinterpret_cast<WordUnit*>(&segment_result);
        for(size_t i=0; i < std::min(kNumAvxBits/kNumWordBits, num_segments_ - offset); i++){
            AppendBitVector<OPT>(working_unit, buffer[i], 
                    bvblock, cursor_word_id, cursor_offset);
        }
    }
    //Write the last half-full working unit
    FlushBitVector<OPT>(working_unit, bvblock, cursor_word_id, cursor_offset);
    bvblock->ClearTail();

}

//Scan Kernel
template <size_t BIT_WIDTH>
template <Comparator CMP>
void HbpColumnBlock<BIT_WIDTH>::ScanKernel(AvxUnit &result, AvxUnit &data, AvxUnit &other_data,
        AvxUnit &m_base, AvxUnit &m_delimiter, AvxUnit &m_complement) const{

    AvxUnit m_lt, m_gt, m_neq;

    //after this step, the delimiter bit contains the T/F value
    //but the mask is dirty: other bits need to be clean up
    switch(CMP){
        case Comparator::kGreaterEqual:
        case Comparator::kLess:
            m_lt = _mm256_add_epi64(other_data, avx_xor(m_complement, data));
            break;
        case Comparator::kLessEqual:
        case Comparator::kGreater:
            m_gt = _mm256_add_epi64(data, avx_xor(m_complement, other_data));
            break;
        case Comparator::kInequal:
        case Comparator::kEqual:
            m_neq = _mm256_add_epi64(m_complement, avx_xor(data, other_data));
            break;
    }

    AvxUnit x;
    switch(CMP){
        case Comparator::kGreaterEqual:
            x = avx_not(m_lt);
            break;
        case Comparator::kLess:
            x = m_lt;
            break;
        case Comparator::kLessEqual:
            x = avx_not(m_gt);
            break;
        case Comparator::kGreater:
            x = m_gt;
            break;
        case Comparator::kInequal:
            x = m_neq;
            break;
        case Comparator::kEqual:
            x = avx_not(m_neq);
            break;
    }
    result = avx_and(x, m_delimiter);
}

//Append bit vector
template <size_t BIT_WIDTH>
template <Bitwise OPT>
void HbpColumnBlock<BIT_WIDTH>::AppendBitVector(WordUnit &working_unit, WordUnit &source,
        BitVectorBlock* bvblock, size_t &cursor_word_id, size_t &cursor_offset) const{

    working_unit |= (source << cursor_offset);
    cursor_offset += kNumCodesPerSegment;
    if(cursor_offset >= kNumWordBits){
        //write the working unit to bit vector
        WordUnit x = working_unit;
        switch(OPT){
            case Bitwise::kSet:
                break;
            case Bitwise::kAnd:
                x = x & bvblock->GetWordUnit(cursor_word_id);
                break;
            case Bitwise::kOr:
                x = x | bvblock->GetWordUnit(cursor_word_id);
                break;
        }
        bvblock->SetWordUnit(x, cursor_word_id);
        working_unit = 0ULL;
        working_unit |= (source >> (kNumCodesPerSegment - (cursor_offset - kNumWordBits)));
        cursor_word_id++;
        cursor_offset -= kNumWordBits;
    }
}

template <size_t BIT_WIDTH>
template <Bitwise OPT>
void HbpColumnBlock<BIT_WIDTH>::FlushBitVector(WordUnit &working_unit, 
        BitVectorBlock* bvblock, size_t &cursor_word_id, size_t &cursor_offset) const{
    if(cursor_offset > 0 && cursor_word_id*kNumWordBits < kNumTuplesPerBlock){
         //write the working unit to bit vector
        WordUnit x = working_unit;
        switch(OPT){
            case Bitwise::kSet:
                break;
            case Bitwise::kAnd:
                x = x & bvblock->GetWordUnit(cursor_word_id);
                break;
            case Bitwise::kOr:
                x = x | bvblock->GetWordUnit(cursor_word_id);
                break;
        }
        bvblock->SetWordUnit(x, cursor_word_id);
    }
}

template <size_t BIT_WIDTH>
void HbpColumnBlock<BIT_WIDTH>::BulkLoadArray(const WordUnit* codes,
        size_t num, size_t start_pos){
    assert(start_pos + num <= num_tuples_);
    for(size_t i = 0; i < num; i++){
        SetTuple(start_pos+i, codes[i]);
    }
}

//explict specialization
template class HbpColumnBlock<1>;
template class HbpColumnBlock<2>;
template class HbpColumnBlock<3>;
template class HbpColumnBlock<4>;
template class HbpColumnBlock<5>;
template class HbpColumnBlock<6>;
template class HbpColumnBlock<7>;
template class HbpColumnBlock<8>;
template class HbpColumnBlock<9>;
template class HbpColumnBlock<10>;
template class HbpColumnBlock<11>;
template class HbpColumnBlock<12>;
template class HbpColumnBlock<13>;
template class HbpColumnBlock<14>;
template class HbpColumnBlock<15>;
template class HbpColumnBlock<16>;
template class HbpColumnBlock<17>;
template class HbpColumnBlock<18>;
template class HbpColumnBlock<19>;
template class HbpColumnBlock<20>;
template class HbpColumnBlock<21>;
template class HbpColumnBlock<22>;
template class HbpColumnBlock<23>;
template class HbpColumnBlock<24>;
template class HbpColumnBlock<25>;
template class HbpColumnBlock<26>;
template class HbpColumnBlock<27>;
template class HbpColumnBlock<28>;
template class HbpColumnBlock<29>;
template class HbpColumnBlock<30>;
template class HbpColumnBlock<31>;
template class HbpColumnBlock<32>;

}   //namespace
