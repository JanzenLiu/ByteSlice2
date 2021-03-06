#ifndef COLUMN_BLOCK_H
#define COLUMN_BLOCK_H

#include    "common.h"
#include    "types.h"
#include    "avx-utility.h"
#include    "bitvector_block.h"
#include    "byte_mask_block.h"
#include    "sequential_binary_file.h"

namespace byteslice{

class ColumnBlock{
public:
    virtual ~ColumnBlock(){
    }

    virtual WordUnit GetTuple(size_t pos_in_block) const = 0;
    virtual void SetTuple(size_t pos_in_block, WordUnit value) = 0;
    virtual void Scan(Comparator comparator, WordUnit literal, BitVectorBlock* bv_block, Bitwise bit_opt=Bitwise::kSet) const = 0;
    virtual void Scan(Comparator comparator, const ColumnBlock* column_block, BitVectorBlock* bv_block, Bitwise bit_opti=Bitwise::kSet) const = 0;
    virtual void BulkLoadArray(const WordUnit* codes, size_t num, size_t start_pos=0) = 0;
    virtual void SerToFile(SequentialWriteBinaryFile &file) const = 0;
    virtual void DeserFromFile(const SequentialReadBinaryFile &file) = 0;
    virtual bool Resize(size_t size) = 0;

    virtual AvxUnit GetAvxUnit(size_t offset, size_t byte_id) const{
        return avx_zero();
    }
    virtual void Prefetch(size_t byte_id, size_t offset, size_t distance) const{
    }

    //Scan procedure that takes in and output 8-bit masks
    //This method is only used by ByteSlice
    //Otherwise it does nothing!
    virtual void Scan(Comparator comparator, WordUnit literal, ByteMaskBlock* bmblk, 
            Bitwise opt = Bitwise::kSet) const{
    }

    //Scan procedure that scan a particular byte
    //These methods is only used by ByteSlice
    //Ohterwise it does nothing!
    virtual void ScanByte(size_t byte_id, Comparator comparator, ByteUnit literal,
        ByteMaskBlock* bm_less, ByteMaskBlock* bm_greater, ByteMaskBlock* bm_equal, 
        ByteMaskBlock* input_mask) const{
    }
    virtual void ScanByte(size_t byte_id, Comparator comparator, ByteUnit literal,
    	ByteMaskBlock* bm_less, ByteMaskBlock* bm_greater, ByteMaskBlock* bm_equal) const{
    }

    //accessor
    ColumnType type() const;
    size_t bit_width() const;
    size_t num_tuples() const;


protected:
    ColumnBlock(ColumnType type, size_t bit_width, size_t num):
        type_(type), bit_width_(bit_width), num_tuples_(num){
    }
    const ColumnType type_;
    const size_t bit_width_;
    size_t num_tuples_;
    

};

inline ColumnType ColumnBlock::type() const{
    return type_;
}

inline size_t ColumnBlock::bit_width() const{
    return bit_width_;
}

inline size_t ColumnBlock::num_tuples() const{
    return num_tuples_;
}

}

#endif  //COLUMN_BLOCK_H
