#ifndef COLUMN_H
#define COLUMN_H

#include    <vector>
#include    <cassert>
#include    <string>
#include    "common.h"
#include    "types.h"
#include    "bitvector.h"
#include    "byte_mask_block.h"
#include    "byte_mask_vector.h"
#include    "column_block.h"
#include    "naive_column_block.h"
#include    "naive_avx_column_block.h"
#include    "hbp_column_block.h"
#include    "bitslice_column_block.h"
#include    "byteslice_column_block.h"
#include    "hybridslice_column_block.h"
#include    "avx2scan_column_block.h"
#include    "vbp_column_block.h"
#include    "dualbyteslice_column_block.h"
#include    "sequential_binary_file.h"
#include    "superscalar2_byteslice_column_block.h"
#include    "superscalar4_byteslice_column_block.h"

namespace byteslice{

class BitVector;

class Column{
public:
    Column();
    Column(ColumnType type, size_t bit_width, size_t num=0);
    ~Column();
    void Destroy();    

    WordUnit GetTuple(size_t id) const;
    void SetTuple(size_t id, WordUnit value);
    void Resize(size_t num);

    void SerToFile(SequentialWriteBinaryFile &file) const;
    void DeserFromFile(const SequentialReadBinaryFile &file);

    /**
      @brief Load the column from a projection file in text format. One value per line.
      */
    size_t LoadTextFile(std::string filepath);

    void BulkLoadArray(const WordUnit* codes, size_t num, size_t pos=0);

    void Scan(Comparator comparator, WordUnit literal,
            BitVector* bitvector, Bitwise bit_opt = Bitwise::kSet) const;
    void Scan(Comparator comparator, const Column* other_column, 
            BitVector* bitvector, Bitwise bit_opt = Bitwise::kSet) const;

    ColumnBlock* CreateNewBlock() const;

    //accessors
    size_t num_tuples() const;
    size_t bit_width() const;
    ColumnType type() const;
    size_t GetNumBlocks() const;
    ColumnBlock* GetBlock(size_t block_id) const;

private:
    size_t num_tuples_;
    size_t bit_width_;
    ColumnType type_;
    std::vector<ColumnBlock*> blocks_;
};

inline size_t Column::num_tuples() const{
    return num_tuples_;
}

inline size_t Column::bit_width() const{
    return bit_width_;
}

inline ColumnType Column::type() const{
    return type_;
}

inline size_t Column::GetNumBlocks() const{
    return blocks_.size();
}

inline ColumnBlock* Column::GetBlock(size_t block_id) const{
    return blocks_[block_id];
}


}   //namespace

#endif  //COLUMN_H
