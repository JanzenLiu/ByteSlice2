#ifndef TABLE_H
#define TABLE_H

#include    <string>
#include    <map>
#include    "column.h"

class Table{
public:
    struct Option{
        bool in_memory;
    };

    void Open(Option option);
    size_t num_tuples() const {return num_tuples_;}
    bool CreateColumn(std::string name, ColumnType type, size_t bit_width);

private:
    size_t num_tuples_;
    std::string path_;
    std::map<std::string, Column*> map_;

};


#endif  //TABLE_H
