#ifndef _EXP_UTILITY_H_
#define _EXP_UTILITY_H_

#include    <string>
#include    "include/types.h"
#include    "common.h"

namespace byteslice{
struct ScanCondition{
    ScanCondition(std::string name, Comparator cmp, WordUnit lit):
        cname(name),
        comparator(cmp),
        literal(lit){
    }
    
    std::string cname;  //column name
    Comparator comparator;
    WordUnit literal;
};

struct ScanColumnCondition{
    ScanColumnCondition(std::string name, Comparator cmp, std::string name2):
        cname(name),
        comparator(cmp),
        cname_other(name2){
    }

    std::string cname;
    Comparator comparator;
    std::string cname_other;
};


}   //namespace
#endif
