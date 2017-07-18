#include    <iostream>
#include    <fstream>
#include    <string>
#include    <map>
#include    <vector>

#include    "exp-utility.h"

#include    "util/system_perf_monitor.h"
#include    "include/hybrid_timer.h"

#include    "include/types.h"
#include    "include/column.h"
#include    "include/bitvector.h"
#include    "include/bitvector_iterator.h"

using namespace byteslice;

int main(int argc, char* argv[]){

    ColumnType type = ColumnType::kByteSlicePadRight;
    size_t repeat = 5;
    std::string query_file;
    std::map<std::string, Column*> table;
    std::vector<ScanCondition> selections;
    std::vector<ScanColumnCondition> columnscans;
    std::vector<std::string> aggregates;
    size_t num_selections;
    size_t num_columncmp;
    size_t num_aggregates;
    size_t num_rows;
    size_t num_columns;

    //Options:
    //t - column type; r - repeat
    //q - query file
    int c;
    while((c = getopt(argc, argv, "t:r:q:")) != -1){
        switch(c){
            case 't':
                if(0 == strcmp(optarg, "n"))
                    type = ColumnType::kNaive;
                else if(0 == strcmp(optarg, "navx"))
                    type = ColumnType::kNaiveAvx;
                else if(0 == strcmp(optarg, "hbp"))
                    type = ColumnType::kHbp;
                else if(0 == strcmp(optarg, "bits"))
                    type = ColumnType::kBitSlice;
                else if(0 == strcmp(optarg, "bytes"))
                    type = ColumnType::kByteSlicePadRight;
                else if(0 == strcmp(optarg, "hs"))
                    type = ColumnType::kHybridSlice;
                else if(0 == strcmp(optarg, "nnavx"))
                    type = ColumnType::kNumpsAvx;
                else if(0 == strcmp(optarg, "shs"))
                    type = ColumnType::kSmartHybridSlice;
                else if(0 == strcmp(optarg, "avx2"))
                    type = ColumnType::kAvx2Scan;
                else if(0 == strcmp(optarg, "vbp"))
                    type = ColumnType::kVbp;
                else{
                    std::cerr << "Unknown column type:" << optarg << std::endl;
                    exit(1);
                }
                break;
            case 'r':
                repeat = atoi(optarg);
                break;
            case 'q':
                query_file = std::string(optarg);
                break;
        }
    }

    //Parse the query file
    std::ifstream ifs(query_file, std::ifstream::in);
    ifs >> num_rows;
    std::cout << "num of rows: " << num_rows << std::endl;
    ifs >> num_columns;
    //Load columns from text files
    for(size_t i=0; i<num_columns; i++){
        std::string cname;
        size_t bit_width;
        ifs >> cname >> bit_width;
        Column *column = new Column(type, bit_width, num_rows);
        column->LoadTextFile(cname);
        table[cname] = column;
        std::cout << "Load column: " << cname << std::endl;
    }
    //Add scan conditions
    ifs >> num_selections;
    for(size_t i=0; i<num_selections; i++){
        std::string cname;
        std::string scmp;
        WordUnit literal;
        Comparator cmp;
        ifs >> cname >> scmp >> literal;
        //translate comparator
        if(0 == scmp.compare("lt"))
            cmp = Comparator::kLess;
        else if(0 == scmp.compare("le"))
            cmp = Comparator::kLessEqual;
        else if(0 == scmp.compare("gt"))
            cmp = Comparator::kGreater;
        else if(0 == scmp.compare("ge"))
            cmp = Comparator::kGreaterEqual;
        else if(0 == scmp.compare("eq"))
            cmp = Comparator::kEqual;
        else if(0 == scmp.compare("ne"))
            cmp = Comparator::kInequal;
        else{
            std::cerr << "Unknown predicate: " << scmp << std::endl;
            exit(1);
        }

        selections.push_back(ScanCondition(cname, cmp, literal));
        std::cout << "Add Condition: " << cname << " " << cmp << " " << literal << std::endl;
    }
    //Add multi-column scans
    ifs >> num_columncmp;
    for(size_t i=0; i < num_columncmp; i++){
        std::string cname1;
        std::string scmp;
        std::string cname2;
        Comparator cmp;
        ifs >> cname1 >> scmp >> cname2;
        //translate comparator
        if(0 == scmp.compare("lt"))
            cmp = Comparator::kLess;
        else if(0 == scmp.compare("le"))
            cmp = Comparator::kLessEqual;
        else if(0 == scmp.compare("gt"))
            cmp = Comparator::kGreater;
        else if(0 == scmp.compare("ge"))
            cmp = Comparator::kGreaterEqual;
        else if(0 == scmp.compare("eq"))
            cmp = Comparator::kEqual;
        else if(0 == scmp.compare("ne"))
            cmp = Comparator::kInequal;
        else{
            std::cerr << "Unknown predicate: " << scmp << std::endl;
            exit(1);
        }
        columnscans.push_back(ScanColumnCondition(cname1, cmp, cname2));
        std::cout << "Add Condition: " << cname1 << " " << cmp << " " << cname2 << std::endl;
    }
    //Add aggregations
    ifs >> num_aggregates;
    for(size_t i=0; i<num_aggregates; i++){
        std::string cname;
        ifs >> cname;
        aggregates.push_back(cname);
        std::cout << "Add aggregate: " << cname << std::endl;
    }

    ifs.close();

    /*-------------------------------------------*/
    HybridTimer t1;
    uint64_t sum_rdtsc_scan = 0, sum_rdtsc_agg = 0;
    BitVector *bitvector = new BitVector(num_rows);

    for(size_t run=0; run < 1+repeat; run++){
        bitvector->SetOnes();
        //---Scan
        t1.Start();
        for(size_t s=0; s < selections.size(); s++){
            Column *column = table[selections[s].cname];
            Comparator comparator = selections[s].comparator;
            WordUnit literal = selections[s].literal;
            column->Scan(comparator, literal, bitvector, (0==s)? Bitwise::kSet : Bitwise::kAnd);
        }
        for(size_t t=0; t < columnscans.size(); t++){
            Column *column1 = table[columnscans[t].cname];
            Comparator comparator = columnscans[t].comparator;
            Column *column2 = table[columnscans[t].cname_other];
            column1->Scan(comparator, column2, bitvector, Bitwise::kAnd);
        }
        t1.Stop();
        if(run > 0){
            sum_rdtsc_scan += t1.GetNumCycles();
        }
        //-------

        //---Aggregate
        //Prepare the RID-list
        BitVectorIterator* itor = new BitVectorIterator(bitvector);
        std::vector<size_t> rid_list;
        size_t count = bitvector->CountOnes();
        rid_list.reserve(count);
        while(itor->Next()){
            size_t id = itor->GetPosition();
            rid_list.push_back(id);
        }

        t1.Start();

//        {   //Implentation 1
//            WordUnit dummy = 1;
//            while(itor->Next()){
//                size_t id = itor->GetPosition();
//                for(size_t a=0; a < aggregates.size(); a++){
//                    Column *column = table[aggregates[a]];
//                    dummy += column->GetTuple(id);
//                }
//            }
//        }

        {   //Implementation 2
            WordUnit dummy = 1;
            for(size_t a = 0; a < aggregates.size(); a++){
                Column *column = table[aggregates[a]];
                for(size_t i=0; i < rid_list.size(); i++){
                    size_t rid = rid_list[i];
                    dummy += column->GetTuple(rid);
                }
            }
        }

        t1.Stop();

        delete itor;
        if(run > 0){
            sum_rdtsc_agg += t1.GetNumCycles();
        }
        //-------
    }

    std::cout << "Result: " << bitvector->CountOnes() << std::endl;

    delete bitvector;
    /*------------------------------------------*/

    //Print result
    float scan_time = float(sum_rdtsc_scan / repeat) / num_rows;
    float agg_time = float(sum_rdtsc_agg / repeat) / num_rows;
    //std::cout << "#ColumnType selection\t  aggregation\t total(cy/row)" << std::endl;
    std::cout << type << std::endl;
    std::cout << "Total = selection + projection" << std::endl;
    std::cout << (scan_time + agg_time) << "\t"
              << scan_time << "\t"
              << agg_time << std::endl;
    
    //Destroy columns
    std::map<std::string, Column*>::iterator it;
    for(it = table.begin(); it != table.end(); it++){
        delete it->second;
    }
}
