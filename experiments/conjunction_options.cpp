#include    <iostream>
#include    <iomanip>
#include    <fstream>
#include    <unistd.h>
#include    <cstring>
#include    <cstdlib>
#include    <ctime>
#include    <vector>

#include    "util/system_perf_monitor.h"
#include    "include/hybrid_timer.h"

#include    "include/types.h"
#include    "include/column.h"
#include    "include/bitvector.h"
#include    "include/pipeline_scan.h"

using namespace byteslice;

enum class ConjunctType{
    kNaive,
    kPipeline,
    kStandard
};

int main(int argc, char* argv[]){

    //default options
    ColumnType type = ColumnType::kByteSlicePadRight;
    size_t num_rows = 512*1024*1024;
    size_t code_length = 16;
    double selectivity = 0.1;
    Comparator comparator = Comparator::kLess;
    size_t repeat = 20;
    const char* filename = "conjunction_options.data";
    std::ofstream of;
    ConjunctType opt = ConjunctType::kStandard;

    //get options:
    //t - column type; s - column size; b - bit width
    //f - selectivity; r - repeat; p - predicate
    //o - output file
    int c;
    while((c = getopt(argc, argv, "t:s:f:r:o:p:c:h")) != -1){
        switch(c){
            case 'h':
                std::cout << "Usage: " << argv[0]
                            << " -t <column type>"
                            << " -c <conjunction algo = naive|std|pl"
                            << std::endl;
                break;
            case 't':
                if(0 == strcmp(optarg, "hbp"))
                    type = ColumnType::kHbp;
                else if(0 == strcmp(optarg, "bits"))
                    type = ColumnType::kBitSlice;
                else if(0 == strcmp(optarg, "bytes"))
                    type = ColumnType::kByteSlicePadRight;
                else if(0 == strcmp(optarg, "hs"))
                    type = ColumnType::kHybridSlice;
                else if(0 == strcmp(optarg, "shs"))
                    type = ColumnType::kSmartHybridSlice;
                else if(0 == strcmp(optarg, "avx2"))
                    type = ColumnType::kAvx2Scan;
                else if(0 == strcmp(optarg, "vbp"))
                    type = ColumnType::kVbp;
                else if(0 == strcmp(optarg, "dbs"))
                    type = ColumnType::kDualByteSlicePadRight;
                else{
                    std::cerr << "Unknown column type:" << optarg << std::endl;
                    exit(1);
                }
                break;
            case 'p':
                if(0 == strcmp(optarg, "lt"))
                    comparator = Comparator::kLess;
                else if(0 == strcmp(optarg, "le"))
                    comparator = Comparator::kLessEqual;
                else if(0 == strcmp(optarg, "gt"))
                    comparator = Comparator::kGreater;
                else if(0 == strcmp(optarg, "ge"))
                    comparator = Comparator::kGreaterEqual;
                else if(0 == strcmp(optarg, "eq"))
                    comparator = Comparator::kEqual;
                else if(0 == strcmp(optarg, "ne"))
                    comparator = Comparator::kInequal;
                else{
                    std::cerr << "Unknown predicate: " << optarg << std::endl;
                    exit(1);
                }
                break;
            case 's':
                num_rows = atoi(optarg);
                break;
            case 'b':
                code_length = atoi(optarg);
                break;
            case 'f':
                selectivity = atof(optarg);
                break;
            case 'r':
                repeat = atoi(optarg);
                break;
            case 'o':
                filename = optarg;
                break;
            case 'c':
                if(0 == strcmp(optarg, "naive"))
                    opt = ConjunctType::kNaive;
                else if(0 == strcmp(optarg, "pl"))
                    opt = ConjunctType::kPipeline;
                else if(0 == strcmp(optarg, "std"))
                    opt = ConjunctType::kStandard;
                else{
                    std::cerr << "Unknown conjunction option: " << optarg << std::endl;
                    exit(0);
                }
                break;
        }
    }

    of.open(filename, std::ofstream::out);

    //Init PCM
    SystemPerfMonitor pm;
    pm.Init();
    
    HybridTimer t1;

    size_t code_length1 = 14;
    size_t code_length2 = 16;

    //print options
    of << "# "
        << "ColumnType= " << type 
        << "Predicate= " << comparator
        << " num_rows= " << num_rows 
        << " selectivity= " << selectivity
        << " repeat= " << repeat << std::endl;

    of << "#Selectivity\t timeofday\t rdtsc/tuple\t insn/tuple\t";
    of << "L2Miss/tuple\t L3Miss/tuple\t IPC\t";
    of << "Bandwidth(R)\t Bandwidth(W)" << std::endl;


    std::srand(std::time(0));
    /*-------------------------------------------------------------*/
    //Iterate through different selectivity
    std::vector<double> selectivity_vec = {1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.002, 0.001};
    for(double selectivity : selectivity_vec){

        double sum_timeofday = 0;
        uint64_t sum_rdtsc = 0;
        uint64_t sum_l2_misses = 0;
        uint64_t sum_l3_misses = 0;
        double sum_ipc = 0;
        uint64_t sum_insn = 0;
        uint64_t sum_bytes_read = 0;
        uint64_t sum_bytes_write = 0;


        const WordUnit mask1 = (1ULL << code_length1) - 1;
        const WordUnit mask2 = (1ULL << code_length2) - 1;
        WordUnit literal1 = static_cast<WordUnit>(mask1 * selectivity);
        WordUnit literal2 = static_cast<WordUnit>(mask2 * 0.5);

        //Prepare the material
        Column* column1 = new Column(type, code_length1, num_rows);
        Column* column2 = new Column(type, code_length2, num_rows);
        BitVector* bitvector = new BitVector(num_rows);

        //populate the column with random data
        for(size_t i = 0; i < num_rows; i++){
            WordUnit code1 = std::rand() & mask1;
            column1->SetTuple(i, code1);
        }
        for(size_t i = 0; i < num_rows; i++){
            WordUnit code2 = std::rand() & mask2;
            column2->SetTuple(i, code2);
        }

        //Prepare the pipelinescan object
        PipelineScan scan;
        scan.AddPredicate(AtomPredicate(column1, comparator, literal1));
        scan.AddPredicate(AtomPredicate(column2, comparator, literal2));
 

        for(size_t run=0; run < repeat + 1; run++){
            pm.Start();
            t1.Start();

            switch(opt){
                case ConjunctType::kNaive:
                    scan.ExecuteNaive(bitvector);
                    break;
                case ConjunctType::kPipeline:
                    scan.ExecuteBlockwise(bitvector);
                    break;
                case ConjunctType::kStandard:
                    scan.ExecuteStandard(bitvector);
                    break;
            }

            pm.Stop();
            t1.Stop();

            if(run > 0){
                sum_timeofday += t1.GetSeconds();
                sum_rdtsc += t1.GetNumCycles();
                sum_l2_misses += pm.GetL2CacheMisses();
                sum_l3_misses += pm.GetL3CacheMisses();
                sum_ipc += pm.GetIPC();
                sum_insn += pm.GetRetiredInstructions();
                sum_bytes_read += pm.GetBytesReadFromMC();
                sum_bytes_write += pm.GetBytesWrittenToMC();
            }

        }
    
        delete bitvector;
        delete column1;
        delete column2;
        //delete bwv1;
        //delete bwv2;

        of << std::fixed;
        of << std::setprecision(4);
        of << std::left;
        of << selectivity << "\t"
           << sum_timeofday / repeat << "\t"
           << double(sum_rdtsc/repeat) / num_rows << "\t"
           << double(sum_insn/repeat) / num_rows << "\t"
           << double(sum_l2_misses/repeat) / num_rows << "\t"
           << double(sum_l3_misses/repeat) / num_rows << "\t"
           << sum_ipc/repeat << "\t"
           << sum_bytes_read / sum_timeofday << "\t"
           << sum_bytes_write / sum_timeofday << "\t"
           << std::endl;

        of.flush();
    }
    /*-------------------------------------------------------------*/

    pm.Destroy();
    of.close();
}

