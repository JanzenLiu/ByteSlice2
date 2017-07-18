#include    <iostream>
#include    <unistd.h>
#include    <string>
#include    <cstdlib>
#include    <ctime>
#include    <omp.h>
#include    <map>
#include    <random>
#include    <functional>
#include    <hybrid_timer.h>

#include    "include/types.h"
#include    "include/column.h"
#include    "include/bitvector.h"

#include    "include/hybrid_timer.h"
#include    "util/system_perf_monitor.h"

using namespace byteslice;

std::map<std::string, ColumnType> ctypeMap = {
    {"na",      ColumnType::kNaive},
    {"navx",    ColumnType::kNaiveAvx},
    {"avx",     ColumnType::kAvx2Scan},
    {"hbp",     ColumnType::kHbp},
    {"vbp",     ColumnType::kBitSlice},
    {"vbp2",    ColumnType::kVbp},
    {"bs",      ColumnType::kByteSlicePadRight},
    {"hs",      ColumnType::kHybridSlice},
    {"shs",     ColumnType::kSmartHybridSlice},
    {"dbs",     ColumnType::kDualByteSlicePadRight},
	{"s2bs", 	ColumnType::kSuperscalar2ByteSlicePadRight},
	{"s4bs", 	ColumnType::kSuperscalar4ByteSlicePadRight}
};

std::map<std::string, Comparator> cmpMap = {
    {"lt",  Comparator::kLess},
    {"le",  Comparator::kLessEqual},
    {"gt",  Comparator::kGreater},
    {"ge",  Comparator::kGreaterEqual},
    {"eq",  Comparator::kEqual},
    {"ne",  Comparator::kInequal}
};

typedef struct {
    ColumnType  coltype = ColumnType::kByteSlicePadRight;
    size_t      size    = 16*1024*1024;
    size_t      nbits   = 12;
    double      literal_ratio = 0.1;
    size_t      repeat  = 3;
    Comparator  comparator = Comparator::kLess;
} arg_t;


void parse_arg(arg_t &arg, int &argc, char** &argv);
void print_arg(const arg_t& arg);
void print_help(char* prog_name);


    
    
int main(int argc, char* argv[]){
    arg_t arg;
    parse_arg(arg, argc, argv);
    
    std::cout << "[INFO ] omp_max_threads = " << omp_get_max_threads() << std::endl;
    
    std::cout << "[INFO ] Creating column ..." << std::endl;
    Column* column = new Column(arg.coltype, arg.nbits, arg.size);
    std::cout << "[INFO ] Creating bit vector ..." << std::endl;
    BitVector* bitvector = new BitVector(column);
    
    std::cout << "[INFO ] Populating column with random values ..." << std::endl;
    auto dice = std::bind(std::uniform_int_distribution<WordUnit>(
                            std::numeric_limits<WordUnit>::min(),
                            std::numeric_limits<WordUnit>::max()),
                            std::default_random_engine(std::time(0)));
    WordUnit mask = (1ULL << arg.nbits) - 1;
    for(size_t i=0; i < arg.size; i++){
        column->SetTuple(i, dice() & mask);
    }
    
    WordUnit literal = arg.literal_ratio * mask;
    std::cout << "[INFO ] calculated literal = " << literal << std::endl;
    
    std::cout << "[INFO ] Executing scan ..." << std::endl;
    
    HybridTimer t1;
    SystemPerfMonitor pm;
    pm.Init();
    
    
    t1.Start();
    pm.Start();
    
    for(size_t r = 0; r < arg.repeat; r++){
        column->Scan(Comparator::kLess,
                    literal,
                    bitvector,
                    Bitwise::kSet);
    }
    t1.Stop();
    pm.Stop();
    
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "total seconds, cycle/code" << std::endl;
    std::cout << t1.GetSeconds()/arg.repeat << ", "
                << double(t1.GetNumCycles()/arg.repeat)/arg.size
                << std::endl;
    std::cout << "insn/tuple, L2hr, L3hr, L2mis/tuple, L3mis/tuple, IPC, BW(GB/s), L2lost, L3lost" << std::endl;
    std::cout << double(pm.GetRetiredInstructions()/arg.repeat)/arg.size << ", "
              << pm.GetL2CacheHitRatio() << ", "
              << pm.GetL3CacheHitRatio() << ", "
              << double(pm.GetL2CacheMisses()/arg.repeat)/arg.size << ", "
              << double(pm.GetL3CacheMisses()/arg.repeat)/arg.size << ", "
              << pm.GetIPC() << ", "
              << double((pm.GetBytesReadFromMC() + pm.GetBytesWrittenToMC())/t1.GetSeconds())/1e9 << ", "
              << pm.GetL2LostCycles() << ", "
              << pm.GetL3LostCycles()
              << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;
                
    std::cout << "[INFO ] Releasing memory ..." << std::endl;
    delete column;
    delete bitvector;
    pm.Destroy();
}

void print_help(char* prog_name){
    std::cout << std::endl << "Usage: " << std::endl;
    std::cout << prog_name << " -h | [options]" << std::endl;
    std::cout << "Descriptions of <options>: " << std::endl;
    std::cout << "\t -h                     => display this message." << std::endl;
    std::cout << "\t -s <size>              => set the size of the column (number of codes). Default 16M." << std::endl;
    std::cout << "\t -t <column type>       => set the column type (the data layout). Default bs." << std::endl
              << "\t                           Acceptable column types are:" << std::endl
              << "\t                           na   : naive array, scan one tuple at a time" << std::endl
              << "\t                           navx : naive array, SIMD scan using AVX2" << std::endl
              << "\t                           avx  : bit-packed layout with AVX unpacking and scan" << std::endl
              << "\t                           hbp  : HBP layout in BitWeaving" << std::endl
              << "\t                           vbp  : VBP layout in BitWeaving" << std::endl
              << "\t                           vbp2 : VBP layout in BitWeaving (improved version)" << std::endl
              << "\t                           bs   : Byteslice layout in our SIGMOD'15 paper" << std::endl
              << "\t                           s2bs : Byteslice by flattening 2X" << std::endl
              << "\t                           s4bs : Byteslice by flattining 4X" << std::endl
              << "\t                           hs   : HybridSlice -- using option 2 of our SIGMOD'15 paper" << std::endl
              << "\t                           shs  : Smart HybridSlice -- switch between <bs> and <hs> using heuristic" << std::endl
              << "\t                           dbs  : Using 16-bit banks in ByteSlice"  << std::endl;
    std::cout << "\t -l <ratio>             => the literal in the predicate (say, v > L) is set to L = ratio * 2^k. Default 0.1." << std::endl;
    std::cout << "\t -p <predicate>         => set the predicate type. Default lt." << std::endl
              << "\t                           Acceptable predicate types are:" << std::endl
              << "\t                           lt | le | gt | ge | eq | ne , meaning" << std::endl
              << "\t                           < | <= | > | >= | == | != " << std::endl;
    std::cout << "\t -b <code width>        => set the code width (number of bits): 1~32. Default 12." << std::endl;
    std::cout << "\t -r <repetition>        => set the number of repeated experiment runs. Default 3." << std::endl;
}

void parse_arg(arg_t &arg, int &argc, char** &argv){
    int c;
    std::string s;
    while((c = getopt(argc, argv, "p:t:s:b:l:r:h")) != -1){
        switch(c){
            case 'h':
                print_help(argv[0]);
                exit(0);
            case 't':
                s = std::string(optarg);
                if(ctypeMap.find(s) == ctypeMap.end()){
                    std::cerr << "Unknown column type: " << s << std::endl;
                    exit(1);
                }
                else{
                    arg.coltype = ctypeMap[s];
					//std::cout << "coltype = " << arg.coltype << std::endl;
                }
                break;
            case 'p':
                s = std::string(optarg);
                if(cmpMap.find(s) == cmpMap.end()){
                    std::cerr << "Unknown predicate type: " << s << std::endl;
                    exit(1);
                }
                else{
                    arg.comparator = cmpMap[s];
                }
                break;
            case 's':
                arg.size = atoi(optarg);
                break;
            case 'b':
                arg.nbits = atoi(optarg);
                break;
            case 'l':
                arg.literal_ratio = atof(optarg);
                break;
            case 'r':
                arg.repeat = atoi(optarg);
                break;
        }
    }
    
    print_arg(arg);   
}

void print_arg(const arg_t& arg){
    std::cout
    << "[INFO ] column type = "  << arg.coltype  << std::endl
    << "[INFO ] predicate ="     << arg.comparator << std::endl
    << "[INFO ] table size = "   << arg.size     << std::endl
    << "[INFO ] bit width = "    << arg.nbits    << std::endl
    << "[INFO ] literal ratio = " << arg.literal_ratio << std::endl
    << "[INFO ] repeat = "       << arg.repeat   << std::endl;
}
