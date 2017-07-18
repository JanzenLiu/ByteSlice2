CXX=g++
CXXFLAGS=-O3 -std=c++11 -fopenmp


default: stream

stream: stream.o system_perf_monitor.o ../libpcm.a
	$(CXX) $(CXXFLAGS) $^ -o $@
#	$(CXX) $(CXXFLAGS) stream.o system_perf_monitor.o -o stream -rdynamic ../libpcm.a

stream.o: stream.c
	$(CXX) $(CXXFLAGS) -I.. -DSTREAM_ARRAY_SIZE=50000000 -DNTIMES=20 -c $< -o $@

system_perf_monitor.o: ../system_perf_monitor.cpp ../system_perf_monitor.h
	$(CXX) $(CXXFLAGS) -c $< -o $@



clean:
	rm -rf *.o stream
