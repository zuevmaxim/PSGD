CPP=g++ --std=c++11
CPP += -O3 -march=native
#CPP += -O0 -g -fsanitize=address

ifeq (, $(shell which numactl))
else
 NUMA_LIB=-lnuma
endif
LIBS=-lpthread $(NUMA_LIB)

all: bin/svm bin/analysis

bin:
	mkdir -p "bin"
data:
	mkdir -p "data"

bin/svm: bin src/svm.cpp
	$(CPP) -o bin/svm src/svm.cpp $(LIBS)

clean:
	rm -rf bin/*

bin/analysis: bin src/analysis.cpp
	$(CPP) -o bin/analysis src/analysis.cpp


datasets: data rcv1 news20 url kdda

rcv1: data/rcv1 data/rcv1.t
data/rcv1:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2
	bunzip2 rcv1_test.binary.bz2
	mv rcv1_test.binary data/rcv1
data/rcv1.t:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2
	bunzip2 rcv1_train.binary.bz2
	mv rcv1_train.binary data/rcv1.t
#data/rcv1.v: data/rcv1
#	python3 test_train_split.py 0.9 data/rcv1 data/rcv1 data/rcv1.v

news20: data/news20
data/news20:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2
	bunzip2 news20.binary.bz2
	python3 test_train_split.py 0.8 news20.binary data/news20 data/news20.t && rm news20.binary

url: data/url
data/url:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/url_combined.bz2
	bunzip2 url_combined.bz2
	python3 test_train_split.py 0.8 url_combined data/url data/url.t && rm url_combined

kdda: data/kdda data/kdda.t
data/kdda:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdda.bz2
	bunzip2 kdda.bz2
	mv kdda data/kdda
data/kdda.t:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdda.t.bz2
	bunzip2 kdda.t.bz2
	mv kdda.t data/kdda.t
