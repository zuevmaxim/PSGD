CPP=g++ --std=c++11
CPP += -O3 -march=native
#CPP += -O0 -g -fsanitize=address

LIBS=-lpthread -lnuma

all: bin/svm

dirs:
	mkdir -p "bin"
	mkdir -p "data"

bin/svm: dirs src/svm.cpp
	$(CPP) -o bin/svm src/svm.cpp $(LIBS)

clean:
	rm -rf bin/*


datasets: dirs data/rcv1 data/rcv1.t
data/rcv1.t:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2
	bunzip2 rcv1_train.binary.bz2
	mv rcv1_train.binary data/rcv1.t

#data/rcv1.v: data/rcv1
#	python3 test_train_split.py 0.9 data/rcv1 data/rcv1 data/rcv1.v

data/rcv1:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2
	bunzip2 rcv1_test.binary.bz2
	mv rcv1_test.binary data/rcv1
