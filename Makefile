CPP=g++ --std=c++11
CPP += -O3 -march=native
#CPP += -O0 -g -fsanitize=address

LIBS=-lpthread #-lnuma

all: bin/svm

bin/svm: src/svm.cpp
	$(CPP) -o bin/svm src/svm.cpp $(LIBS)

clean:
	rm -rf bin/*


datasets: data/a8a data/a8a.t data/a8a.v data/rcv1 data/rcv1.t data/rcv1.v # data/news20_train.tsv data/rcv1_train.tsv data/rcv1_test.tsv # data/epsilon_test.tsv data/epsilon_train.tsv data/webspam_train.tsv

data/rcv1.t:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2
	bunzip2 rcv1_train.binary.bz2
	mv rcv1_train.binary data/rcv1.t

data/rcv1.v: data/rcv1
	python3 test_train_split.py 0.9 data/rcv1 data/rcv1 data/rcv1.v

data/rcv1:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2
	bunzip2 rcv1_test.binary.bz2
	mv rcv1_test.binary data/rcv1

data/a8a:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a8a
	mv a8a data/a8a

data/a8a.v:
	python3 test_train_split.py 0.9 data/a8a data/a8a data/a8a.v

data/a8a.t:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a8a.t
	mv a8a.t data/a8a.t
#
#data/news20_train.tsv:
#	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2
#	bunzip2 news20.binary.bz2
#	python3 convert2hogwild.py news20.binary data/news20 --split && rm news20.binary
#
#data/epsilon_test.tsv:
#	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.xz
#	unxz epsilon_normalized.t.xz
#	python3 convert2hogwild.py epsilon_normalized.t data/epsilon_test.tsv && rm epsilon_normalized.t
#
#data/epsilon_train.tsv:
#	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.xz
#	unxz epsilon_normalized.xz
#	python3 convert2hogwild.py epsilon_normalized data/epsilon_train.tsv && rm epsilon_normalized
#
#data/webspam_train.tsv:
#	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.xz
#	unxz webspam_wc_normalized_trigram.svm.xz
#	python3 convert2hogwild.py webspam_wc_normalized_trigram.svm data/webspam --split && rm webspam_wc_normalized_trigram.svm
#
