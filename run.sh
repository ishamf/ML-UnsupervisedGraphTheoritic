./extract.sh
g++ -o mst MST.cpp -O2 -std=c++11
g++ -o mst2graph MSTResult2graphTheoritic.cpp -O2 -std=c++11

python3 pre-cpp.py > dataset1

head -n 1 dataset1 > dataset2
echo 2 >> dataset2

./mst < dataset1 >> dataset2
./mst2graph < dataset2 > dataset3

head -n 1 dataset_m > dataset_m2
echo 2 >> dataset_m2
./mst 108 < dataset_m >> dataset_m2

head -n 1 dataset_mt > dataset_mt2
echo 2 >> dataset_mt2
./mst 108 < dataset_mt >> dataset_mt2
