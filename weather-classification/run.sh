#!/usr/bin/expect -f

cp main.py research.py

count=1
while [ -d "save${count}" ]
do
  let "count++"
done
mkdir "save${count}"


#declare -a arr=("courant" "dijkstra" "galois" "hardy" "noyce" "nygaard" "perlis" "shannon" "ulam" "wilkinson" "zeno" "zorn" "ada" "alonzo" "babbage" "borg" "cantor" "cray" "dahl" "eckert" "erdos" "euclid" "euler" "fermat" "fourier" "frege" "gauss" "godel" "grace" "hamming" "hilbert" "noether" "pascal" "peano" "riemann" "wilkes" "zermelo")
declare -a arr=("courant" "zermelo")

for i in "${arr[@]}"
do
	ssh "${i}.hbg.psu.edu" "cd ~/Documents/GitHub/SemanticSegmentationOfLowVisibilityRoadways/weather-classification/; python3 research.py ~/Documents/weather_dataset/resized/ save${count}/ > /dev/null 2>&1 & exit"
done