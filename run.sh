#!/usr/bin/expect -f

# python3 cleanup.py
declare -a arr=("courant" "dijkstra" "galois" "hardy" "noyce" "nygaard" "perlis" "shannon" "ulam" "wilkinson" "zeno" "zorn" "ada" "alonzo" "babbage" "borg" "cantor" "cray" "dahl" "eckert" "erdos" "euclid" "euler" "fermat" "fourier" "frege" "gauss" "godel" "grace" "hamming" "hilbert" "noether" "pascal" "peano" "riemann" "wilkes" "zermelo")

for i in "${arr[@]}"

# TODO: must change this directory and file name
do
	ssh "$i" "cd ~/PycharmProjects/ASL_classification/; python3 research.py 1 > /dev/null 2>&1 & exit"
done