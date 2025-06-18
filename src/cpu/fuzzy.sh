#!/bin/bash
numTests=(1)
bytesMax=(128)
testOps=(128)
seeds=($(seq 0 2000))
echo "checking all combinations"
for a in "${testOps[@]}"; do
    for b in "${bytesMax[@]}"; do
      for c in "${numTests[@]}"; do
        for s in "${seeds[@]}"; do
          if timeout 3 ./fuzzy-cpu -t $a -b $b -n $c -s $s > log.out; then
          :
        else
            echo "##### caused timeout with input: -t$a -b$b -n$c -s$s"
          fi
        done
      done
    done
done