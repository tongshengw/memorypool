#!/bin/bash
numTests=(1 2 3 4 5 6 7 8 9 16)
bytesMax=(128 1 2 3 4 5 6 7 8 9 16 32)
testOps=(128 1 2 3 4 5 6 7 8 9 16 32)
echo "checking all combinations"
for a in "${testOps[@]}"; do
    for b in "${bytesMax[@]}"; do
      for c in "${numTests[@]}"; do
        if gtimeout 3 ./fuzzy -t $a -b $b -n $c > log.out; then
          :
        else
          echo "##### caused timeout with input: -t$a -b$b -n$c"
        fi
      done
    done
done