#!/bin/bash
numbers=(1 2 3 4 5 6 7 8 9 16)
echo "checking all combinations"
for a in "${numbers[@]}"; do
    for b in "${numbers[@]}"; do
      for c in "${numbers[@]}"; do
        if gtimeout 3 ./fuzzy -t $a -b $b -n $c > log.out; then
          :
        else
          echo "##### caused timeout with input: -t$a -b$b -n$c"
        fi
      done
    done
done