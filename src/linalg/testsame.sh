#!/bin/bash

# used to compare output correctness of main.c using pool vs using malloc
set -e
set -o pipefail

make pool
./pool-run > pool.out

if diff pool.out correct.out > /dev/null; then
  echo "✅ Output matches correct.out"
else
  echo "❌ Output doesn't match correct.out"
  diff pool.out correct.out
  exit 1
fi
