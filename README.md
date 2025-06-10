# memorypool

## How to build
### building
```
mkdir build && cd build
cmake ..
make all
```
### tests
```
ctest
```

## FIX
- fuzzy-gpu fails with 4 threads, 10 ops, 3 seed
- somehow bug when numtests is 128, look into this

## todo
- add fuzzy testing to automatic tests
