from numba import njit
from numba.core.registry import cpu_target

@njit
def f(x):
    return x + 1

# compile it
cres = cpu_target.compile_extra(f.py_func.__code__, f.signatures[0], f.signatures[0])
llvm_ir = str(cres.library.get_llvm_str())

with open("f.ll", "w") as f_out:
    f_out.write(llvm_ir)

