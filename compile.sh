gcc -o ffvec.so ffvec.c -fPIC -shared \
    $(python3-config --includes) \
    $(python3-config --ldflags) \
    -fopenmp -O3 \
    $(if [ "$(uname -m)" = "x86_64" ]; then echo "-mavx -mavx2 -mfma -march=native -mtune=native"; else echo "-mfpu=neon"; fi) \
    
