mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=DEBUG -DBUILD_EXAMPLES=ON ..
make -j8
