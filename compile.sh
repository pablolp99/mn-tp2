echo "Cleaning build directory..."
rm -r ./build
mkdir -p ./build
cd build
echo "Running CMake"
cmake ..
echo "Compiling"
make
