cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug -j
ln -sf build/release/compile_commands.json compile_commands.json