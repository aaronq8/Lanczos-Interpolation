cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j
ln -sf build/release/compile_commands.json compile_commands.json