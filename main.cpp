#include <iostream>

/*
 * Legacy interactive entrypoint.
 * Prefer:
 *   ./build/examples/train_transformer
 *   ./build/tests/unit_tests
 *   ./build/benches/bench_jasmine
 * See TESTING.md
 */
int main()
{
    std::cout
        << "jasmine: use CMake targets instead of this stub.\n"
        << "  cmake -S . -B build && cmake --build build -j\n"
        << "  ctest --test-dir build --output-on-failure\n"
        << "  ./build/examples/train_transformer\n"
        << "  ./build/benches/bench_jasmine\n";
    return 0;
}
