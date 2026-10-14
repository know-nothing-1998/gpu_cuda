// Exercise the numerical oracle/error detector independently of a CUDA kernel.
// No replacement GEMM kernel is implemented for the user-owned naive path.
#define main gemm_cli_entry_for_tests
#include "../src/gemm.cu"
#undef main

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main() {
    try {
        const Shape shape{2, 2, 3};
        const Layout layout(shape);
        const Inputs inputs{{1, 2, 3, -1, 0, 2}, {1, 4, 2, 5, 3, 6}};
        const auto ref = cpu_reference(shape, layout, inputs);
        require(ref.values == std::vector<double>({14, 32, 5, 8}), "row-major oracle/layout mismatch");
        require(compare_cpu({14, 32, 5, 8}, ref).valid(), "correct result rejected");
        require(compare_cpu({14, 5, 32, 8}, ref).failed == 2, "transposed output accepted");
        require(compare_cpu({14, 32, 5, 9}, ref).failed == 1, "wrong output accepted");

        const Reference zeros{{0,1}, {0,0}, "zero"};
        const auto exact = compare_cpu({0,0}, zeros);
        require(exact.valid() && exact.relative_frob == 0, "zero norm protection failed");
        require(std::isfinite(compare_cpu({0.01f,0}, zeros).relative_frob), "zero norm division invalid");
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const float inf = std::numeric_limits<float>::infinity();
        const Reference sample{{0}, {14}, "sample"};
        require(compare_cpu({14,nan}, sample).nonfinite == 1, "unsampled NaN escaped full scan");
        require(!compare_cpu({inf,32,5,8}, ref).valid(), "infinite output accepted");
        require(!compare_cpu({nan,nan,nan,nan}, ref).valid(), "unwritten output accepted");
        require(compare_cpu({0.0009f,0}, zeros).valid(), "absolute tolerance rejected");
        require(!compare_cpu({0.0011f,0}, zeros).valid(), "absolute tolerance too wide");
        bool overflow = false;
        try { checked_mul(std::numeric_limits<size_t>::max(), 4); }
        catch (const std::runtime_error&) { overflow = true; }
        require(overflow, "allocation multiplication overflow missed");
        overflow = false;
        try { checked_add(std::numeric_limits<size_t>::max(), 1); }
        catch (const std::runtime_error&) { overflow = true; }
        require(overflow, "allocation addition overflow missed");
        const auto first = generate(shape, layout, 42, "random");
        const auto again = generate(shape, layout, 42, "random");
        require(first.a == again.a && first.b == again.b, "input generation is not reproducible");
        require(first.a != generate(shape, layout, 43, "random").a, "seed has no effect");
        Timing timing{{4, 1, 3, 2}};
        timing.summarize();
        require(timing.median == 2.5 && timing.minimum == 1 && timing.maximum == 4, "timing summary incorrect");
        std::cout << "harness tests passed: layout, corruption, NaN/Inf, tolerance, overflow, RNG, timing\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
