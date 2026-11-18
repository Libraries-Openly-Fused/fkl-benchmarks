// Host-only check (no GPU): fkl_loop's grey as fklReference in e19_batched_box.cu computes it (std::fma, then std::nearbyint), for every
// (R, G, B) of 8 bits, written as 2^24 bytes in the order R * 65536 + G * 256 + B.
#include <cmath>
#include <cstdio>
#include <vector>
int main() {
    std::vector<unsigned char> out(1u << 24);
    for (int r = 0; r < 256; ++r)
        for (int g = 0; g < 256; ++g)
            for (int b = 0; b < 256; ++b) {
                volatile float t = 0.587f * static_cast<float>(g);
                t = std::fma(0.299f, static_cast<float>(r), t);
                t = std::fma(0.114f, static_cast<float>(b), t);
                out[(r << 16) | (g << 8) | b] = static_cast<unsigned char>(std::nearbyint(static_cast<float>(t)));
            }
    std::FILE* f = std::fopen("fkl_rule_table.u8", "wb");
    std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
    return 0;
}
