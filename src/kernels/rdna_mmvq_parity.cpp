// src/kernels/rdna_mmvq_parity.cpp - the opt-in RDNA GEMVs (rdna_mmvq.cu: STRATA_RDNA_Q6K=1) against native_mmvq on
// the same weights and the same q8_1 input (GPU, synthetic, no model).  The RDNA kernels sum in another order, so the
// check is a tolerance, not bitwise: max |rdna - native| / max |native| per case.  Covers Q4_K, Q5_K, Q6_K, IQ4_XS and
// IQ3_S at 1..8 columns, single and paired (STRATA_RDNA_PAIR's second matrix).  Q6_K's 210-byte superblocks put every
// other block's dword loads (ld32u) on a 2-byte boundary, so a card that cannot take unaligned dword loads fails here.
//
//   rdna_mmvq_parity    exit 0: every case within tolerance; 1: a mismatch; 77: a build without the RDNA kernels
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace strata::kernels {
bool rdna_kq_mmvq2(int type, const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream,
                   const void* w2, float* y2, int n_out2);
bool rdna_q6k_mmvq2(const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream,
                    const void* w2, float* y2, int n_out2);
}  // namespace strata::kernels

namespace K = strata::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Fmt { int type; const char* name; int block; int d_off; bool dmin; };
// block bytes and where the fp16 scale sits (every other byte is a code, a sub-scale or a grid index: any value is valid)
const Fmt kFmts[] = {
    {12, "Q4_K", 144, 0, true},
    {13, "Q5_K", 176, 0, true},
    {14, "Q6_K", 210, 208, false},
    {23, "IQ4_XS", 136, 0, false},
    {21, "IQ3_S", 110, 0, false},
};

uint16_t f16(float v) {   // small, normal values only
    uint32_t b;
    std::memcpy(&b, &v, 4);
    const uint32_t sign = (b >> 16) & 0x8000u;
    const int e = (int) ((b >> 23) & 0xff) - 127 + 15;
    return (uint16_t) (sign | ((uint32_t) e << 10) | ((b >> 13) & 0x3ffu));
}

std::vector<uint8_t> random_blocks(const Fmt& f, int n_in, int n_out, std::mt19937& rng) {
    const size_t nb = (size_t) n_out * (n_in / 256);
    std::vector<uint8_t> w(nb * f.block);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.002f, 0.02f);
    for (auto& v : w) v = (uint8_t) byte(rng);
    for (size_t i = 0; i < nb; ++i) {
        uint8_t* b = w.data() + i * f.block;
        const uint16_t d = f16(sc(rng));
        std::memcpy(b + f.d_off, &d, 2);
        if (f.dmin) {
            const uint16_t m = f16(sc(rng));
            std::memcpy(b + f.d_off + 2, &m, 2);
        }
    }
    return w;
}
}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { std::printf("no GPU\n"); return 77; }
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, dev), "props");
    std::printf("rdna_mmvq_parity on %s\n", prop.name);
    setenv("STRATA_RDNA_KQ_ALL", "1", 1);
    cudaStream_t st = nullptr;   // native_mmvq requires an explicit stream
    ck(cudaStreamCreate(&st), "stream");   // Q4_K / Q5_K below 4 columns too (the kernels' bench switch)

    const int n_in = 4096, n_out = 72, n_out2 = 40;
    std::mt19937 rng(11);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> hx((size_t) n_in * 8);
    for (auto& v : hx) v = g(rng);
    float* d_x = nullptr;
    void* d_xq = nullptr;
    ck(cudaMalloc(&d_x, hx.size() * 4), "m x");
    ck(cudaMalloc(&d_xq, (size_t) 8 * (n_in / 32) * 36), "m xq");
    ck(cudaMemcpy(d_x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "c x");
    K::native_quantize_q8_1(d_x, d_xq, n_in, 8, st);

    float *d_ref = nullptr, *d_ref2 = nullptr, *d_y = nullptr, *d_y2 = nullptr;
    ck(cudaMalloc(&d_ref, (size_t) 8 * n_out * 4), "m ref");
    ck(cudaMalloc(&d_ref2, (size_t) 8 * n_out2 * 4), "m ref2");
    ck(cudaMalloc(&d_y, (size_t) 8 * n_out * 4), "m y");
    ck(cudaMalloc(&d_y2, (size_t) 8 * n_out2 * 4), "m y2");

    const double tol = 1e-4;   // relative to the output's scale; fp32 sums of 4096 products in another order
    int bad = 0, ran = 0;
    for (const Fmt& f : kFmts) {
        const std::vector<uint8_t> hw = random_blocks(f, n_in, n_out, rng), hw2 = random_blocks(f, n_in, n_out2, rng);
        void *d_w = nullptr, *d_w2 = nullptr;
        ck(cudaMalloc(&d_w, hw.size()), "m w");
        ck(cudaMalloc(&d_w2, hw2.size()), "m w2");
        ck(cudaMemcpy(d_w, hw.data(), hw.size(), cudaMemcpyHostToDevice), "c w");
        ck(cudaMemcpy(d_w2, hw2.data(), hw2.size(), cudaMemcpyHostToDevice), "c w2");
        for (int pair = 0; pair < 2; ++pair)
            for (int nc = 1; nc <= 8; ++nc) {
                K::native_mmvq(f.type, d_w, d_xq, d_ref, n_in, n_out, nc, st);
                if (pair) K::native_mmvq(f.type, d_w2, d_xq, d_ref2, n_in, n_out2, nc, st);
                ck(cudaMemsetAsync(d_y, 0xff, (size_t) 8 * n_out * 4, st), "z y");
                ck(cudaMemsetAsync(d_y2, 0xff, (size_t) 8 * n_out2 * 4, st), "z y2");
                const bool ok = f.type == 14
                    ? K::rdna_q6k_mmvq2(d_w, d_xq, d_y, n_in, n_out, nc, st, pair ? d_w2 : nullptr, d_y2, n_out2)
                    : K::rdna_kq_mmvq2(f.type, d_w, d_xq, d_y, n_in, n_out, nc, st, pair ? d_w2 : nullptr, d_y2,
                                       n_out2);
                ck(cudaDeviceSynchronize(), f.name);
                if (!ok) {
                    if (f.type == 14 && nc == 1 && pair == 0) {
                        std::printf("the RDNA kernels are not in this build\n");
                        return 77;
                    }
                    continue;   // the kernel declines this shape: native_mmvq runs, nothing to compare
                }
                ++ran;
                double worst = 0.0, scale = 0.0;
                const auto cmp = [&](float* dr, float* dy, int rows) {
                    std::vector<float> r((size_t) nc * rows), y(r.size());
                    ck(cudaMemcpy(r.data(), dr, r.size() * 4, cudaMemcpyDeviceToHost), "c r");
                    ck(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost), "c y");
                    for (size_t i = 0; i < r.size(); ++i) {
                        scale = std::max(scale, (double) std::fabs(r[i]));
                        const double d = std::isfinite(y[i]) ? std::fabs((double) y[i] - r[i]) : INFINITY;
                        worst = std::max(worst, d);
                    }
                };
                cmp(d_ref, d_y, n_out);
                if (pair) cmp(d_ref2, d_y2, n_out2);
                const double rel = scale > 0 ? worst / scale : worst;
                const bool pass = rel <= tol;
                if (!pass) ++bad;
                if (!pass || nc == 1 || nc == 8)
                    std::printf("  %-6s %s ncols %d: max rel diff %.2e %s\n", f.name, pair ? "pair  " : "single", nc, rel,
                                pass ? "ok" : "MISMATCH");
            }
        cudaFree(d_w);
        cudaFree(d_w2);
    }
    std::printf("%d cases, %d mismatches\n%s\n", ran, bad, bad ? "FAIL" : "ok: the RDNA GEMVs match native_mmvq");
    return bad ? 1 : 0;
}
