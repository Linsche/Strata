// tests/hip/qsa_pattn_flash.cpp - STRATA_PATTN_FLASH's kernel (qsa_pattn_flash_batch, both query precisions) against
// qsa_decode_attn_batch, the prompt path's attention without it, on k8v4 pools: random codes and scales, selections of
// 4-cell blocks that neighbouring queries mostly share, some short selections (early queries), and optionally pages
// the KV streaming left out (-1).  Fails when the relative RMS error passes 1e-5 (FP32 query) or 2e-4 (FP16 query);
// prints the time of both.  Args: [context] [queries] [masked 0|1]; no args runs the default set.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_pattn_flash.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>
namespace K = strata::kernels;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)
static uint16_t f2h(float f) { __half h = __float2half(f); uint16_t u; std::memcpy(&u, &h, 2); return u; }
static int run(int ctx, int nq, int masked) {
    const int reps = 5;
    K::QsaShapes s = K::qsa_real_shapes();
    const int kvh = s.n_head_kv, hd = s.head_dim, nh = s.n_head, ps = s.page_size;
    const int cap = (int) K::qsa_selection_width(ctx, s);
    const int pages = (ctx + ps - 1) / ps;
    std::mt19937 rng(7);
    std::vector<int32_t> tab(pages); for (int i = 0; i < pages; ++i) tab[i] = (masked && i % 97 == 5) ? -1 : i;
    const size_t rows = (size_t) pages * ps * kvh;
    std::vector<int8_t> hk(rows * hd); for (auto& v : hk) v = (int8_t) (rng() % 255 - 127);
    std::vector<uint16_t> hks(rows * 4); for (auto& v : hks) v = f2h(0.002f + 0.01f * (rng() % 1000) / 1000.0f);
    const size_t vb = 144;
    std::vector<uint8_t> hv(rows * vb);
    for (size_t r = 0; r < rows; ++r) for (int b = 0; b < 8; ++b) {
        uint8_t* blk = hv.data() + r * vb + b * 18; uint16_t d = f2h(0.05f + 0.1f * (rng() % 1000) / 1000.0f);
        std::memcpy(blk, &d, 2); for (int j = 0; j < 16; ++j) blk[2 + j] = (uint8_t) rng();
    }
    int32_t* dtab; int8_t* dk; uint16_t* dks; uint8_t* dv;
    CK(cudaMalloc(&dtab, pages * 4)); CK(cudaMalloc(&dk, hk.size())); CK(cudaMalloc(&dks, hks.size() * 2)); CK(cudaMalloc(&dv, hv.size()));
    CK(cudaMemcpy(dtab, tab.data(), pages * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dk, hk.data(), hk.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dks, hks.data(), hks.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dv, hv.data(), hv.size(), cudaMemcpyHostToDevice));
    // selections: sorted cells, neighbouring queries share most blocks; widths vary (some queries early in the prompt)
    const int nblk = ctx / 4, want = cap / 4;
    std::vector<int32_t> base(nblk); for (int b = 0; b < nblk; ++b) base[b] = b;
    const int nnew = std::min(64, nblk / 2);   // the newest blocks every query selects
    std::shuffle(base.begin(), base.end() - nnew, rng);
    std::vector<int32_t> ids((size_t) nq * cap, 0), steps((size_t) nq * K::kStepCount, 0);
    for (int z = 0; z < nq; ++z) {
        const int w = (z % 17 == 3) ? 1 + (int) (rng() % cap) : cap;
        std::vector<char> on(nblk, 0); int n = 0;
        for (int b = nblk - nnew; b < nblk; ++b) { on[b] = 1; ++n; }
        for (int i = (z / 8) % 32; n < want && n < nblk; ++i) { int b = base[i % (nblk - nnew)]; if (!on[b]) { on[b] = 1; ++n; } }
        int k = 0;
        for (int b = 0; b < nblk && k < cap; ++b) if (on[b]) for (int c = 0; c < 4 && k < cap; ++c) ids[(size_t) z * cap + k++] = b * 4 + c;
        steps[(size_t) z * K::kStepCount + K::kStepPos] = ctx - 1;
        steps[(size_t) z * K::kStepCount + K::kStepWidth] = w;
    }
    std::normal_distribution<float> g(0, 1);
    std::vector<float> hq((size_t) nq * nh * hd); for (auto& v : hq) v = g(rng);
    int32_t *dids, *dst; float *dq, *dscr, *o1, *o2;
    const int batch = 32;
    CK(cudaMalloc(&dids, ids.size() * 4)); CK(cudaMalloc(&dst, steps.size() * 4)); CK(cudaMalloc(&dq, hq.size() * 4));
    CK(cudaMalloc(&dscr, (size_t) K::qsa_decode_attn_scratch_floats(cap, s) * batch * 4));
    CK(cudaMalloc(&o1, hq.size() * 4)); CK(cudaMalloc(&o2, hq.size() * 4));
    CK(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dst, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(o2, 0xFF, hq.size() * 4));
    K::QsaAttnPools p{}; p.k_q = dk; p.k_scale = dks; p.v_q4 = dv; p.page_table = dtab;
    auto ref = [&] {
        for (int t0 = 0; t0 < nq; t0 += batch)
            K::qsa_decode_attn_batch(dq + (size_t) t0 * nh * hd, p, dids + (size_t) t0 * cap, dst + (size_t) t0 * K::kStepCount,
                                     cap, s, dscr, o1 + (size_t) t0 * nh * hd, std::min(batch, nq - t0), nullptr);
    };
    int fails = 0;
    for (const bool q16 : {false, true}) {
    CK(cudaMemset(o2, 0xFF, hq.size() * 4));
    auto neu = [&] { if (!K::qsa_pattn_flash_batch(dq, p, dids, dst, cap, s, o2, nq, q16, nullptr)) { std::printf("refused\n"); std::exit(1); } };
    ref(); neu(); CK(cudaDeviceSynchronize());
    std::vector<float> a(hq.size()), b(hq.size());
    CK(cudaMemcpy(a.data(), o1, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), o2, b.size() * 4, cudaMemcpyDeviceToHost));
    double maxd = 0, maxa = 0, sum2 = 0, sumd2 = 0; size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(b[i])) { ++bad; continue; }
        const double d = std::fabs((double) a[i] - b[i]); maxd = std::max(maxd, d); maxa = std::max(maxa, (double) std::fabs(a[i]));
        sum2 += (double) a[i] * a[i]; sumd2 += d * d;
    }
    auto timeit = [&](auto f) {
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0); for (int r = 0; r < reps; ++r) f(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1); return ms / reps;
    };
    const float tr = timeit(ref), tn = timeit(neu);
    const double rel = std::sqrt(sumd2 / std::max(sum2, 1e-30));
    const bool ok = bad == 0 && rel < (q16 ? 2e-4 : 1e-5);
    fails += !ok;
    std::printf("%s ctx %d cap %d nq %d masked %d q%s | max |ref| %.4g  max diff %.3g  rel rms %.3g  non-finite %zu | decode %.2f ms (%.2f us/q)  flash %.2f ms (%.2f us/q)  x%.2f\n",
                ok ? "ok  " : "FAIL", ctx, cap, nq, masked, q16 ? "16" : "32", maxa, maxd, rel, bad, tr, 1000 * tr / nq, tn,
                1000 * tn / nq, tr / tn);
    }
    cudaFree(dtab); cudaFree(dk); cudaFree(dks); cudaFree(dv); cudaFree(dids); cudaFree(dst); cudaFree(dq); cudaFree(dscr);
    cudaFree(o1); cudaFree(o2);
    return fails;
}

int main(int argc, char** argv) {
    if (argc > 1) return run(std::atoi(argv[1]), argc > 2 ? std::atoi(argv[2]) : 1024, argc > 3 ? std::atoi(argv[3]) : 0) ? 1 : 0;
    int fails = 0;
    fails += run(13568, 1024, 0);
    fails += run(13568, 1024, 1);
    fails += run(4096, 512, 1);
    fails += run(1500, 512, 0);    // context below the selection width: the identity selection
    fails += run(64, 64, 0);
    std::printf("qsa_pattn_flash: %s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
