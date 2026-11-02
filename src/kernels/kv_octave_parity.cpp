// src/kernels/kv_octave_parity.cpp - the 3-bit Lloyd-Max KV format against a host reference (GPU, synthetic,
// no model): the device bytes must equal the host reference bit for bit, and the error it leaves on rotated
// vectors is reported next to a 4-bit uniform grid's on the same data.  Section 6 runs the decode attention's chunk
// pass on oct3 pools with and without the exact fp16 window through both chunk kernels (attn_chunk_kernel and
// attn_chunk_kernel_pre75, the one gfx103x and sm_6x run) and requires the outputs to be identical bit for bit.
#include "strata/kernels/kv_octave.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_stream.hpp"   // KvHostPools
#include "strata/kernels/kv_exact_window.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void rel_err(const char* label, const std::vector<float>& a, const std::vector<float>& b, double* mse_out) {
    double se = 0, sx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - (double) b[i];
        se += d * d;
        sx += (double) a[i] * (double) a[i];
    }
    const double mse = se / (double) a.size();
    if (mse_out) *mse_out = mse;
    std::printf("  %-22s MSE/coord %.6f   rel L2 %.5f\n", label, mse, std::sqrt(se / std::max(1e-30, sx)));
}

// A 4-bit uniform grid on the same rotated vectors: per-32 amax, q4_0's d = amax/-8 (the codebook q4_0 ships).
void q4_uniform_ref(const std::vector<float>& x, std::vector<float>& out) {
    out.assign(x.size(), 0.0f);
    for (size_t g = 0; g < x.size() / 32; ++g) {
        const float* p = &x[g * 32];
        float amax = 0.0f, mval = 0.0f;
        for (int i = 0; i < 32; ++i)
            if (std::fabs(p[i]) > amax || (std::fabs(p[i]) == amax && p[i] > mval)) { amax = std::fabs(p[i]); mval = p[i]; }
        const float d = mval / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        for (int i = 0; i < 32; ++i) {
            int q = (int) std::trunc(p[i] * id + 8.5f);
            q = q < 0 ? 0 : (q > 15 ? 15 : q);
            out[g * 32 + i] = (float) (q - 8) * d;
        }
    }
}
}  // namespace

int main() {
    const int groups = 4096;                       // 131,072 values
    const size_t n = (size_t) groups * K::QK_OCT;
    std::mt19937 rng(7);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x(n);
    for (auto& v : x) v = gauss(rng);
    for (size_t i = 0; i < n; i += 97) x[i] *= 8.0f;   // KV-like outliers
    std::vector<float> rot(n);

    float* d_x = nullptr;
    float* d_rot = nullptr;
    float* d_back = nullptr;
    float* d_dec = nullptr;
    uint8_t* d_enc = nullptr;
    const size_t enc_bytes = (size_t) groups * sizeof(K::block_oct3);
    ck(cudaMalloc(&d_x, n * 4), "malloc x");
    ck(cudaMalloc(&d_rot, n * 4), "malloc rot");
    ck(cudaMalloc(&d_back, n * 4), "malloc back");
    ck(cudaMalloc(&d_dec, n * 4), "malloc dec");
    ck(cudaMalloc(&d_enc, enc_bytes), "malloc enc");
    ck(cudaMemcpy(d_x, x.data(), n * 4, cudaMemcpyHostToDevice), "copy x");

    // ---- 1. device vs host reference, byte for byte -------------------------------------------------
    K::kv_oct_quantize(d_x, d_enc, groups, nullptr);
    std::vector<uint8_t> enc(enc_bytes);
    ck(cudaMemcpy(enc.data(), d_enc, enc_bytes, cudaMemcpyDeviceToHost), "copy enc");
    int enc_mismatch = 0, dec_mismatch = 0;
    K::kv_oct_dequantize(d_enc, d_dec, groups, nullptr);
    std::vector<float> dec(n);
    ck(cudaMemcpy(dec.data(), d_dec, n * 4, cudaMemcpyDeviceToHost), "copy dec");
    for (int gi = 0; gi < groups; ++gi) {
        K::block_oct3 ref{};
        K::kv_oct_quantize_group_host(&x[(size_t) gi * K::QK_OCT], ref);
        if (std::memcmp(&ref, &enc[(size_t) gi * sizeof(K::block_oct3)], sizeof(ref)) != 0) ++enc_mismatch;
        float href[K::QK_OCT];
        K::kv_oct_dequantize_group_host(ref, href);
        for (int i = 0; i < K::QK_OCT; ++i)
            if (std::memcmp(&href[i], &dec[(size_t) gi * K::QK_OCT + i], sizeof(float)) != 0) ++dec_mismatch;
    }
    std::printf("host reference: %d/%d groups differ in the packed bytes, %d/%zu dequantized values differ\n",
                enc_mismatch, groups, dec_mismatch, n);

    // ---- 2. the error on the vectors themselves (unrotated domain) ----------------------------------
    double mse_oct = 0, mse_q4 = 0;
    std::vector<float> q4;
    q4_uniform_ref(x, q4);
    std::printf("\nunrotated data (MSE is what the format costs, before rotation helps):\n");
    rel_err("oct3 (3-bit, Lloyd-Max)", x, dec, &mse_oct);
    rel_err("q4_0 (4-bit, uniform)", x, q4, &mse_q4);

    // ---- 3. what the KV path actually stores: rotate first ------------------------------------------
    K::fwht256_cuda(d_x, d_rot, (int64_t) (n / 256), nullptr);   // rows of 256 = one head
    K::kv_oct_quantize(d_rot, d_enc, groups, nullptr);
    K::kv_oct_dequantize(d_enc, d_dec, groups, nullptr);
    std::vector<float> rot_host(n), rot_dec(n);
    ck(cudaMemcpy(rot_host.data(), d_rot, n * 4, cudaMemcpyDeviceToHost), "copy rot");
    ck(cudaMemcpy(rot_dec.data(), d_dec, n * 4, cudaMemcpyDeviceToHost), "copy rot dec");
    K::fwht256_cuda(d_dec, d_back, (int64_t) (n / 256), nullptr);   // H is its own inverse
    std::vector<float> back(n);
    ck(cudaMemcpy(back.data(), d_back, n * 4, cudaMemcpyDeviceToHost), "copy back");

    std::vector<float> rot4;
    q4_uniform_ref(rot_host, rot4);
    std::printf("\nrotated domain (the frame K and V are stored in):\n");
    double mse_rot = 0, mse_rot4 = 0, mse_trip = 0;
    rel_err("oct3 (3-bit, Lloyd-Max)", rot_host, rot_dec, &mse_rot);
    rel_err("q4_0 (4-bit, uniform)", rot_host, rot4, &mse_rot4);
    std::printf("  oct3 is %.2fx the 4-bit MSE at %.0f%% of its bytes\n", mse_rot / std::max(1e-30, mse_rot4),
                100.0 * 112.0 / 144.0);
    rel_err("oct3 full round trip", x, back, &mse_trip);

    // ---- 4. what it costs in bytes ------------------------------------------------------------------
    const K::QsaShapes s = K::qsa_real_shapes();
    const uint64_t oct_head = K::kv_oct_bytes_per_head((int) s.head_dim);
    std::printf("\nbytes: %llu per head per K or V (%d values), %llu per cell (K+V, both KV heads)\n",
                (unsigned long long) oct_head, (int) s.head_dim, (unsigned long long) K::kv_oct_bytes_per_cell(s));
    std::printf("       q4_0 144/576, k8v4 528/816, int8 528/1056 -> oct3 is %.0f%% of k8v4's cell\n",
                100.0 * (double) K::kv_oct_bytes_per_cell(s) / 816.0);

    // ---- 5. engine-shaped storage: page table + host copy, then gather by ids ------------------------
    int oct_bad = 0;
    {
        const int pages = 4, page_size = (int) s.page_size, cells = pages * page_size;
        const int rows = cells * (int) s.n_head_kv;
        const int bph = (int) (s.head_dim / K::QK_OCT);
        const size_t vals = (size_t) rows * s.head_dim;
        const size_t pool_bytes = (size_t) rows * bph * sizeof(K::block_oct3);
        std::vector<float> src(vals);
        for (auto& v : src) v = gauss(rng);
        float *d_src = nullptr, *d_rot = nullptr, *d_rotv = nullptr;
        ck(cudaMalloc(&d_src, vals * 4), "m src");
        ck(cudaMalloc(&d_rot, vals * 4), "m rot");
        ck(cudaMalloc(&d_rotv, vals * 4), "m rotv");
        ck(cudaMemcpy(d_src, src.data(), vals * 4, cudaMemcpyHostToDevice), "c");
        K::fwht256_cuda(d_src, d_rot, (int64_t) rows, nullptr);
        ck(cudaMemcpy(d_rotv, d_rot, vals * 4, cudaMemcpyDeviceToDevice), "c");
        std::vector<float> rotv(vals);
        ck(cudaMemcpy(rotv.data(), d_rot, vals * 4, cudaMemcpyDeviceToHost), "c");

        std::vector<int32_t> tab(pages);
        for (int pgi = 0; pgi < pages; ++pgi) tab[pgi] = pgi;      // every page resident in its own slot
        int32_t* d_tab = nullptr;
        ck(cudaMalloc(&d_tab, pages * 4), "m tab");
        ck(cudaMemcpy(d_tab, tab.data(), pages * 4, cudaMemcpyHostToDevice), "c tab");
        uint8_t *d_k = nullptr, *d_v = nullptr, *h_k = nullptr, *h_v = nullptr;
        ck(cudaMalloc(&d_k, pool_bytes), "m k");
        ck(cudaMalloc(&d_v, pool_bytes), "m v");
        ck(cudaMalloc(&h_k, pool_bytes), "m hk");
        ck(cudaMalloc(&h_v, pool_bytes), "m hv");
        K::KvHostPools hostp{};
        hostp.k_q4 = h_k;
        hostp.v_q4 = h_v;
        K::kv_append_oct(d_k, d_v, d_tab, 0, cells, d_rot, d_rotv, s, nullptr, &hostp, nullptr);
        ck(cudaDeviceSynchronize(), "append");

        const int ids[4] = {0, 5, 9, cells - 1};
        int32_t* d_ids = nullptr;
        ck(cudaMalloc(&d_ids, sizeof(ids)), "m ids");
        ck(cudaMemcpy(d_ids, ids, sizeof(ids), cudaMemcpyHostToDevice), "c ids");
        std::vector<int32_t> step(4, 0);
        step[K::kStepWidth] = 4;
        int32_t* d_step = nullptr;
        ck(cudaMalloc(&d_step, 16), "m step");
        ck(cudaMemcpy(d_step, step.data(), 16, cudaMemcpyHostToDevice), "c step");
        const size_t scr = (size_t) 4 * s.n_head_kv * s.head_dim;
        uint16_t *d_ks = nullptr, *d_vs = nullptr;
        ck(cudaMalloc(&d_ks, scr * 2), "m ks");
        ck(cudaMalloc(&d_vs, scr * 2), "m vs");
        K::kv_gather_oct_step(d_k, d_v, d_tab, d_ids, d_step, 4, s, d_ks, d_vs, nullptr);
        ck(cudaDeviceSynchronize(), "gather");
        std::vector<uint16_t> ks(scr), vs(scr);
        ck(cudaMemcpy(ks.data(), d_ks, scr * 2, cudaMemcpyDeviceToHost), "c ks");
        ck(cudaMemcpy(vs.data(), d_vs, scr * 2, cudaMemcpyDeviceToHost), "c vs");

        for (int i = 0; i < 4; ++i)
            for (int h = 0; h < (int) s.n_head_kv; ++h)
                for (int b = 0; b < bph; ++b) {
                    K::block_oct3 ref{};
                    K::kv_oct_quantize_group_host(
                        &rotv[((size_t) ids[i] * s.n_head_kv + h) * s.head_dim + (size_t) b * K::QK_OCT], ref);
                    float dec[K::QK_OCT];
                    K::kv_oct_dequantize_group_host(ref, dec);
                    for (int t = 0; t < K::QK_OCT; ++t) {
                        const uint16_t want = K::f16_from_f32(dec[t]);
                        const size_t at = (size_t) i * s.n_head_kv * s.head_dim + (size_t) h * s.head_dim +
                                          (size_t) b * K::QK_OCT + t;
                        if (ks[at] != want) ++oct_bad;
                        if (vs[at] != want) ++oct_bad;   // the test feeds the same pattern as K and V
                    }
                }
        std::printf("\nengine-shaped storage (4 pages x 4 cells, page table + host copy): %s (%d mismatches)\n",
                    oct_bad ? "MISMATCH" : "gather == reference", oct_bad);
    }

    // ---- 6. decode attention over oct3 pools: attn_chunk_kernel against attn_chunk_kernel_pre75, bit for bit ---
    // With and without the exact window, and with one page masked (-1), so the pre75 kernel's grouped and its
    // one-cell-at-a-time paths both meet window cells.  Three queries with different selections and widths.
    int attn_bad = 0;
    {
        const int page_size = (int) s.page_size, cells = 320, pages = (cells + page_size - 1) / page_size;
        const int kvh = (int) s.n_head_kv, hd = (int) s.head_dim, nh = (int) s.n_head;
        const size_t vals = (size_t) cells * kvh * hd;
        const int bph = hd / K::QK_OCT;
        const size_t pool_bytes = (size_t) pages * page_size * kvh * bph * sizeof(K::block_oct3);
        std::vector<float> hk(vals), hv(vals);
        for (auto& v : hk) v = gauss(rng);
        for (auto& v : hv) v = gauss(rng);
        float *d_kf = nullptr, *d_vf = nullptr;
        ck(cudaMalloc(&d_kf, vals * 4), "m kf");
        ck(cudaMalloc(&d_vf, vals * 4), "m vf");
        ck(cudaMemcpy(d_kf, hk.data(), vals * 4, cudaMemcpyHostToDevice), "c kf");
        ck(cudaMemcpy(d_vf, hv.data(), vals * 4, cudaMemcpyHostToDevice), "c vf");
        std::vector<int32_t> tab(pages);
        for (int pgi = 0; pgi < pages; ++pgi) tab[pgi] = pgi;
        int32_t *d_tab = nullptr, *d_tab_m = nullptr;
        ck(cudaMalloc(&d_tab, pages * 4), "m tab");
        ck(cudaMalloc(&d_tab_m, pages * 4), "m tabm");
        ck(cudaMemcpy(d_tab, tab.data(), pages * 4, cudaMemcpyHostToDevice), "c tab");
        tab[pages - 3] = -1;   // a page the streaming could not make resident: inside the window's tail
        ck(cudaMemcpy(d_tab_m, tab.data(), pages * 4, cudaMemcpyHostToDevice), "c tabm");
        uint8_t *d_k = nullptr, *d_v = nullptr;
        ck(cudaMalloc(&d_k, pool_bytes), "m k");
        ck(cudaMalloc(&d_v, pool_bytes), "m v");
        K::kv_append_oct(d_k, d_v, d_tab, 0, cells, d_kf, d_vf, s, nullptr);

        const int sinks = 4, tail = 64;
        K::KvExactWindow win{};
        const size_t wrows = (size_t) (sinks + tail) * kvh * hd;
        uint16_t *d_wk = nullptr, *d_wv = nullptr;
        int32_t* d_wtag = nullptr;
        ck(cudaMalloc(&d_wk, wrows * 2), "m wk");
        ck(cudaMalloc(&d_wv, wrows * 2), "m wv");
        ck(cudaMalloc(&d_wtag, (sinks + tail) * 4), "m wtag");
        win.k = d_wk; win.v = d_wv; win.tag = d_wtag;
        K::kv_exact_window_init(win, s, 1, sinks, tail, K::kStepCount, nullptr);
        K::kv_exact_window_append_range(win, 0, 0, cells, d_kf, d_vf, cells, nullptr);
        ck(cudaDeviceSynchronize(), "window");

        const int n_q = 3, cap = cells;
        const int width[n_q] = {cells, 257, 64};
        std::vector<int32_t> ids((size_t) n_q * cap, 0);
        for (int i = 0; i < cap; ++i) ids[i] = i;                                  // ascending, every cell
        for (int i = 0; i < cap; ++i) ids[cap + i] = (int) ((i * 97 + 13) % cells);   // scattered (97 is odd: a permutation)
        for (int i = 0; i < 64; ++i) ids[2 * cap + i] = cells - 64 + i;            // the newest 64: all in the ring
        std::vector<int32_t> steps((size_t) n_q * K::kStepCount, 0);
        for (int z = 0; z < n_q; ++z) {
            steps[(size_t) z * K::kStepCount + K::kStepPos] = cells - 1;
            steps[(size_t) z * K::kStepCount + K::kStepWidth] = width[z];
        }
        std::vector<float> hq((size_t) n_q * nh * hd);
        for (auto& v : hq) v = gauss(rng);
        int32_t *d_ids = nullptr, *d_steps = nullptr;
        float *d_q = nullptr, *d_scr = nullptr, *d_out = nullptr;
        const size_t scr_floats = (size_t) K::qsa_decode_attn_scratch_floats(cap, s) * n_q;
        const size_t out_floats = (size_t) n_q * nh * hd;
        ck(cudaMalloc(&d_ids, ids.size() * 4), "m ids");
        ck(cudaMalloc(&d_steps, steps.size() * 4), "m steps");
        ck(cudaMalloc(&d_q, hq.size() * 4), "m q");
        ck(cudaMalloc(&d_scr, scr_floats * 4), "m scr");
        ck(cudaMalloc(&d_out, out_floats * 4), "m out");
        ck(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "c ids");
        ck(cudaMemcpy(d_steps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "c steps");
        ck(cudaMemcpy(d_q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "c q");

        const auto run = [&](bool with_win, bool masked, int chunk_kernel) {
            K::QsaAttnPools p{};
            p.k_q4 = d_k; p.v_q4 = d_v; p.oct3 = true;
            p.page_table = masked ? d_tab_m : d_tab;
            if (with_win) p.win = win;
            ck(cudaMemset(d_scr, 0, scr_floats * 4), "z scr");
            K::qsa_decode_attn_chunks_only(d_q, p, d_ids, d_steps, cap, s, d_scr, n_q, chunk_kernel, nullptr);
            K::qsa_decode_attn_merge_only(d_steps, cap, s, d_scr, d_out, n_q, 0, nullptr);
            ck(cudaDeviceSynchronize(), "attn");
            std::vector<float> out(out_floats);
            ck(cudaMemcpy(out.data(), d_out, out_floats * 4, cudaMemcpyDeviceToHost), "c out");
            return out;
        };
        std::printf("\ndecode attention on oct3 pools, attn_chunk_kernel vs attn_chunk_kernel_pre75 (bitwise):\n");
        std::vector<float> plain_win, plain_nowin;
        for (int w = 1; w >= 0; --w)
            for (int m = 0; m < 2; ++m) {
                const std::vector<float> a = run(w != 0, m != 0, 0), b = run(w != 0, m != 0, 1);
                int diff = 0, nonfinite = 0;
                for (size_t i = 0; i < out_floats; ++i) {
                    if (std::memcmp(&a[i], &b[i], 4) != 0) ++diff;
                    if (!std::isfinite(a[i])) ++nonfinite;
                }
                if (m == 0) (w ? plain_win : plain_nowin) = a;
                std::printf("  window %-3s masked page %-3s: %s (%d of %zu differ, %d non-finite)\n", w ? "on" : "off",
                            m ? "yes" : "no", diff == 0 && nonfinite == 0 ? "identical" : "MISMATCH", diff, out_floats,
                            nonfinite);
                attn_bad += diff + nonfinite;
            }
        // the window must change the result (else the cases above never read it)
        int changed = 0;
        for (size_t i = 0; i < out_floats; ++i) changed += std::memcmp(&plain_win[i], &plain_nowin[i], 4) != 0;
        std::printf("  the window changes %d of %zu outputs (must be > 0)\n", changed, out_floats);
        if (changed == 0) ++attn_bad;
    }

    const int bad = (enc_mismatch != 0 || dec_mismatch != 0 || oct_bad != 0 || attn_bad != 0);
    std::printf("\n%s\n", bad ? "FAIL: device and host reference disagree" : "ok: device bytes are the reference's");
    return bad ? 1 : 0;
}
