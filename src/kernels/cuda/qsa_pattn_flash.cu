// src/kernels/cuda/qsa_pattn_flash.cu - see include/strata/kernels/qsa_pattn_flash.hpp.
#include "strata/kernels/qsa_pattn_flash.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/core/geometry.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

int pattn_flash_mode() {
    static const int m = [] {
        const char* v = std::getenv("STRATA_PATTN_FLASH");
        const int x = v != nullptr && *v != '\0' ? std::atoi(v) : 0;
        if (x == 1 || x == 2)
            std::fprintf(stderr, "strata pattn-flash: STRATA_PATTN_FLASH=%d: the prompt path's QSA attention in one pass, "
                                 "query in %s (k8v4 pools, where the tensor-core prompt kernel does not run; not bitwise "
                                 "the decode kernels)\n", x, x == 2 ? "FP16" : "FP32");
        else if (x != 0)
            std::fprintf(stderr, "strata pattn-flash: STRATA_PATTN_FLASH=%d is not 1 or 2; off\n", x);
        return x == 1 || x == 2 ? x : 0;
    }();
    return m;
}

namespace {

constexpr int HD = 256;
constexpr int G = 12;
static_assert(HD == strata::core::kCompiledGeometry.head_dim &&
              G == strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv,
              "qsa_pattn_flash is compiled for core/geometry.hpp");
constexpr int TC = 64;                                       // cells per tile
constexpr int NT = 256;                                      // threads per block
constexpr int NW = NT / 32;                                  // warps
constexpr int NSC = HD / KV_Q8_GROUP;                        // K scales per row (4)
constexpr int KW = HD / 4 + 1;                               // staged K row in 32-bit words, padded (no bank conflicts)
constexpr int VB = (HD / QK4_0) * (int) sizeof(block_q4_0);  // V bytes per row (144)
static_assert(VB % 16 == 0, "V rows are staged with 16-byte loads");
static_assert(sizeof(block_q4_0) == 18, "q4_0 block layout");

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 32);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o, 32));
    return v;
}
__device__ __forceinline__ float i8f(uint32_t w, int b) { return (float) (int8_t) ((w >> (8 * b)) & 0xFFu); }

// FP16 pairs (two halves in a 32-bit word): c + a.lo * b.lo + a.hi * b.hi with an FP32 accumulator - one
// v_dot2_f32_f16 on gfx103x+; the same arithmetic in FP32 elsewhere.
__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
#if defined(__HIPCC__)
    typedef _Float16 h2v __attribute__((ext_vector_type(2)));
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(h2v, a), __builtin_bit_cast(h2v, b), c, false);
#else
    const float2 x = __half22float2(*reinterpret_cast<const __half2*>(&a));
    const float2 y = __half22float2(*reinterpret_cast<const __half2*>(&b));
    return fmaf(x.y, y.y, fmaf(x.x, y.x, c));
#endif
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}
// four int8 codes -> two exact FP16 pairs (bytes 0,1 and 2,3): 0x64XX is 1024 + XX, the code biased by 128
__device__ __forceinline__ void i8x4_h2(uint32_t w, uint32_t& lo, uint32_t& hi) {
    const uint32_t x = w ^ 0x80808080u;
    lo = __byte_perm(x, 0x64646464u, 0x4140);
    hi = __byte_perm(x, 0x64646464u, 0x4342);
    const __half2 bias = __float2half2_rn(1152.0f);
    const __half2 l = __hsub2(*reinterpret_cast<const __half2*>(&lo), bias);
    const __half2 h = __hsub2(*reinterpret_cast<const __half2*>(&hi), bias);
    lo = *reinterpret_cast<const uint32_t*>(&l);
    hi = *reinterpret_cast<const uint32_t*>(&h);
}

// one block per (KV head, query): grid (n_kv_heads, n_q).  Q16: the query in FP16 pairs (v_dot2_f32_f16, the
// tensor-core prompt kernels' precision), else FP32 FMAs.
template <bool Q16>
__global__ void __launch_bounds__(NT) flash_k8v4_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                        const int32_t* __restrict__ ids,
                                                        const int32_t* __restrict__ steps, int n_kv_heads,
                                                        int page_size, int cap, float scale, float* __restrict__ attn) {
    const int kvh = blockIdx.x, z = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    q += ((size_t) z * n_kv_heads + kvh) * G * HD;
    ids += (size_t) z * cap;
    const int width = steps[(size_t) z * kStepCount + kStepWidth];

    __shared__ __align__(16) uint32_t sk[TC][KW];      // 16.3 KB, the tile's K codes
    __shared__ __align__(16) float sks[TC][NSC];        // its K scales
    __shared__ __align__(16) uint8_t sv[TC][VB];       // 9 KB, the tile's V blocks
    __shared__ __align__(16) float sp[TC][G];          // scores, then probabilities (cell-major)
    __shared__ long long srow[2][TC];                  // pool rows: this tile's and the next one's
    __shared__ float sm[G], sl[G], salpha[G];

    if (t < G) { sm[t] = -FLT_MAX; sl[t] = 0.0f; }

    // value phase: thread = 4 dims (q4_0 block vbk, bytes 2 jp and 2 jp + 1: dims 32 vbk + 2 jp + {0, 1} and + 16) x 12
    // heads, cell quarter cq = t / 64 (cells cq, cq + 4, ...).  A cell's 12 weights are three broadcast reads.
    const int dq = t & 63, vbk = dq >> 3, jp = dq & 7, cq = t >> 6;
    float acc[G][4];
#pragma unroll
    for (int h = 0; h < G; ++h)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[h][e] = 0.0f;
    // score phase: thread (head group hg = warp & 3, cells cb .. cb + 7, dim slice ds): heads 3 hg .. 3 hg + 2,
    // dims 4 ds + 32 m + e (m < 8, e < 4) - its q slice stays in registers for the whole selection, a K word is one
    // conflict-free 32-bit read (row stride 65 words), and the 8 slices of a cell are summed by a reduce-scatter
    const int ds = lane & 7, hg = warp & 3, h0 = 3 * hg;
    const int cb = (warp >> 2) * 32 + (lane >> 3) * 8;
    uint32_t qh[3][Q16 ? 16 : 1];   // Q16: the q slice as FP16 pairs
    float qr[3][Q16 ? 1 : 32];      // else in FP32
#pragma unroll
    for (int k = 0; k < 3; ++k)
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            const float4 v = *reinterpret_cast<const float4*>(q + (size_t) (h0 + k) * HD + 32 * m + 4 * ds);
            if constexpr (Q16) {
                qh[k][2 * m] = pack_h2(v.x, v.y);
                qh[k][2 * m + 1] = pack_h2(v.z, v.w);
            } else {
                qr[k][4 * m] = v.x; qr[k][4 * m + 1] = v.y; qr[k][4 * m + 2] = v.z; qr[k][4 * m + 3] = v.w;
            }
        }

    const int n_tiles = width > 0 ? (width + TC - 1) / TC : 0;
    // pool row of cell t of a tile; -1 masked (page -1) or past the selection.  Masked / absent cells are staged from
    // row 0 and get weight 0.
    const auto row_of = [&](int tl, int c) -> long long {
        const int cc = tl * TC + c;
        if (cc >= width) return -1;
        const int cell = ids[cc];
        const long long page = (long long) p.page_table[cell / page_size];
        return page >= 0 ? (page * n_kv_heads + kvh) * page_size + (cell % page_size) : -1;
    };
    constexpr int KJ = TC * (HD / 16) / NT;                    // 16-byte K loads a thread (4)
    constexpr int VJ = (TC * (VB / 16) + NT - 1) / NT;         // 16-byte V loads a thread (3, the last partial)
    static_assert(KJ == 4 && VJ == 3, "the staging below is written out for 4 K and 3 V loads a thread");
    uint4 k0r, k1r, k2r, k3r, v0r, v1r, v2r;
    float kscr = 0.0f;
    const int kc = t >> 4, kp = t & 15;   // K load j: cell kc + 16 j, part kp
#define FLASH_LOAD_K(rows)                                                                                    \
    do {                                                                                                      \
        const long long r0 = (rows)[kc] >= 0 ? (rows)[kc] : 0, r1 = (rows)[kc + 16] >= 0 ? (rows)[kc + 16] : 0; \
        const long long r2 = (rows)[kc + 32] >= 0 ? (rows)[kc + 32] : 0;                                      \
        const long long r3 = (rows)[kc + 48] >= 0 ? (rows)[kc + 48] : 0;                                      \
        k0r = reinterpret_cast<const uint4*>(p.k_q + r0 * HD)[kp];                                            \
        k1r = reinterpret_cast<const uint4*>(p.k_q + r1 * HD)[kp];                                            \
        k2r = reinterpret_cast<const uint4*>(p.k_q + r2 * HD)[kp];                                            \
        k3r = reinterpret_cast<const uint4*>(p.k_q + r3 * HD)[kp];                                            \
        const long long rs = (rows)[t >> 2] >= 0 ? (rows)[t >> 2] : 0;                                        \
        kscr = __half2float(__ushort_as_half(p.k_scale[rs * NSC + (t & 3)]));                                 \
    } while (0)
    const auto put4 = [&](int c, uint4 v) {
        uint32_t* d = &sk[c][kp * 4];
        d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
    };
#define FLASH_STORE_K()                                                                                       \
    do {                                                                                                      \
        put4(kc, k0r); put4(kc + 16, k1r); put4(kc + 32, k2r); put4(kc + 48, k3r);                            \
        sks[t >> 2][t & 3] = kscr;                                                                            \
    } while (0)
    // V load j: 16-byte part idx = t + 256 j of the tile's 64 x 9
    constexpr int NVP = TC * (VB / 16);
    const int vi0 = t, vi1 = t + NT, vi2 = t + 2 * NT;
#define FLASH_LOAD_V(rows)                                                                                    \
    do {                                                                                                      \
        const long long r0 = (rows)[vi0 / 9] >= 0 ? (rows)[vi0 / 9] : 0;                                     \
        const long long r1 = (rows)[vi1 / 9] >= 0 ? (rows)[vi1 / 9] : 0;                                     \
        v0r = reinterpret_cast<const uint4*>(p.v_q4 + r0 * VB)[vi0 % 9];                                      \
        v1r = reinterpret_cast<const uint4*>(p.v_q4 + r1 * VB)[vi1 % 9];                                      \
        if (vi2 < NVP) {                                                                                      \
            const long long r2 = (rows)[vi2 / 9] >= 0 ? (rows)[vi2 / 9] : 0;                                  \
            v2r = reinterpret_cast<const uint4*>(p.v_q4 + r2 * VB)[vi2 % 9];                                  \
        }                                                                                                     \
    } while (0)
#define FLASH_STORE_V()                                                                                       \
    do {                                                                                                      \
        reinterpret_cast<uint4*>(sv[vi0 / 9])[vi0 % 9] = v0r;                                                 \
        reinterpret_cast<uint4*>(sv[vi1 / 9])[vi1 % 9] = v1r;                                                 \
        if (vi2 < NVP) reinterpret_cast<uint4*>(sv[vi2 / 9])[vi2 % 9] = v2r;                                  \
    } while (0)
    static_assert(VB / 16 == 9, "9 parts a V row");
    if (n_tiles > 0) {
        if (t < TC) srow[0][t] = row_of(0, t);
        __syncthreads();
        FLASH_LOAD_K(srow[0]);
        FLASH_STORE_K();
    }
    for (int tile = 0; tile < n_tiles; ++tile) {
        const int b = tile & 1;
        const int n_here = min(TC, width - tile * TC);
        const bool more = tile + 1 < n_tiles;
        if (t < TC && more) srow[b ^ 1][t] = row_of(tile + 1, t);
        __syncthreads();   // this tile's K and rows are staged; the previous tile's value phase is done
        FLASH_LOAD_V(srow[b]);   // in flight during the scores
        // scores: 8 cells x 3 heads x 32 dims a thread, then the 8 slices summed (reduce-scatter: lane ds ends with
        // cell cb + ds, all three heads)
        {
            float v[24];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t* krow = sk[cb + i];
                const float4 ks = *reinterpret_cast<const float4*>(sks[cb + i]);
                const float ksc[4] = {ks.x, ks.y, ks.z, ks.w};
                float s3[3] = {0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int gq = 0; gq < NSC; ++gq) {
                    float a[3] = {0.0f, 0.0f, 0.0f};
#pragma unroll
                    for (int mm = 0; mm < 2; ++mm) {
                        const int m = 2 * gq + mm;
                        if constexpr (Q16) {
                            uint32_t klo, khi;
                            i8x4_h2(krow[8 * m + ds], klo, khi);
#pragma unroll
                            for (int k = 0; k < 3; ++k) a[k] = dot2(khi, qh[k][2 * m + 1], dot2(klo, qh[k][2 * m], a[k]));
                        } else {
                            const uint32_t k4 = krow[8 * m + ds];
                            const float k0 = i8f(k4, 0), k1 = i8f(k4, 1), k2 = i8f(k4, 2), k3 = i8f(k4, 3);
#pragma unroll
                            for (int k = 0; k < 3; ++k) {
                                a[k] = fmaf(k0, qr[k][4 * m], a[k]); a[k] = fmaf(k1, qr[k][4 * m + 1], a[k]);
                                a[k] = fmaf(k2, qr[k][4 * m + 2], a[k]); a[k] = fmaf(k3, qr[k][4 * m + 3], a[k]);
                            }
                        }
                    }
#pragma unroll
                    for (int k = 0; k < 3; ++k) s3[k] = fmaf(a[k], ksc[gq], s3[k]);
                }
#pragma unroll
                for (int k = 0; k < 3; ++k) v[3 * i + k] = s3[k];
            }
            const bool b2 = (ds & 4) != 0, b1 = (ds & 2) != 0, b0 = (ds & 1) != 0;
            float r12[12];
#pragma unroll
            for (int k = 0; k < 12; ++k)
                r12[k] = (b2 ? v[12 + k] : v[k]) + __shfl_xor_sync(0xffffffffu, b2 ? v[k] : v[12 + k], 4, 32);
            float r6[6];
#pragma unroll
            for (int k = 0; k < 6; ++k)
                r6[k] = (b1 ? r12[6 + k] : r12[k]) + __shfl_xor_sync(0xffffffffu, b1 ? r12[k] : r12[6 + k], 2, 32);
            float r3[3];
#pragma unroll
            for (int k = 0; k < 3; ++k)
                r3[k] = (b0 ? r6[3 + k] : r6[k]) + __shfl_xor_sync(0xffffffffu, b0 ? r6[k] : r6[3 + k], 1, 32);
            const int c = cb + ds;
            const bool ok = c < n_here && srow[b][c] >= 0;
#pragma unroll
            for (int k = 0; k < 3; ++k) sp[c][h0 + k] = ok ? r3[k] * scale : -FLT_MAX;
        }
        FLASH_STORE_V();
        __syncthreads();   // scores and V are staged; the K tile is free
        if (more) FLASH_LOAD_K(srow[b ^ 1]);   // in flight during the softmax and the values
        // online softmax: warp w takes heads w and w + 8
        for (int h = warp; h < G; h += NW) {
            const float a = sp[lane][h], b = sp[lane + 32][h];
            const float m_old = sm[h];
            const float m_new = fmaxf(m_old, wmax(fmaxf(a, b)));
            const float ea = a > -FLT_MAX ? __expf(a - m_new) : 0.0f;
            const float eb = b > -FLT_MAX ? __expf(b - m_new) : 0.0f;
            const float sum = wsum(ea + eb);
            sp[lane][h] = ea;
            sp[lane + 32][h] = eb;
            if (lane == 0) {
                const float alpha = m_old > -FLT_MAX ? __expf(m_old - m_new) : 0.0f;
                sm[h] = m_new;
                sl[h] = fmaf(sl[h], alpha, sum);
                salpha[h] = alpha;
            }
        }
        __syncthreads();
        // values: rescale, then fold this quarter's cells
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float al = salpha[h];
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[h][e] *= al;
        }
        for (int c = cq; c < n_here; c += 4) {
            const uint8_t* blk = sv[c] + vbk * 18;
            const float dd = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(blk)));
            const uint32_t w2 = *reinterpret_cast<const uint16_t*>(blk + 2 + 2 * jp);
            const float v0 = (float) ((int) (w2 & 0x0Fu) - 8) * dd;          // dim 2 jp
            const float v1 = (float) ((int) ((w2 >> 8) & 0x0Fu) - 8) * dd;   // dim 2 jp + 1
            const float v2 = (float) ((int) ((w2 >> 4) & 0x0Fu) - 8) * dd;   // dim 2 jp + 16
            const float v3 = (float) ((int) (w2 >> 12) - 8) * dd;            // dim 2 jp + 17
            const float4* w4 = reinterpret_cast<const float4*>(sp[c]);
            const float4 x0 = w4[0], x1 = w4[1], x2 = w4[2];
            const float ww[G] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w, x2.x, x2.y, x2.z, x2.w};
#pragma unroll
            for (int h = 0; h < G; ++h) {
                acc[h][0] = fmaf(ww[h], v0, acc[h][0]); acc[h][1] = fmaf(ww[h], v1, acc[h][1]);
                acc[h][2] = fmaf(ww[h], v2, acc[h][2]); acc[h][3] = fmaf(ww[h], v3, acc[h][3]);
            }
        }
        if (more) FLASH_STORE_K();   // nobody reads the K tile after the scores
    }
#undef FLASH_LOAD_K
#undef FLASH_STORE_K
#undef FLASH_LOAD_V
#undef FLASH_STORE_V
    // the four cell quarters, one slot at a time through the (now free) K tile: 3 into 2, 1 into 0, 2 into 0
    float* red = reinterpret_cast<float*>(&sk[0][0]);
    static_assert(64 * G * 4 <= TC * KW, "the reduction fits in the K tile");
    const auto put = [&]() {
#pragma unroll
        for (int h = 0; h < G; ++h)
#pragma unroll
            for (int e = 0; e < 4; ++e) red[(h * 4 + e) * 64 + dq] = acc[h][e];
    };
    const auto add = [&]() {
#pragma unroll
        for (int h = 0; h < G; ++h)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[h][e] += red[(h * 4 + e) * 64 + dq];
    };
    __syncthreads();
    if (cq == 3) put();
    __syncthreads();
    if (cq == 2) add();
    __syncthreads();
    if (cq == 1) put();
    __syncthreads();
    if (cq == 0) add();
    __syncthreads();
    if (cq == 2) put();
    __syncthreads();
    if (cq == 0) {
        add();
        float* out = attn + ((size_t) z * n_kv_heads + kvh) * G * HD;
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float L = sl[h];
            const float inv = L > 0.0f ? 1.0f / L : 0.0f;
            float* row = out + (size_t) h * HD + 32 * vbk + 2 * jp;
            *reinterpret_cast<float2*>(row) = make_float2(acc[h][0] * inv, acc[h][1] * inv);
            *reinterpret_cast<float2*>(row + 16) = make_float2(acc[h][2] * inv, acc[h][3] * inv);
        }
    }
}

}  // namespace

bool qsa_pattn_flash_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, bool q16, void* stream) {
    if (n_q <= 0) return true;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids || !steps || !attn ||
        !pools.page_table || !pools.k_q || !pools.k_scale || !pools.v_q4 || pools.k_q4 || pools.v_q || n_q > 65535)
        return false;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) n_q);
    if (q16)
        flash_k8v4_kernel<true><<<grid, NT, 0, (cudaStream_t) stream>>>(q, pools, ids, steps, (int) s.n_head_kv,
                                                                        (int) s.page_size, (int) cap, scale, attn);
    else
        flash_k8v4_kernel<false><<<grid, NT, 0, (cudaStream_t) stream>>>(q, pools, ids, steps, (int) s.n_head_kv,
                                                                         (int) s.page_size, (int) cap, scale, attn);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr, "strata pattn-flash: launch failed (%s); the usual attention runs instead\n",
                         cudaGetErrorString(e));
        }
        return false;
    }
    return true;
}

}  // namespace strata::kernels
