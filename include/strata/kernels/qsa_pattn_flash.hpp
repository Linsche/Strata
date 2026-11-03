// include/strata/kernels/qsa_pattn_flash.hpp - STRATA_PATTN_FLASH: the prompt path's QSA attention in one pass, for
// cards without the tensor-core / WMMA prompt kernels (RDNA2 gfx103x first).
//
// Where qsa_prompt_attn_batch refuses (an AMD card without WMMA, a CUDA card below sm_70), the prompt reads every QSA
// layer's attention through the decode kernels, 32 queries a call: one block per (64-cell chunk, KV head, query) that
// reloads the query (12 KB), and writes a 12 KB partial per chunk that a second kernel merges.  At a 2,051-cell
// selection that is ~1.6 MB of scratch traffic per query and layer.  Measured on an RX 6800 (k8v4): 14.3 us per query
// and layer, 3.5 TFLOPS.
//
// Here one block per (query, KV head) walks the whole selection in 64-cell tiles with an online softmax: the query is
// staged once, K (int8 + scales) and V (rotated Q4_0) are staged per tile, and the output is written once.  No scratch,
// no merge kernel.  The sums run in another order than the decode kernels', so the output is not bitwise theirs.
// STRATA_PATTN_FLASH=1 keeps the query in FP32 (FP32 FMAs throughout); =2 holds it in FP16 pairs for v_dot2_f32_f16
// (FP32 accumulation, K exact in FP16 - the input precision of the tensor-core prompt kernels), about 13% faster.
// tests/hip/qsa_pattn_flash.cpp compares both with the decode kernels.
//
// Off unless STRATA_PATTN_FLASH is 1 or 2.  k8v4 pools only (int8 K + scales, Q4_0 V); anything else answers false and
// the caller runs the usual loop.
#pragma once

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace strata::kernels {

/// STRATA_PATTN_FLASH: 0 off (default); 1 on with an FP32 query; 2 on with an FP16 query.  Read once.
int pattn_flash_mode();

/// The attention of `n_q` prompt queries (each with its own selection `ids` + cap, step row `steps`) into `attn`
/// ([n_q][n_head][head_dim], V still rotated as qsa_decode_attn_batch leaves it).  False: not applicable (pool
/// format, geometry) or the launch failed - nothing was written that the caller can rely on.
bool qsa_pattn_flash_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, bool q16, void* stream);

}  // namespace strata::kernels
