//! Experimental CUDA/cuTile interop for Candle quantized kernels.
//!
//! This module lives inside `candle-core` rather than `pi-ai-candle` because the
//! useful backend seam is in Candle's private quantized CUDA path. Candle stays
//! the owner of the CUDA device, context, stream, allocations, synchronization,
//! and fallback policy; cuTile only borrows Candle's raw handles and receives
//! Candle-owned device pointers.

use super::cuda::{QCudaStorage, MATRIX_ROW_PADDING};
use super::GgmlDType;
use crate::{
    backend::{BackendDevice, BackendStorage},
    CudaDevice, CudaStorage, DType, Result, Shape,
};

const QK_K: usize = 256;
const Q8_1_BLOCK_SIZE: usize = 32;
const Q8_1_BLOCK_BYTES: usize = 36;
const Q8_1_MMQ_BLOCK_SIZE: usize = 4 * Q8_1_BLOCK_SIZE;
const Q8_1_MMQ_BLOCK_BYTES: usize = 4 * Q8_1_BLOCK_BYTES;

#[inline]
fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

#[cutile::module]
mod q4k_q8_1_matvec_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn q4k_q8_1_matvec_b1_f32(
        q4_ptr: *mut u8,
        q8_ptr: *mut u8,
        out_ptr: *mut f32,
        ncols: i32,
        nrows: i32,
    ) {
        let pid = get_tile_block_id();
        let row: i32 = pid.0;
        if row >= nrows {
            return;
        }

        let q4_base: PointerTile<*mut u8, { [] }> = pointer_to_tile(q4_ptr);
        let q4_half_base: PointerTile<*mut f16, { [] }> = ptr_to_ptr(q4_base);
        let q4_i8_base: PointerTile<*mut i8, { [] }> = ptr_to_ptr(q4_base);
        let q4_half_1: PointerTile<*mut f16, { [1] }> = q4_half_base.reshape(const_shape![1]);
        let q4_i8_1: PointerTile<*mut i8, { [1] }> = q4_i8_base.reshape(const_shape![1]);
        let q8_base: PointerTile<*mut u8, { [] }> = pointer_to_tile(q8_ptr);
        let q8_half_base: PointerTile<*mut f16, { [] }> = ptr_to_ptr(q8_base);
        let q8_i8_base: PointerTile<*mut i8, { [] }> = ptr_to_ptr(q8_base);
        let q8_half_1: PointerTile<*mut f16, { [1] }> = q8_half_base.reshape(const_shape![1]);
        let q8_i8_1: PointerTile<*mut i8, { [1] }> = q8_i8_base.reshape(const_shape![1]);

        let mut acc: Tile<f32, { [] }> = constant(0.0f32, const_shape![]);
        let blocks_per_row: i32 = ncols / 256i32;
        let q4_row_base: i32 = row * blocks_per_row * 144i32;

        for block in 0i32..blocks_per_row {
            let q4_block_base: i32 = q4_row_base + block * 144i32;

            let d_offset: Tile<i32, { [1] }> =
                broadcast_scalar(q4_block_base / 2i32, const_shape![1]);
            let d_ptr: PointerTile<*mut f16, { [1] }> = q4_half_1.offset_tile(d_offset);
            let (d_h_1, _d_tok): (Tile<f16, { [1] }>, Token) = load_ptr_tko(
                d_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let dmin_offset: Tile<i32, { [1] }> =
                broadcast_scalar(q4_block_base / 2i32 + 1i32, const_shape![1]);
            let dmin_ptr: PointerTile<*mut f16, { [1] }> = q4_half_1.offset_tile(dmin_offset);
            let (dmin_h_1, _dmin_tok): (Tile<f16, { [1] }>, Token) = load_ptr_tko(
                dmin_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let d_1: Tile<f32, { [1] }> = ftof(d_h_1, rounding::NearestEven);
            let dmin_1: Tile<f32, { [1] }> = ftof(dmin_h_1, rounding::NearestEven);
            let d: Tile<f32, { [] }> = d_1.reshape(const_shape![]);
            let dmin: Tile<f32, { [] }> = dmin_1.reshape(const_shape![]);
            let d32: Tile<f32, { [32] }> = d.reshape(const_shape![1]).broadcast(const_shape![32]);
            let dmin32: Tile<f32, { [32] }> =
                dmin.reshape(const_shape![1]).broadcast(const_shape![32]);

            for group in 0i32..8i32 {
                let q4_byte_base: i32 = q4_block_base + 16i32 + (group / 2i32) * 32i32;
                let offsets: Tile<i32, { [32] }> = iota(const_shape![32]);
                let c15_32: Tile<i32, { [32] }> = constant(15i32, const_shape![32]);
                let c255_32: Tile<i32, { [32] }> = constant(255i32, const_shape![32]);
                let q4_total_offsets: Tile<i32, { [32] }> =
                    offsets + broadcast_scalar(q4_byte_base, const_shape![32]);
                let q4_i8_32: PointerTile<*mut i8, { [32] }> = q4_i8_1.broadcast(const_shape![32]);
                let q4_ptrs: PointerTile<*mut i8, { [32] }> =
                    q4_i8_32.offset_tile(q4_total_offsets);
                let (q4_bytes, _q4_tok): (Tile<i8, { [32] }>, Token) = load_ptr_tko(
                    q4_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let q4_byte_i32: Tile<i32, { [32] }> = exti(q4_bytes) & c255_32;
                let shift_32: Tile<i32, { [32] }> =
                    broadcast_scalar((group % 2i32) * 4i32, const_shape![32]);
                let q4_i32: Tile<i32, { [32] }> = shri(q4_byte_i32, shift_32) & c15_32;
                let q4_f32: Tile<f32, { [32] }> = convert_tile(q4_i32);

                #[allow(unused_assignments)]
                let mut scale_i32: Tile<i32, { [] }> = constant(0i32, const_shape![]);
                #[allow(unused_assignments)]
                let mut min_i32: Tile<i32, { [] }> = constant(0i32, const_shape![]);
                if group < 4i32 {
                    let scale_offset: Tile<i32, { [1] }> =
                        broadcast_scalar(q4_block_base + 4i32 + group, const_shape![1]);
                    let scale_ptr: PointerTile<*mut i8, { [1] }> =
                        q4_i8_1.offset_tile(scale_offset);
                    let (scale_b_1, _scale_tok): (Tile<i8, { [1] }>, Token) = load_ptr_tko(
                        scale_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let min_offset: Tile<i32, { [1] }> =
                        broadcast_scalar(q4_block_base + 8i32 + group, const_shape![1]);
                    let min_ptr: PointerTile<*mut i8, { [1] }> = q4_i8_1.offset_tile(min_offset);
                    let (min_b_1, _min_tok): (Tile<i8, { [1] }>, Token) = load_ptr_tko(
                        min_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let scale_ext_1: Tile<i32, { [1] }> = exti(scale_b_1);
                    let min_ext_1: Tile<i32, { [1] }> = exti(min_b_1);
                    let c63_1: Tile<i32, { [1] }> = constant(63i32, const_shape![1]);
                    scale_i32 = (scale_ext_1 & c63_1).reshape(const_shape![]);
                    min_i32 = (min_ext_1 & c63_1).reshape(const_shape![]);
                } else {
                    let packed_offset: Tile<i32, { [1] }> =
                        broadcast_scalar(q4_block_base + 8i32 + group, const_shape![1]);
                    let packed_ptr: PointerTile<*mut i8, { [1] }> =
                        q4_i8_1.offset_tile(packed_offset);
                    let (packed_b_1, _packed_tok): (Tile<i8, { [1] }>, Token) = load_ptr_tko(
                        packed_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let scale_hi_offset: Tile<i32, { [1] }> =
                        broadcast_scalar(q4_block_base + group, const_shape![1]);
                    let scale_hi_ptr: PointerTile<*mut i8, { [1] }> =
                        q4_i8_1.offset_tile(scale_hi_offset);
                    let (scale_hi_b_1, _scale_hi_tok): (Tile<i8, { [1] }>, Token) = load_ptr_tko(
                        scale_hi_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let min_hi_offset: Tile<i32, { [1] }> =
                        broadcast_scalar(q4_block_base + 4i32 + group, const_shape![1]);
                    let min_hi_ptr: PointerTile<*mut i8, { [1] }> =
                        q4_i8_1.offset_tile(min_hi_offset);
                    let (min_hi_b_1, _min_hi_tok): (Tile<i8, { [1] }>, Token) = load_ptr_tko(
                        min_hi_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let packed_ext_1: Tile<i32, { [1] }> = exti(packed_b_1);
                    let scale_hi_ext_1: Tile<i32, { [1] }> = exti(scale_hi_b_1);
                    let min_hi_ext_1: Tile<i32, { [1] }> = exti(min_hi_b_1);
                    let c255_1: Tile<i32, { [1] }> = constant(255i32, const_shape![1]);
                    let packed: Tile<i32, { [] }> = (packed_ext_1 & c255_1).reshape(const_shape![]);
                    let scale_hi: Tile<i32, { [] }> =
                        (scale_hi_ext_1 & c255_1).reshape(const_shape![]);
                    let min_hi: Tile<i32, { [] }> = (min_hi_ext_1 & c255_1).reshape(const_shape![]);
                    let c4: Tile<i32, { [] }> = constant(4i32, const_shape![]);
                    let c6: Tile<i32, { [] }> = constant(6i32, const_shape![]);
                    let c15: Tile<i32, { [] }> = constant(15i32, const_shape![]);
                    scale_i32 = (packed & c15) | shli(shri(scale_hi, c6), c4, overflow::NoWrap);
                    min_i32 = shri(packed, c4) | shli(shri(min_hi, c6), c4, overflow::NoWrap);
                }

                let scale_f32: Tile<f32, { [] }> = convert_tile(scale_i32);
                let min_f32: Tile<f32, { [] }> = convert_tile(min_i32);
                let scale32: Tile<f32, { [32] }> = scale_f32
                    .reshape(const_shape![1])
                    .broadcast(const_shape![32]);
                let min32: Tile<f32, { [32] }> =
                    min_f32.reshape(const_shape![1]).broadcast(const_shape![32]);

                let q8_block_index: i32 = block * 8i32 + group;
                let q8_block_base: i32 = q8_block_index * 36i32;
                let d8_offset: Tile<i32, { [1] }> =
                    broadcast_scalar(q8_block_base / 2i32, const_shape![1]);
                let d8_ptr: PointerTile<*mut f16, { [1] }> = q8_half_1.offset_tile(d8_offset);
                let (d8_h_1, _d8_tok): (Tile<f16, { [1] }>, Token) = load_ptr_tko(
                    d8_ptr,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None,
                    Latency::<0>,
                );
                let d8_1: Tile<f32, { [1] }> = ftof(d8_h_1, rounding::NearestEven);
                let d8: Tile<f32, { [] }> = d8_1.reshape(const_shape![]);
                let d8_32: Tile<f32, { [32] }> =
                    d8.reshape(const_shape![1]).broadcast(const_shape![32]);

                let q8_total_offsets: Tile<i32, { [32] }> =
                    offsets + broadcast_scalar(q8_block_base + 4i32, const_shape![32]);
                let q8_i8_32: PointerTile<*mut i8, { [32] }> = q8_i8_1.broadcast(const_shape![32]);
                let q8_ptrs: PointerTile<*mut i8, { [32] }> =
                    q8_i8_32.offset_tile(q8_total_offsets);
                let (q8_bytes, _q8_tok): (Tile<i8, { [32] }>, Token) = load_ptr_tko(
                    q8_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let q8_i32: Tile<i32, { [32] }> = exti(q8_bytes);
                let q8_f32: Tile<f32, { [32] }> = convert_tile(q8_i32);

                let q4_val: Tile<f32, { [32] }> = d32 * scale32 * q4_f32 - dmin32 * min32;
                let q8_val: Tile<f32, { [32] }> = d8_32 * q8_f32;
                let prod: Tile<f32, { [32] }> = q4_val * q8_val;
                let group_sum: Tile<f32, { [] }> = reduce_sum(prod, 0i32);
                acc = acc + group_sum;
            }
        }

        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_1: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_offset: Tile<i32, { [1] }> = broadcast_scalar(row, const_shape![1]);
        let out_dst: PointerTile<*mut f32, { [1] }> = out_1.offset_tile(out_offset);
        store_ptr_tko(
            out_dst,
            acc.reshape(const_shape![1]),
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<0>,
        );
    }
}

#[cutile::module]
mod q4k_q8_1_matmul_batched_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn q4k_q8_1_matmul_batched_f32(
        q4_ptr: *mut u8,
        q8_ptr: *mut u8,
        out_ptr: *mut f32,
        ncols: i32,
        nrows: i32,
        b_size: i32,
        q8_row_stride_bytes: i32,
    ) {
        let pid = get_tile_block_id();
        let row: i32 = pid.0;
        let batch: i32 = pid.1;
        if row >= nrows || batch >= b_size {
            return;
        }

        let q4_base: PointerTile<*mut u8, { [] }> = pointer_to_tile(q4_ptr);
        let q4_half_base: PointerTile<*mut f16, { [] }> = ptr_to_ptr(q4_base);
        let q4_i8_base: PointerTile<*mut i8, { [] }> = ptr_to_ptr(q4_base);
        let q4_half_1: PointerTile<*mut f16, { [1] }> = q4_half_base.reshape(const_shape![1]);
        let q4_i8_1: PointerTile<*mut i8, { [1] }> = q4_i8_base.reshape(const_shape![1]);
        let q8_base: PointerTile<*mut u8, { [] }> = pointer_to_tile(q8_ptr);
        let q8_half_base: PointerTile<*mut f16, { [] }> = ptr_to_ptr(q8_base);
        let q8_i8_base: PointerTile<*mut i8, { [] }> = ptr_to_ptr(q8_base);
        let q8_half_1: PointerTile<*mut f16, { [1] }> = q8_half_base.reshape(const_shape![1]);
        let q8_i8_1: PointerTile<*mut i8, { [1] }> = q8_i8_base.reshape(const_shape![1]);

        let mut acc: Tile<f32, { [] }> = constant(0.0f32, const_shape![]);
        let blocks_per_row: i32 = ncols / 256i32;
        let q4_row_base: i32 = row * blocks_per_row * 144i32;

        for block in 0i32..blocks_per_row {
            let q4_block_base: i32 = q4_row_base + block * 144i32;

            let d_offset: Tile<i32, { [1] }> =
                broadcast_scalar(q4_block_base / 2i32, const_shape![1]);
            let d_ptr: PointerTile<*mut f16, { [1] }> = q4_half_1.offset_tile(d_offset);
            let (d_h_1, _d_tok): (Tile<f16, { [1] }>, Token) = load_ptr_tko(
                d_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let dmin_offset: Tile<i32, { [1] }> =
                broadcast_scalar(q4_block_base / 2i32 + 1i32, const_shape![1]);
            let dmin_ptr: PointerTile<*mut f16, { [1] }> = q4_half_1.offset_tile(dmin_offset);
            let (dmin_h_1, _dmin_tok): (Tile<f16, { [1] }>, Token) = load_ptr_tko(
                dmin_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let d_1: Tile<f32, { [1] }> = ftof(d_h_1, rounding::NearestEven);
            let dmin_1: Tile<f32, { [1] }> = ftof(dmin_h_1, rounding::NearestEven);
            let d: Tile<f32, { [] }> = d_1.reshape(const_shape![]);
            let dmin: Tile<f32, { [] }> = dmin_1.reshape(const_shape![]);
            let d32: Tile<f32, { [32] }> = d.reshape(const_shape![1]).broadcast(const_shape![32]);
            let dmin32: Tile<f32, { [32] }> =
                dmin.reshape(const_shape![1]).broadcast(const_shape![32]);

            for group in 0i32..8i32 {
                let q4_byte_base: i32 = q4_block_base + 16i32 + (group / 2i32) * 32i32;
                let offsets: Tile<i32, { [32] }> = iota(const_shape![32]);
                let c15_32: Tile<i32, { [32] }> = constant(15i32, const_shape![32]);
                let c255_32: Tile<i32, { [32] }> = constant(255i32, const_shape![32]);
                let q4_total_offsets: Tile<i32, { [32] }> =
                    offsets + broadcast_scalar(q4_byte_base, const_shape![32]);
                let q4_i8_32: PointerTile<*mut i8, { [32] }> = q4_i8_1.broadcast(const_shape![32]);
                let q4_ptrs: PointerTile<*mut i8, { [32] }> =
                    q4_i8_32.offset_tile(q4_total_offsets);
                let (q4_bytes, _q4_tok): (Tile<i8, { [32] }>, Token) = load_ptr_tko(
                    q4_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let q4_byte_i32: Tile<i32, { [32] }> = exti(q4_bytes) & c255_32;
                let shift_32: Tile<i32, { [32] }> =
                    broadcast_scalar((group % 2i32) * 4i32, const_shape![32]);
                let q4_i32: Tile<i32, { [32] }> = shri(q4_byte_i32, shift_32) & c15_32;
                let q4_f32: Tile<f32, { [32] }> = convert_tile(q4_i32);

                #[allow(unused_assignments)]
                let mut scale_i32: Tile<i32, { [] }> = constant(0i32, const_shape![]);
                #[allow(unused_assignments)]
                let mut min_i32: Tile<i32, { [] }> = constant(0i32, const_shape![]);
                if group < 4i32 {
                    let scale_offset: Tile<i32, { [1] }> =
                        broadcast_scalar(q4_block_base + 4i32 + group, const_shape![1]);
                    let scale_ptr: PointerTile<*mut i8, { [1] }> =
                        q4_i8_1.offset_tile(scale_offset);
                    let (scale_b_1, _scale_tok): (Tile<i8, { [1] }>, Token) = load_ptr_tko(
                        scale_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let min_offset: Tile<i32, { [1] }> =
                        broadcast_scalar(q4_block_base + 8i32 + group, const_shape![1]);
                    let min_ptr: PointerTile<*mut i8, { [1] }> = q4_i8_1.offset_tile(min_offset);
                    let (min_b_1, _min_tok): (Tile<i8, { [1] }>, Token) = load_ptr_tko(
                        min_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let scale_ext_1: Tile<i32, { [1] }> = exti(scale_b_1);
                    let min_ext_1: Tile<i32, { [1] }> = exti(min_b_1);
                    let c63_1: Tile<i32, { [1] }> = constant(63i32, const_shape![1]);
                    scale_i32 = (scale_ext_1 & c63_1).reshape(const_shape![]);
                    min_i32 = (min_ext_1 & c63_1).reshape(const_shape![]);
                } else {
                    let packed_offset: Tile<i32, { [1] }> =
                        broadcast_scalar(q4_block_base + 8i32 + group, const_shape![1]);
                    let packed_ptr: PointerTile<*mut i8, { [1] }> =
                        q4_i8_1.offset_tile(packed_offset);
                    let (packed_b_1, _packed_tok): (Tile<i8, { [1] }>, Token) = load_ptr_tko(
                        packed_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let scale_hi_offset: Tile<i32, { [1] }> =
                        broadcast_scalar(q4_block_base + group, const_shape![1]);
                    let scale_hi_ptr: PointerTile<*mut i8, { [1] }> =
                        q4_i8_1.offset_tile(scale_hi_offset);
                    let (scale_hi_b_1, _scale_hi_tok): (Tile<i8, { [1] }>, Token) = load_ptr_tko(
                        scale_hi_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let min_hi_offset: Tile<i32, { [1] }> =
                        broadcast_scalar(q4_block_base + 4i32 + group, const_shape![1]);
                    let min_hi_ptr: PointerTile<*mut i8, { [1] }> =
                        q4_i8_1.offset_tile(min_hi_offset);
                    let (min_hi_b_1, _min_hi_tok): (Tile<i8, { [1] }>, Token) = load_ptr_tko(
                        min_hi_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let packed_ext_1: Tile<i32, { [1] }> = exti(packed_b_1);
                    let scale_hi_ext_1: Tile<i32, { [1] }> = exti(scale_hi_b_1);
                    let min_hi_ext_1: Tile<i32, { [1] }> = exti(min_hi_b_1);
                    let c255_1: Tile<i32, { [1] }> = constant(255i32, const_shape![1]);
                    let packed: Tile<i32, { [] }> = (packed_ext_1 & c255_1).reshape(const_shape![]);
                    let scale_hi: Tile<i32, { [] }> =
                        (scale_hi_ext_1 & c255_1).reshape(const_shape![]);
                    let min_hi: Tile<i32, { [] }> = (min_hi_ext_1 & c255_1).reshape(const_shape![]);
                    let c4: Tile<i32, { [] }> = constant(4i32, const_shape![]);
                    let c6: Tile<i32, { [] }> = constant(6i32, const_shape![]);
                    let c15: Tile<i32, { [] }> = constant(15i32, const_shape![]);
                    scale_i32 = (packed & c15) | shli(shri(scale_hi, c6), c4, overflow::NoWrap);
                    min_i32 = shri(packed, c4) | shli(shri(min_hi, c6), c4, overflow::NoWrap);
                }

                let scale_f32: Tile<f32, { [] }> = convert_tile(scale_i32);
                let min_f32: Tile<f32, { [] }> = convert_tile(min_i32);
                let scale32: Tile<f32, { [32] }> = scale_f32
                    .reshape(const_shape![1])
                    .broadcast(const_shape![32]);
                let min32: Tile<f32, { [32] }> =
                    min_f32.reshape(const_shape![1]).broadcast(const_shape![32]);

                let q8_block_index: i32 = block * 8i32 + group;
                let q8_block_base: i32 = batch * q8_row_stride_bytes + q8_block_index * 36i32;
                let d8_offset: Tile<i32, { [1] }> =
                    broadcast_scalar(q8_block_base / 2i32, const_shape![1]);
                let d8_ptr: PointerTile<*mut f16, { [1] }> = q8_half_1.offset_tile(d8_offset);
                let (d8_h_1, _d8_tok): (Tile<f16, { [1] }>, Token) = load_ptr_tko(
                    d8_ptr,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None,
                    Latency::<0>,
                );
                let sum8_offset: Tile<i32, { [1] }> =
                    broadcast_scalar(q8_block_base / 2i32 + 1i32, const_shape![1]);
                let sum8_ptr: PointerTile<*mut f16, { [1] }> = q8_half_1.offset_tile(sum8_offset);
                let (sum8_h_1, _sum8_tok): (Tile<f16, { [1] }>, Token) = load_ptr_tko(
                    sum8_ptr,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None,
                    Latency::<0>,
                );
                let d8_1: Tile<f32, { [1] }> = ftof(d8_h_1, rounding::NearestEven);
                let sum8_1: Tile<f32, { [1] }> = ftof(sum8_h_1, rounding::NearestEven);
                let d8: Tile<f32, { [] }> = d8_1.reshape(const_shape![]);
                let sum8: Tile<f32, { [] }> = sum8_1.reshape(const_shape![]);
                let d8_32: Tile<f32, { [32] }> =
                    d8.reshape(const_shape![1]).broadcast(const_shape![32]);

                let q8_total_offsets: Tile<i32, { [32] }> =
                    offsets + broadcast_scalar(q8_block_base + 4i32, const_shape![32]);
                let q8_i8_32: PointerTile<*mut i8, { [32] }> = q8_i8_1.broadcast(const_shape![32]);
                let q8_ptrs: PointerTile<*mut i8, { [32] }> =
                    q8_i8_32.offset_tile(q8_total_offsets);
                let (q8_bytes, _q8_tok): (Tile<i8, { [32] }>, Token) = load_ptr_tko(
                    q8_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let q8_i32: Tile<i32, { [32] }> = exti(q8_bytes);
                let q8_f32: Tile<f32, { [32] }> = convert_tile(q8_i32);

                if b_size > 8i32 {
                    // Candle's prompt/prefill MMQ path uses the original F32 sum
                    // stored in Q8_1.ds.y for the Q4_K min term. Match that
                    // numerics for b_size > 8 rather than the MMVQ decode formula.
                    let dot_prod: Tile<f32, { [] }> = reduce_sum(q4_f32 * q8_f32, 0i32);
                    let d_term: Tile<f32, { [] }> = d * scale_f32 * d8 * dot_prod;
                    let m_term: Tile<f32, { [] }> = dmin * min_f32 * sum8;
                    acc = acc + d_term - m_term;
                } else {
                    let q4_val: Tile<f32, { [32] }> = d32 * scale32 * q4_f32 - dmin32 * min32;
                    let q8_val: Tile<f32, { [32] }> = d8_32 * q8_f32;
                    let prod: Tile<f32, { [32] }> = q4_val * q8_val;
                    let group_sum: Tile<f32, { [] }> = reduce_sum(prod, 0i32);
                    acc = acc + group_sum;
                }
            }
        }

        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_1: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_offset: Tile<i32, { [1] }> = broadcast_scalar(batch * nrows + row, const_shape![1]);
        let out_dst: PointerTile<*mut f32, { [1] }> = out_1.offset_tile(out_offset);
        store_ptr_tko(
            out_dst,
            acc.reshape(const_shape![1]),
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<0>,
        );
    }
}

#[cutile::module]
mod q4k_q8_1_mmq_matmul_batched_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn q4k_q8_1_mmq_matmul_batched_f32(
        q4_ptr: *mut u8,
        q8_ptr: *mut u8,
        out_ptr: *mut f32,
        ncols: i32,
        nrows: i32,
        b_size: i32,
        _q8_row_stride_bytes: i32,
    ) {
        let pid = get_tile_block_id();
        let row: i32 = pid.0;
        let batch: i32 = pid.1;
        if row >= nrows || batch >= b_size {
            return;
        }

        let q4_base: PointerTile<*mut u8, { [] }> = pointer_to_tile(q4_ptr);
        let q4_half_base: PointerTile<*mut f16, { [] }> = ptr_to_ptr(q4_base);
        let q4_i8_base: PointerTile<*mut i8, { [] }> = ptr_to_ptr(q4_base);
        let q4_half_1: PointerTile<*mut f16, { [1] }> = q4_half_base.reshape(const_shape![1]);
        let q4_i8_1: PointerTile<*mut i8, { [1] }> = q4_i8_base.reshape(const_shape![1]);
        let q8_base: PointerTile<*mut u8, { [] }> = pointer_to_tile(q8_ptr);
        let q8_half_base: PointerTile<*mut f16, { [] }> = ptr_to_ptr(q8_base);
        let q8_i8_base: PointerTile<*mut i8, { [] }> = ptr_to_ptr(q8_base);
        let q8_half_1: PointerTile<*mut f16, { [1] }> = q8_half_base.reshape(const_shape![1]);
        let q8_i8_1: PointerTile<*mut i8, { [1] }> = q8_i8_base.reshape(const_shape![1]);

        let mut acc: Tile<f32, { [] }> = constant(0.0f32, const_shape![]);
        let blocks_per_row: i32 = ncols / 256i32;
        let q4_row_base: i32 = row * blocks_per_row * 144i32;

        for block in 0i32..blocks_per_row {
            let q4_block_base: i32 = q4_row_base + block * 144i32;

            let d_offset: Tile<i32, { [1] }> =
                broadcast_scalar(q4_block_base / 2i32, const_shape![1]);
            let d_ptr: PointerTile<*mut f16, { [1] }> = q4_half_1.offset_tile(d_offset);
            let (d_h_1, _d_tok): (Tile<f16, { [1] }>, Token) = load_ptr_tko(
                d_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let dmin_offset: Tile<i32, { [1] }> =
                broadcast_scalar(q4_block_base / 2i32 + 1i32, const_shape![1]);
            let dmin_ptr: PointerTile<*mut f16, { [1] }> = q4_half_1.offset_tile(dmin_offset);
            let (dmin_h_1, _dmin_tok): (Tile<f16, { [1] }>, Token) = load_ptr_tko(
                dmin_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let d_1: Tile<f32, { [1] }> = ftof(d_h_1, rounding::NearestEven);
            let dmin_1: Tile<f32, { [1] }> = ftof(dmin_h_1, rounding::NearestEven);
            let d: Tile<f32, { [] }> = d_1.reshape(const_shape![]);
            let dmin: Tile<f32, { [] }> = dmin_1.reshape(const_shape![]);
            for group in 0i32..8i32 {
                let q4_byte_base: i32 = q4_block_base + 16i32 + (group / 2i32) * 32i32;
                let offsets: Tile<i32, { [32] }> = iota(const_shape![32]);
                let c15_32: Tile<i32, { [32] }> = constant(15i32, const_shape![32]);
                let c255_32: Tile<i32, { [32] }> = constant(255i32, const_shape![32]);
                let q4_total_offsets: Tile<i32, { [32] }> =
                    offsets + broadcast_scalar(q4_byte_base, const_shape![32]);
                let q4_i8_32: PointerTile<*mut i8, { [32] }> = q4_i8_1.broadcast(const_shape![32]);
                let q4_ptrs: PointerTile<*mut i8, { [32] }> =
                    q4_i8_32.offset_tile(q4_total_offsets);
                let (q4_bytes, _q4_tok): (Tile<i8, { [32] }>, Token) = load_ptr_tko(
                    q4_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let q4_byte_i32: Tile<i32, { [32] }> = exti(q4_bytes) & c255_32;
                let shift_32: Tile<i32, { [32] }> =
                    broadcast_scalar((group % 2i32) * 4i32, const_shape![32]);
                let q4_i32: Tile<i32, { [32] }> = shri(q4_byte_i32, shift_32) & c15_32;
                let q4_f32: Tile<f32, { [32] }> = convert_tile(q4_i32);

                #[allow(unused_assignments)]
                let mut scale_i32: Tile<i32, { [] }> = constant(0i32, const_shape![]);
                #[allow(unused_assignments)]
                let mut min_i32: Tile<i32, { [] }> = constant(0i32, const_shape![]);
                if group < 4i32 {
                    let scale_offset: Tile<i32, { [1] }> =
                        broadcast_scalar(q4_block_base + 4i32 + group, const_shape![1]);
                    let scale_ptr: PointerTile<*mut i8, { [1] }> =
                        q4_i8_1.offset_tile(scale_offset);
                    let (scale_b_1, _scale_tok): (Tile<i8, { [1] }>, Token) = load_ptr_tko(
                        scale_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let min_offset: Tile<i32, { [1] }> =
                        broadcast_scalar(q4_block_base + 8i32 + group, const_shape![1]);
                    let min_ptr: PointerTile<*mut i8, { [1] }> = q4_i8_1.offset_tile(min_offset);
                    let (min_b_1, _min_tok): (Tile<i8, { [1] }>, Token) = load_ptr_tko(
                        min_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let scale_ext_1: Tile<i32, { [1] }> = exti(scale_b_1);
                    let min_ext_1: Tile<i32, { [1] }> = exti(min_b_1);
                    let c63_1: Tile<i32, { [1] }> = constant(63i32, const_shape![1]);
                    scale_i32 = (scale_ext_1 & c63_1).reshape(const_shape![]);
                    min_i32 = (min_ext_1 & c63_1).reshape(const_shape![]);
                } else {
                    let packed_offset: Tile<i32, { [1] }> =
                        broadcast_scalar(q4_block_base + 8i32 + group, const_shape![1]);
                    let packed_ptr: PointerTile<*mut i8, { [1] }> =
                        q4_i8_1.offset_tile(packed_offset);
                    let (packed_b_1, _packed_tok): (Tile<i8, { [1] }>, Token) = load_ptr_tko(
                        packed_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let scale_hi_offset: Tile<i32, { [1] }> =
                        broadcast_scalar(q4_block_base + group, const_shape![1]);
                    let scale_hi_ptr: PointerTile<*mut i8, { [1] }> =
                        q4_i8_1.offset_tile(scale_hi_offset);
                    let (scale_hi_b_1, _scale_hi_tok): (Tile<i8, { [1] }>, Token) = load_ptr_tko(
                        scale_hi_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let min_hi_offset: Tile<i32, { [1] }> =
                        broadcast_scalar(q4_block_base + 4i32 + group, const_shape![1]);
                    let min_hi_ptr: PointerTile<*mut i8, { [1] }> =
                        q4_i8_1.offset_tile(min_hi_offset);
                    let (min_hi_b_1, _min_hi_tok): (Tile<i8, { [1] }>, Token) = load_ptr_tko(
                        min_hi_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let packed_ext_1: Tile<i32, { [1] }> = exti(packed_b_1);
                    let scale_hi_ext_1: Tile<i32, { [1] }> = exti(scale_hi_b_1);
                    let min_hi_ext_1: Tile<i32, { [1] }> = exti(min_hi_b_1);
                    let c255_1: Tile<i32, { [1] }> = constant(255i32, const_shape![1]);
                    let packed: Tile<i32, { [] }> = (packed_ext_1 & c255_1).reshape(const_shape![]);
                    let scale_hi: Tile<i32, { [] }> =
                        (scale_hi_ext_1 & c255_1).reshape(const_shape![]);
                    let min_hi: Tile<i32, { [] }> = (min_hi_ext_1 & c255_1).reshape(const_shape![]);
                    let c4: Tile<i32, { [] }> = constant(4i32, const_shape![]);
                    let c6: Tile<i32, { [] }> = constant(6i32, const_shape![]);
                    let c15: Tile<i32, { [] }> = constant(15i32, const_shape![]);
                    scale_i32 = (packed & c15) | shli(shri(scale_hi, c6), c4, overflow::NoWrap);
                    min_i32 = shri(packed, c4) | shli(shri(min_hi, c6), c4, overflow::NoWrap);
                }

                let scale_f32: Tile<f32, { [] }> = convert_tile(scale_i32);
                let min_f32: Tile<f32, { [] }> = convert_tile(min_i32);
                let q8_block_index: i32 = block * 8i32 + group;
                let q8_mmq_block_index: i32 = q8_block_index / 4i32;
                let q8_mmq_group: i32 = q8_block_index % 4i32;
                let q8_block_base: i32 = (q8_mmq_block_index * b_size + batch) * 144i32;
                let d8_byte_offset: i32 = q8_block_base + q8_mmq_group * 4i32;
                let d8_offset: Tile<i32, { [1] }> =
                    broadcast_scalar(d8_byte_offset / 2i32, const_shape![1]);
                let d8_ptr: PointerTile<*mut f16, { [1] }> = q8_half_1.offset_tile(d8_offset);
                let (d8_h_1, _d8_tok): (Tile<f16, { [1] }>, Token) = load_ptr_tko(
                    d8_ptr,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None,
                    Latency::<0>,
                );
                let sum8_offset: Tile<i32, { [1] }> =
                    broadcast_scalar(d8_byte_offset / 2i32 + 1i32, const_shape![1]);
                let sum8_ptr: PointerTile<*mut f16, { [1] }> = q8_half_1.offset_tile(sum8_offset);
                let (sum8_h_1, _sum8_tok): (Tile<f16, { [1] }>, Token) = load_ptr_tko(
                    sum8_ptr,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None,
                    Latency::<0>,
                );
                let d8_1: Tile<f32, { [1] }> = ftof(d8_h_1, rounding::NearestEven);
                let sum8_1: Tile<f32, { [1] }> = ftof(sum8_h_1, rounding::NearestEven);
                let d8: Tile<f32, { [] }> = d8_1.reshape(const_shape![]);
                let sum8: Tile<f32, { [] }> = sum8_1.reshape(const_shape![]);

                let q8_total_offsets: Tile<i32, { [32] }> = offsets
                    + broadcast_scalar(
                        q8_block_base + 16i32 + q8_mmq_group * 32i32,
                        const_shape![32],
                    );
                let q8_i8_32: PointerTile<*mut i8, { [32] }> = q8_i8_1.broadcast(const_shape![32]);
                let q8_ptrs: PointerTile<*mut i8, { [32] }> =
                    q8_i8_32.offset_tile(q8_total_offsets);
                let (q8_bytes, _q8_tok): (Tile<i8, { [32] }>, Token) = load_ptr_tko(
                    q8_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let q8_i32: Tile<i32, { [32] }> = exti(q8_bytes);
                let q8_f32: Tile<f32, { [32] }> = convert_tile(q8_i32);

                // Candle's prompt/prefill MMQ path uses the original F32 sum
                // stored in Q8_1.ds.y for the Q4_K min term. Match that
                // numerics for b_size > 8 rather than the MMVQ decode formula.
                let dot_prod: Tile<f32, { [] }> = reduce_sum(q4_f32 * q8_f32, 0i32);
                let d_term: Tile<f32, { [] }> = d * scale_f32 * d8 * dot_prod;
                let m_term: Tile<f32, { [] }> = dmin * min_f32 * sum8;
                acc = acc + d_term - m_term;
            }
        }

        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_1: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_offset: Tile<i32, { [1] }> = broadcast_scalar(batch * nrows + row, const_shape![1]);
        let out_dst: PointerTile<*mut f32, { [1] }> = out_1.offset_tile(out_offset);
        store_ptr_tko(
            out_dst,
            acc.reshape(const_shape![1]),
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<0>,
        );
    }
}

#[cutile::module]
mod q6k_q8_1_matvec_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn q6k_q8_1_matvec_b1_f32(
        q6_ptr: *mut u8,
        q8_ptr: *mut u8,
        out_ptr: *mut f32,
        ncols: i32,
        nrows: i32,
    ) {
        let pid = get_tile_block_id();
        let row: i32 = pid.0;
        if row >= nrows {
            return;
        }

        let q6_base: PointerTile<*mut u8, { [] }> = pointer_to_tile(q6_ptr);
        let q6_half_base: PointerTile<*mut f16, { [] }> = ptr_to_ptr(q6_base);
        let q6_i8_base: PointerTile<*mut i8, { [] }> = ptr_to_ptr(q6_base);
        let q6_half_1: PointerTile<*mut f16, { [1] }> = q6_half_base.reshape(const_shape![1]);
        let q6_i8_1: PointerTile<*mut i8, { [1] }> = q6_i8_base.reshape(const_shape![1]);
        let q8_base: PointerTile<*mut u8, { [] }> = pointer_to_tile(q8_ptr);
        let q8_half_base: PointerTile<*mut f16, { [] }> = ptr_to_ptr(q8_base);
        let q8_i8_base: PointerTile<*mut i8, { [] }> = ptr_to_ptr(q8_base);
        let q8_half_1: PointerTile<*mut f16, { [1] }> = q8_half_base.reshape(const_shape![1]);
        let q8_i8_1: PointerTile<*mut i8, { [1] }> = q8_i8_base.reshape(const_shape![1]);

        let lanes: Tile<i32, { [4] }> = iota(const_shape![4]);
        let c3_4: Tile<i32, { [4] }> = constant(3i32, const_shape![4]);
        let c4_4: Tile<i32, { [4] }> = constant(4i32, const_shape![4]);
        let c15_4: Tile<i32, { [4] }> = constant(15i32, const_shape![4]);
        let c32_4: Tile<i32, { [4] }> = constant(32i32, const_shape![4]);
        let c255_4: Tile<i32, { [4] }> = constant(255i32, const_shape![4]);

        let mut acc: Tile<f32, { [] }> = constant(0.0f32, const_shape![]);
        let blocks_per_row: i32 = ncols / 256i32;
        let q6_row_base: i32 = row * blocks_per_row * 210i32;

        for block in 0i32..blocks_per_row {
            let q6_block_base: i32 = q6_row_base + block * 210i32;

            let d_offset: Tile<i32, { [1] }> =
                broadcast_scalar((q6_block_base + 208i32) / 2i32, const_shape![1]);
            let d_ptr: PointerTile<*mut f16, { [1] }> = q6_half_1.offset_tile(d_offset);
            let (d_h_1, _d_tok): (Tile<f16, { [1] }>, Token) = load_ptr_tko(
                d_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let d_1: Tile<f32, { [1] }> = ftof(d_h_1, rounding::NearestEven);
            let d: Tile<f32, { [] }> = d_1.reshape(const_shape![]);
            let d4: Tile<f32, { [4] }> = d.reshape(const_shape![1]).broadcast(const_shape![4]);

            for iqs in 0i32..32i32 {
                let bq8_offset: i32 = 4i32 * (iqs / 16i32) + (iqs % 16i32) / 8i32;
                let scale_offset: i32 = 8i32 * (iqs / 16i32) + (iqs % 16i32) / 4i32;
                let vh_shift: i32 = 2i32 * ((iqs % 16i32) / 8i32);

                let ql_offsets: Tile<i32, { [4] }> =
                    lanes + broadcast_scalar(q6_block_base + 4i32 * iqs, const_shape![4]);
                let q6_i8_4: PointerTile<*mut i8, { [4] }> = q6_i8_1.broadcast(const_shape![4]);
                let ql_ptrs: PointerTile<*mut i8, { [4] }> = q6_i8_4.offset_tile(ql_offsets);
                let (ql_bytes, _ql_tok): (Tile<i8, { [4] }>, Token) = load_ptr_tko(
                    ql_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let ql_i32: Tile<i32, { [4] }> = exti(ql_bytes) & c255_4;

                let qh_index: i32 = 8i32 * (iqs / 16i32) + iqs % 8i32;
                let qh_offsets: Tile<i32, { [4] }> = lanes
                    + broadcast_scalar(q6_block_base + 128i32 + 4i32 * qh_index, const_shape![4]);
                let qh_ptrs: PointerTile<*mut i8, { [4] }> = q6_i8_4.offset_tile(qh_offsets);
                let (qh_bytes, _qh_tok): (Tile<i8, { [4] }>, Token) = load_ptr_tko(
                    qh_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let qh_i32: Tile<i32, { [4] }> = exti(qh_bytes) & c255_4;

                for half in 0i32..2i32 {
                    let ql_shift: Tile<i32, { [4] }> =
                        broadcast_scalar(4i32 * half, const_shape![4]);
                    let qh_shift: Tile<i32, { [4] }> =
                        broadcast_scalar(vh_shift + 4i32 * half, const_shape![4]);
                    let ql_part: Tile<i32, { [4] }> = shri(ql_i32, ql_shift) & c15_4;
                    let qh_part: Tile<i32, { [4] }> = shri(qh_i32, qh_shift) & c3_4;
                    let q6_i32: Tile<i32, { [4] }> =
                        (ql_part | shli(qh_part, c4_4, overflow::NoWrap)) - c32_4;
                    let q6_f32: Tile<f32, { [4] }> = convert_tile(q6_i32);

                    let scale_index: i32 = scale_offset + 4i32 * half;
                    let scale_offset_tile: Tile<i32, { [1] }> =
                        broadcast_scalar(q6_block_base + 192i32 + scale_index, const_shape![1]);
                    let scale_ptr: PointerTile<*mut i8, { [1] }> =
                        q6_i8_1.offset_tile(scale_offset_tile);
                    let (scale_i8, _scale_tok): (Tile<i8, { [1] }>, Token) = load_ptr_tko(
                        scale_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let scale_i32: Tile<i32, { [1] }> = exti(scale_i8);
                    let scale_f32: Tile<f32, { [1] }> = convert_tile(scale_i32);
                    let scale4: Tile<f32, { [4] }> = scale_f32.broadcast(const_shape![4]);

                    let q8_block_index: i32 = block * 8i32 + bq8_offset + 2i32 * half;
                    let q8_block_base: i32 = q8_block_index * 36i32;
                    let d8_offset: Tile<i32, { [1] }> =
                        broadcast_scalar(q8_block_base / 2i32, const_shape![1]);
                    let d8_ptr: PointerTile<*mut f16, { [1] }> = q8_half_1.offset_tile(d8_offset);
                    let (d8_h_1, _d8_tok): (Tile<f16, { [1] }>, Token) = load_ptr_tko(
                        d8_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let d8_1: Tile<f32, { [1] }> = ftof(d8_h_1, rounding::NearestEven);
                    let d8: Tile<f32, { [] }> = d8_1.reshape(const_shape![]);
                    let d8_4: Tile<f32, { [4] }> =
                        d8.reshape(const_shape![1]).broadcast(const_shape![4]);

                    let q8_offsets: Tile<i32, { [4] }> = lanes
                        + broadcast_scalar(
                            q8_block_base + 4i32 + 4i32 * (iqs % 8i32),
                            const_shape![4],
                        );
                    let q8_i8_4: PointerTile<*mut i8, { [4] }> = q8_i8_1.broadcast(const_shape![4]);
                    let q8_ptrs: PointerTile<*mut i8, { [4] }> = q8_i8_4.offset_tile(q8_offsets);
                    let (q8_bytes, _q8_tok): (Tile<i8, { [4] }>, Token) = load_ptr_tko(
                        q8_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        Some(0i8),
                        None,
                        Latency::<0>,
                    );
                    let q8_i32: Tile<i32, { [4] }> = exti(q8_bytes);
                    let q8_f32: Tile<f32, { [4] }> = convert_tile(q8_i32);

                    let prod: Tile<f32, { [4] }> = d4 * scale4 * q6_f32 * d8_4 * q8_f32;
                    let partial: Tile<f32, { [] }> = reduce_sum(prod, 0i32);
                    acc = acc + partial;
                }
            }
        }

        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_1: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_offset: Tile<i32, { [1] }> = broadcast_scalar(row, const_shape![1]);
        let out_dst: PointerTile<*mut f32, { [1] }> = out_1.offset_tile(out_offset);
        store_ptr_tko(
            out_dst,
            acc.reshape(const_shape![1]),
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<0>,
        );
    }
}

#[cutile::module]
mod q6k_q8_1_matmul_batched_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn q6k_q8_1_matmul_batched_f32(
        q6_ptr: *mut u8,
        q8_ptr: *mut u8,
        out_ptr: *mut f32,
        ncols: i32,
        nrows: i32,
        b_size: i32,
        q8_row_stride_bytes: i32,
    ) {
        let pid = get_tile_block_id();
        let row: i32 = pid.0;
        let batch: i32 = pid.1;
        if row >= nrows || batch >= b_size {
            return;
        }

        let q6_base: PointerTile<*mut u8, { [] }> = pointer_to_tile(q6_ptr);
        let q6_half_base: PointerTile<*mut f16, { [] }> = ptr_to_ptr(q6_base);
        let q6_i8_base: PointerTile<*mut i8, { [] }> = ptr_to_ptr(q6_base);
        let q6_half_1: PointerTile<*mut f16, { [1] }> = q6_half_base.reshape(const_shape![1]);
        let q6_i8_1: PointerTile<*mut i8, { [1] }> = q6_i8_base.reshape(const_shape![1]);
        let q8_base: PointerTile<*mut u8, { [] }> = pointer_to_tile(q8_ptr);
        let q8_half_base: PointerTile<*mut f16, { [] }> = ptr_to_ptr(q8_base);
        let q8_i8_base: PointerTile<*mut i8, { [] }> = ptr_to_ptr(q8_base);
        let q8_half_1: PointerTile<*mut f16, { [1] }> = q8_half_base.reshape(const_shape![1]);
        let q8_i8_1: PointerTile<*mut i8, { [1] }> = q8_i8_base.reshape(const_shape![1]);

        let lanes: Tile<i32, { [4] }> = iota(const_shape![4]);
        let c3_4: Tile<i32, { [4] }> = constant(3i32, const_shape![4]);
        let c4_4: Tile<i32, { [4] }> = constant(4i32, const_shape![4]);
        let c15_4: Tile<i32, { [4] }> = constant(15i32, const_shape![4]);
        let c32_4: Tile<i32, { [4] }> = constant(32i32, const_shape![4]);
        let c255_4: Tile<i32, { [4] }> = constant(255i32, const_shape![4]);

        let mut acc: Tile<f32, { [] }> = constant(0.0f32, const_shape![]);
        let blocks_per_row: i32 = ncols / 256i32;
        let q6_row_base: i32 = row * blocks_per_row * 210i32;

        for block in 0i32..blocks_per_row {
            let q6_block_base: i32 = q6_row_base + block * 210i32;

            let d_offset: Tile<i32, { [1] }> =
                broadcast_scalar((q6_block_base + 208i32) / 2i32, const_shape![1]);
            let d_ptr: PointerTile<*mut f16, { [1] }> = q6_half_1.offset_tile(d_offset);
            let (d_h_1, _d_tok): (Tile<f16, { [1] }>, Token) = load_ptr_tko(
                d_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let d_1: Tile<f32, { [1] }> = ftof(d_h_1, rounding::NearestEven);
            let d: Tile<f32, { [] }> = d_1.reshape(const_shape![]);
            let d4: Tile<f32, { [4] }> = d.reshape(const_shape![1]).broadcast(const_shape![4]);

            for iqs in 0i32..32i32 {
                let bq8_offset: i32 = 4i32 * (iqs / 16i32) + (iqs % 16i32) / 8i32;
                let scale_offset: i32 = 8i32 * (iqs / 16i32) + (iqs % 16i32) / 4i32;
                let vh_shift: i32 = 2i32 * ((iqs % 16i32) / 8i32);

                let ql_offsets: Tile<i32, { [4] }> =
                    lanes + broadcast_scalar(q6_block_base + 4i32 * iqs, const_shape![4]);
                let q6_i8_4: PointerTile<*mut i8, { [4] }> = q6_i8_1.broadcast(const_shape![4]);
                let ql_ptrs: PointerTile<*mut i8, { [4] }> = q6_i8_4.offset_tile(ql_offsets);
                let (ql_bytes, _ql_tok): (Tile<i8, { [4] }>, Token) = load_ptr_tko(
                    ql_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let ql_i32: Tile<i32, { [4] }> = exti(ql_bytes) & c255_4;

                let qh_index: i32 = 8i32 * (iqs / 16i32) + iqs % 8i32;
                let qh_offsets: Tile<i32, { [4] }> = lanes
                    + broadcast_scalar(q6_block_base + 128i32 + 4i32 * qh_index, const_shape![4]);
                let qh_ptrs: PointerTile<*mut i8, { [4] }> = q6_i8_4.offset_tile(qh_offsets);
                let (qh_bytes, _qh_tok): (Tile<i8, { [4] }>, Token) = load_ptr_tko(
                    qh_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let qh_i32: Tile<i32, { [4] }> = exti(qh_bytes) & c255_4;

                for half in 0i32..2i32 {
                    let ql_shift: Tile<i32, { [4] }> =
                        broadcast_scalar(4i32 * half, const_shape![4]);
                    let qh_shift: Tile<i32, { [4] }> =
                        broadcast_scalar(vh_shift + 4i32 * half, const_shape![4]);
                    let ql_part: Tile<i32, { [4] }> = shri(ql_i32, ql_shift) & c15_4;
                    let qh_part: Tile<i32, { [4] }> = shri(qh_i32, qh_shift) & c3_4;
                    let q6_i32: Tile<i32, { [4] }> =
                        (ql_part | shli(qh_part, c4_4, overflow::NoWrap)) - c32_4;
                    let q6_f32: Tile<f32, { [4] }> = convert_tile(q6_i32);

                    let scale_index: i32 = scale_offset + 4i32 * half;
                    let scale_offset_tile: Tile<i32, { [1] }> =
                        broadcast_scalar(q6_block_base + 192i32 + scale_index, const_shape![1]);
                    let scale_ptr: PointerTile<*mut i8, { [1] }> =
                        q6_i8_1.offset_tile(scale_offset_tile);
                    let (scale_i8, _scale_tok): (Tile<i8, { [1] }>, Token) = load_ptr_tko(
                        scale_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let scale_i32: Tile<i32, { [1] }> = exti(scale_i8);
                    let scale_f32: Tile<f32, { [1] }> = convert_tile(scale_i32);
                    let scale4: Tile<f32, { [4] }> = scale_f32.broadcast(const_shape![4]);

                    let q8_block_index: i32 = block * 8i32 + bq8_offset + 2i32 * half;
                    let q8_block_base: i32 = batch * q8_row_stride_bytes + q8_block_index * 36i32;
                    let d8_offset: Tile<i32, { [1] }> =
                        broadcast_scalar(q8_block_base / 2i32, const_shape![1]);
                    let d8_ptr: PointerTile<*mut f16, { [1] }> = q8_half_1.offset_tile(d8_offset);
                    let (d8_h_1, _d8_tok): (Tile<f16, { [1] }>, Token) = load_ptr_tko(
                        d8_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let d8_1: Tile<f32, { [1] }> = ftof(d8_h_1, rounding::NearestEven);
                    let d8: Tile<f32, { [] }> = d8_1.reshape(const_shape![]);
                    let d8_4: Tile<f32, { [4] }> =
                        d8.reshape(const_shape![1]).broadcast(const_shape![4]);

                    let q8_offsets: Tile<i32, { [4] }> = lanes
                        + broadcast_scalar(
                            q8_block_base + 4i32 + 4i32 * (iqs % 8i32),
                            const_shape![4],
                        );
                    let q8_i8_4: PointerTile<*mut i8, { [4] }> = q8_i8_1.broadcast(const_shape![4]);
                    let q8_ptrs: PointerTile<*mut i8, { [4] }> = q8_i8_4.offset_tile(q8_offsets);
                    let (q8_bytes, _q8_tok): (Tile<i8, { [4] }>, Token) = load_ptr_tko(
                        q8_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        Some(0i8),
                        None,
                        Latency::<0>,
                    );
                    let q8_i32: Tile<i32, { [4] }> = exti(q8_bytes);
                    let q8_f32: Tile<f32, { [4] }> = convert_tile(q8_i32);

                    let prod: Tile<f32, { [4] }> = d4 * scale4 * q6_f32 * d8_4 * q8_f32;
                    let partial: Tile<f32, { [] }> = reduce_sum(prod, 0i32);
                    acc = acc + partial;
                }
            }
        }

        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_1: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_offset: Tile<i32, { [1] }> = broadcast_scalar(batch * nrows + row, const_shape![1]);
        let out_dst: PointerTile<*mut f32, { [1] }> = out_1.offset_tile(out_offset);
        store_ptr_tko(
            out_dst,
            acc.reshape(const_shape![1]),
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<0>,
        );
    }
}

#[cutile::module]
mod q6k_q8_1_mmq_matmul_batched_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn q6k_q8_1_mmq_matmul_batched_f32(
        q6_ptr: *mut u8,
        q8_ptr: *mut u8,
        out_ptr: *mut f32,
        ncols: i32,
        nrows: i32,
        b_size: i32,
        _q8_row_stride_bytes: i32,
    ) {
        let pid = get_tile_block_id();
        let row: i32 = pid.0;
        let batch: i32 = pid.1;
        if row >= nrows || batch >= b_size {
            return;
        }

        let q6_base: PointerTile<*mut u8, { [] }> = pointer_to_tile(q6_ptr);
        let q6_half_base: PointerTile<*mut f16, { [] }> = ptr_to_ptr(q6_base);
        let q6_i8_base: PointerTile<*mut i8, { [] }> = ptr_to_ptr(q6_base);
        let q6_half_1: PointerTile<*mut f16, { [1] }> = q6_half_base.reshape(const_shape![1]);
        let q6_i8_1: PointerTile<*mut i8, { [1] }> = q6_i8_base.reshape(const_shape![1]);
        let q8_base: PointerTile<*mut u8, { [] }> = pointer_to_tile(q8_ptr);
        let q8_f32_base: PointerTile<*mut f32, { [] }> = ptr_to_ptr(q8_base);
        let q8_i8_base: PointerTile<*mut i8, { [] }> = ptr_to_ptr(q8_base);
        let q8_f32_1: PointerTile<*mut f32, { [1] }> = q8_f32_base.reshape(const_shape![1]);
        let q8_i8_1: PointerTile<*mut i8, { [1] }> = q8_i8_base.reshape(const_shape![1]);

        let lanes: Tile<i32, { [4] }> = iota(const_shape![4]);
        let c3_4: Tile<i32, { [4] }> = constant(3i32, const_shape![4]);
        let c4_4: Tile<i32, { [4] }> = constant(4i32, const_shape![4]);
        let c15_4: Tile<i32, { [4] }> = constant(15i32, const_shape![4]);
        let c32_4: Tile<i32, { [4] }> = constant(32i32, const_shape![4]);
        let c255_4: Tile<i32, { [4] }> = constant(255i32, const_shape![4]);

        let mut acc: Tile<f32, { [] }> = constant(0.0f32, const_shape![]);
        let blocks_per_row: i32 = ncols / 256i32;
        let q6_row_base: i32 = row * blocks_per_row * 210i32;

        for block in 0i32..blocks_per_row {
            let q6_block_base: i32 = q6_row_base + block * 210i32;

            let d_offset: Tile<i32, { [1] }> =
                broadcast_scalar((q6_block_base + 208i32) / 2i32, const_shape![1]);
            let d_ptr: PointerTile<*mut f16, { [1] }> = q6_half_1.offset_tile(d_offset);
            let (d_h_1, _d_tok): (Tile<f16, { [1] }>, Token) = load_ptr_tko(
                d_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let d_1: Tile<f32, { [1] }> = ftof(d_h_1, rounding::NearestEven);
            let d: Tile<f32, { [] }> = d_1.reshape(const_shape![]);
            let d4: Tile<f32, { [4] }> = d.reshape(const_shape![1]).broadcast(const_shape![4]);

            for iqs in 0i32..32i32 {
                let bq8_offset: i32 = 4i32 * (iqs / 16i32) + (iqs % 16i32) / 8i32;
                let scale_offset: i32 = 8i32 * (iqs / 16i32) + (iqs % 16i32) / 4i32;
                let vh_shift: i32 = 2i32 * ((iqs % 16i32) / 8i32);

                let ql_offsets: Tile<i32, { [4] }> =
                    lanes + broadcast_scalar(q6_block_base + 4i32 * iqs, const_shape![4]);
                let q6_i8_4: PointerTile<*mut i8, { [4] }> = q6_i8_1.broadcast(const_shape![4]);
                let ql_ptrs: PointerTile<*mut i8, { [4] }> = q6_i8_4.offset_tile(ql_offsets);
                let (ql_bytes, _ql_tok): (Tile<i8, { [4] }>, Token) = load_ptr_tko(
                    ql_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let ql_i32: Tile<i32, { [4] }> = exti(ql_bytes) & c255_4;

                let qh_index: i32 = 8i32 * (iqs / 16i32) + iqs % 8i32;
                let qh_offsets: Tile<i32, { [4] }> = lanes
                    + broadcast_scalar(q6_block_base + 128i32 + 4i32 * qh_index, const_shape![4]);
                let qh_ptrs: PointerTile<*mut i8, { [4] }> = q6_i8_4.offset_tile(qh_offsets);
                let (qh_bytes, _qh_tok): (Tile<i8, { [4] }>, Token) = load_ptr_tko(
                    qh_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let qh_i32: Tile<i32, { [4] }> = exti(qh_bytes) & c255_4;

                for half in 0i32..2i32 {
                    let ql_shift: Tile<i32, { [4] }> =
                        broadcast_scalar(4i32 * half, const_shape![4]);
                    let qh_shift: Tile<i32, { [4] }> =
                        broadcast_scalar(vh_shift + 4i32 * half, const_shape![4]);
                    let ql_part: Tile<i32, { [4] }> = shri(ql_i32, ql_shift) & c15_4;
                    let qh_part: Tile<i32, { [4] }> = shri(qh_i32, qh_shift) & c3_4;
                    let q6_i32: Tile<i32, { [4] }> =
                        (ql_part | shli(qh_part, c4_4, overflow::NoWrap)) - c32_4;
                    let q6_f32: Tile<f32, { [4] }> = convert_tile(q6_i32);

                    let scale_index: i32 = scale_offset + 4i32 * half;
                    let scale_offset_tile: Tile<i32, { [1] }> =
                        broadcast_scalar(q6_block_base + 192i32 + scale_index, const_shape![1]);
                    let scale_ptr: PointerTile<*mut i8, { [1] }> =
                        q6_i8_1.offset_tile(scale_offset_tile);
                    let (scale_i8, _scale_tok): (Tile<i8, { [1] }>, Token) = load_ptr_tko(
                        scale_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let scale_i32: Tile<i32, { [1] }> = exti(scale_i8);
                    let scale_f32: Tile<f32, { [1] }> = convert_tile(scale_i32);
                    let scale4: Tile<f32, { [4] }> = scale_f32.broadcast(const_shape![4]);

                    let q8_block_index: i32 = block * 8i32 + bq8_offset + 2i32 * half;
                    let q8_mmq_block_index: i32 = q8_block_index / 4i32;
                    let q8_mmq_group: i32 = q8_block_index % 4i32;
                    let q8_block_base: i32 = (q8_mmq_block_index * b_size + batch) * 144i32;
                    let d8_byte_offset: i32 = q8_block_base + q8_mmq_group * 4i32;
                    let d8_offset: Tile<i32, { [1] }> =
                        broadcast_scalar(d8_byte_offset / 4i32, const_shape![1]);
                    let d8_ptr: PointerTile<*mut f32, { [1] }> = q8_f32_1.offset_tile(d8_offset);
                    let (d8_1, _d8_tok): (Tile<f32, { [1] }>, Token) = load_ptr_tko(
                        d8_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let d8: Tile<f32, { [] }> = d8_1.reshape(const_shape![]);
                    let d8_4: Tile<f32, { [4] }> =
                        d8.reshape(const_shape![1]).broadcast(const_shape![4]);

                    let q8_offsets: Tile<i32, { [4] }> = lanes
                        + broadcast_scalar(
                            q8_block_base + 16i32 + q8_mmq_group * 32i32 + 4i32 * (iqs % 8i32),
                            const_shape![4],
                        );
                    let q8_i8_4: PointerTile<*mut i8, { [4] }> = q8_i8_1.broadcast(const_shape![4]);
                    let q8_ptrs: PointerTile<*mut i8, { [4] }> = q8_i8_4.offset_tile(q8_offsets);
                    let (q8_bytes, _q8_tok): (Tile<i8, { [4] }>, Token) = load_ptr_tko(
                        q8_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        Some(0i8),
                        None,
                        Latency::<0>,
                    );
                    let q8_i32: Tile<i32, { [4] }> = exti(q8_bytes);
                    let q8_f32: Tile<f32, { [4] }> = convert_tile(q8_i32);

                    let prod: Tile<f32, { [4] }> = d4 * scale4 * q6_f32 * d8_4 * q8_f32;
                    let partial: Tile<f32, { [] }> = reduce_sum(prod, 0i32);
                    acc = acc + partial;
                }
            }
        }

        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_1: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_offset: Tile<i32, { [1] }> = broadcast_scalar(batch * nrows + row, const_shape![1]);
        let out_dst: PointerTile<*mut f32, { [1] }> = out_1.offset_tile(out_offset);
        store_ptr_tko(
            out_dst,
            acc.reshape(const_shape![1]),
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<0>,
        );
    }
}

#[cutile::module]
mod candle_stream_smoke_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn fill_constant_f32(out_ptr: *mut f32, value: f32, len: i32) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_1d: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [128] }> = out_1d.broadcast(const_shape![128]);

        let value_tile: Tile<f32, { [128] }> = broadcast_scalar(value, const_shape![128]);
        let len_tile: Tile<i32, { [128] }> = broadcast_scalar(len, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..len).step_by(step as usize) {
            let offsets: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let mask: Tile<bool, { [128] }> = lt_tile(offsets, len_tile);
            let dst: PointerTile<*mut f32, { [128] }> = out_ptrs.offset_tile(offsets);
            store_ptr_tko(
                dst,
                value_tile,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            );
        }
    }
}

fn cutile_err(err: impl std::fmt::Display) -> crate::Error {
    crate::Error::msg(err)
}

fn borrow_candle_cuda_handles(
    cuda: &CudaDevice,
) -> Result<(
    std::sync::Arc<cutile::cuda_core::Device>,
    std::sync::Arc<cutile::cuda_core::Stream>,
)> {
    use crate::cuda_backend::WrapErr;
    use core::ffi::{c_int, c_void};

    let candle_stream = cuda.cuda_stream();
    let candle_context = candle_stream.context().clone();
    candle_context.bind_to_thread().w()?;

    // SAFETY: these are non-owning wrappers over Candle/cudarc-owned handles.
    // Candle's CUDA device, context, and stream outlive the borrowed cuTile
    // wrappers for each launch. Dropping the wrappers must not destroy the raw
    // handles; cuTile's `borrow_raw` constructors are explicitly for this use.
    let cutile_device = unsafe {
        cutile::cuda_core::Device::borrow_raw(
            candle_context.cu_ctx() as *mut c_void,
            candle_context.cu_device() as c_int,
            candle_context.ordinal(),
        )
    };
    let cutile_stream = unsafe {
        cutile::cuda_core::Stream::borrow_raw(
            candle_stream.cu_stream() as *mut c_void,
            &cutile_device,
        )
    };

    Ok((cutile_device, cutile_stream))
}

pub(crate) fn try_fwd(
    qstorage: &QCudaStorage,
    self_shape: &Shape,
    rhs: &CudaStorage,
    rhs_l: &crate::Layout,
) -> Result<Option<(CudaStorage, Shape)>> {
    use candle_kernels::ffi;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use cutile::cuda_async::device_buffer::DevicePointer;
    use cutile::cuda_async::device_operation::DeviceOp;
    use cutile::tile_kernel::TileKernel;

    let w_dtype = qstorage.dtype();
    if !matches!(w_dtype, GgmlDType::Q4K | GgmlDType::Q6K) || rhs.dtype() != DType::F32 {
        return Ok(None);
    }

    let (nrows, ncols) = self_shape.dims2()?;
    if !ncols.is_multiple_of(QK_K) {
        return Ok(None);
    }

    let (b_size, k) = match rhs_l.shape().dims() {
        [b, m, k] => (b * m, *k),
        [b, k] => (*b, *k),
        _ => return Ok(None),
    };
    if k != ncols {
        return Ok(None);
    }
    if b_size == 0 {
        return Ok(None);
    }

    let (o1, o2) = match rhs_l.contiguous_offsets() {
        Some(offsets) => offsets,
        None => return Ok(None),
    };

    let dev = qstorage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let rhs_slice = rhs.as_cuda_slice::<f32>()?;
    let rhs_slice = rhs_slice.slice(o1..o2);
    let rhs_ptr = rhs_slice.device_ptr(&stream).0 as *const std::ffi::c_void;

    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let use_mmq_q8_layout = matches!(w_dtype, GgmlDType::Q4K | GgmlDType::Q6K) && b_size > 8;
    let q8_row_stride_bytes = if use_mmq_q8_layout {
        (ncols_padded / Q8_1_MMQ_BLOCK_SIZE) * Q8_1_MMQ_BLOCK_BYTES
    } else {
        (ncols_padded / Q8_1_BLOCK_SIZE) * Q8_1_BLOCK_BYTES
    };
    let scratch_bytes = b_size * q8_row_stride_bytes;
    let mut scratch = unsafe { dev.alloc::<u8>(scratch_bytes)? };
    let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };

    {
        let (scratch_ptr, scratch_write) = scratch.device_ptr_mut(&stream);
        unsafe {
            if use_mmq_q8_layout {
                let quantize = match w_dtype {
                    GgmlDType::Q4K => ffi::launch_mmq_quantize_q8_1_DS4,
                    GgmlDType::Q6K => ffi::launch_mmq_quantize_q8_1_D4,
                    _ => unreachable!("MMQ Q8_1 layout only selected for Q4K/Q6K"),
                };
                quantize(
                    rhs_ptr,
                    std::ptr::null(),
                    scratch_ptr as *mut std::ffi::c_void,
                    0,
                    ncols as i64,
                    ncols as i64,
                    0,
                    0,
                    ncols_padded as i64,
                    b_size as i64,
                    1,
                    1,
                    stream_ptr,
                );
            } else {
                ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                    rhs_ptr,
                    scratch_ptr as *mut std::ffi::c_void,
                    ncols as i32,
                    ncols_padded as i32,
                    b_size as i32,
                    stream_ptr,
                );
            }
        }

        let (qweight_ptr, qweight_read) = qstorage.device_ptr_with_guard(&stream)?;
        let (out_ptr, out_write) = out.device_ptr_mut(&stream);

        {
            let (_cutile_device, cutile_stream) = borrow_candle_cuda_handles(dev)?;
            match w_dtype {
                GgmlDType::Q4K => {
                    if b_size == 1 {
                        let qweight_cutile = unsafe {
                            DevicePointer::<u8>::from_cu_deviceptr(
                                qweight_ptr as cutile::cuda_core::sys::CUdeviceptr,
                            )
                        };
                        let q8_cutile = unsafe {
                            DevicePointer::<u8>::from_cu_deviceptr(
                                scratch_ptr as cutile::cuda_core::sys::CUdeviceptr,
                            )
                        };
                        let out_cutile = unsafe {
                            DevicePointer::<f32>::from_cu_deviceptr(
                                out_ptr as cutile::cuda_core::sys::CUdeviceptr,
                            )
                        };
                        let op = unsafe {
                            q4k_q8_1_matvec_kernel::q4k_q8_1_matvec_b1_f32(
                                qweight_cutile,
                                q8_cutile,
                                out_cutile,
                                ncols as i32,
                                nrows as i32,
                            )
                        }
                        .grid((nrows as u32, 1, 1));

                        // SAFETY: all pointers refer to live Candle-owned CUDA allocations.
                        // The q4 storage and q8 scratch are read-only for this launch; `out`
                        // is exclusively written. Quantization and matvec are enqueued on the
                        // same Candle stream, so stream order makes the scratch contents
                        // visible to cuTile. Candle/cudarc records the output write after the
                        // cuTile launch below.
                        unsafe { op.async_on(&cutile_stream) }.map_err(cutile_err)?;
                    } else if use_mmq_q8_layout {
                        let qweight_cutile = unsafe {
                            DevicePointer::<u8>::from_cu_deviceptr(
                                qweight_ptr as cutile::cuda_core::sys::CUdeviceptr,
                            )
                        };
                        let q8_cutile = unsafe {
                            DevicePointer::<u8>::from_cu_deviceptr(
                                scratch_ptr as cutile::cuda_core::sys::CUdeviceptr,
                            )
                        };
                        let out_cutile = unsafe {
                            DevicePointer::<f32>::from_cu_deviceptr(
                                out_ptr as cutile::cuda_core::sys::CUdeviceptr,
                            )
                        };
                        let op = unsafe {
                            q4k_q8_1_mmq_matmul_batched_kernel::q4k_q8_1_mmq_matmul_batched_f32(
                                qweight_cutile,
                                q8_cutile,
                                out_cutile,
                                ncols as i32,
                                nrows as i32,
                                b_size as i32,
                                q8_row_stride_bytes as i32,
                            )
                        }
                        .grid((nrows as u32, b_size as u32, 1));

                        // SAFETY: same Candle-owned allocation and stream-ordering argument
                        // as the decode launch above, using Candle's MMQ Q8_1 scratch layout.
                        unsafe { op.async_on(&cutile_stream) }.map_err(cutile_err)?;
                    } else {
                        let qweight_cutile = unsafe {
                            DevicePointer::<u8>::from_cu_deviceptr(
                                qweight_ptr as cutile::cuda_core::sys::CUdeviceptr,
                            )
                        };
                        let q8_cutile = unsafe {
                            DevicePointer::<u8>::from_cu_deviceptr(
                                scratch_ptr as cutile::cuda_core::sys::CUdeviceptr,
                            )
                        };
                        let out_cutile = unsafe {
                            DevicePointer::<f32>::from_cu_deviceptr(
                                out_ptr as cutile::cuda_core::sys::CUdeviceptr,
                            )
                        };
                        let op = unsafe {
                            q4k_q8_1_matmul_batched_kernel::q4k_q8_1_matmul_batched_f32(
                                qweight_cutile,
                                q8_cutile,
                                out_cutile,
                                ncols as i32,
                                nrows as i32,
                                b_size as i32,
                                q8_row_stride_bytes as i32,
                            )
                        }
                        .grid((nrows as u32, b_size as u32, 1));

                        // SAFETY: same Candle-owned allocation and stream-ordering argument
                        // as the decode launch above, extended to one output row per RHS row.
                        unsafe { op.async_on(&cutile_stream) }.map_err(cutile_err)?;
                    }
                }
                GgmlDType::Q6K => {
                    if b_size == 1 {
                        let qweight_cutile = unsafe {
                            DevicePointer::<u8>::from_cu_deviceptr(
                                qweight_ptr as cutile::cuda_core::sys::CUdeviceptr,
                            )
                        };
                        let q8_cutile = unsafe {
                            DevicePointer::<u8>::from_cu_deviceptr(
                                scratch_ptr as cutile::cuda_core::sys::CUdeviceptr,
                            )
                        };
                        let out_cutile = unsafe {
                            DevicePointer::<f32>::from_cu_deviceptr(
                                out_ptr as cutile::cuda_core::sys::CUdeviceptr,
                            )
                        };
                        let op = unsafe {
                            q6k_q8_1_matvec_kernel::q6k_q8_1_matvec_b1_f32(
                                qweight_cutile,
                                q8_cutile,
                                out_cutile,
                                ncols as i32,
                                nrows as i32,
                            )
                        }
                        .grid((nrows as u32, 1, 1));

                        // SAFETY: same ownership/stream-ordering argument as the Q4K
                        // launch above, with Q6K storage as the read-only weight input.
                        unsafe { op.async_on(&cutile_stream) }.map_err(cutile_err)?;
                    } else if use_mmq_q8_layout {
                        let qweight_cutile = unsafe {
                            DevicePointer::<u8>::from_cu_deviceptr(
                                qweight_ptr as cutile::cuda_core::sys::CUdeviceptr,
                            )
                        };
                        let q8_cutile = unsafe {
                            DevicePointer::<u8>::from_cu_deviceptr(
                                scratch_ptr as cutile::cuda_core::sys::CUdeviceptr,
                            )
                        };
                        let out_cutile = unsafe {
                            DevicePointer::<f32>::from_cu_deviceptr(
                                out_ptr as cutile::cuda_core::sys::CUdeviceptr,
                            )
                        };
                        let op = unsafe {
                            q6k_q8_1_mmq_matmul_batched_kernel::q6k_q8_1_mmq_matmul_batched_f32(
                                qweight_cutile,
                                q8_cutile,
                                out_cutile,
                                ncols as i32,
                                nrows as i32,
                                b_size as i32,
                                q8_row_stride_bytes as i32,
                            )
                        }
                        .grid((nrows as u32, b_size as u32, 1));

                        // SAFETY: same Candle-owned allocation and stream-ordering argument
                        // as the decode launch above, using Candle's MMQ Q8_1 scratch layout.
                        unsafe { op.async_on(&cutile_stream) }.map_err(cutile_err)?;
                    } else {
                        let qweight_cutile = unsafe {
                            DevicePointer::<u8>::from_cu_deviceptr(
                                qweight_ptr as cutile::cuda_core::sys::CUdeviceptr,
                            )
                        };
                        let q8_cutile = unsafe {
                            DevicePointer::<u8>::from_cu_deviceptr(
                                scratch_ptr as cutile::cuda_core::sys::CUdeviceptr,
                            )
                        };
                        let out_cutile = unsafe {
                            DevicePointer::<f32>::from_cu_deviceptr(
                                out_ptr as cutile::cuda_core::sys::CUdeviceptr,
                            )
                        };
                        let op = unsafe {
                            q6k_q8_1_matmul_batched_kernel::q6k_q8_1_matmul_batched_f32(
                                qweight_cutile,
                                q8_cutile,
                                out_cutile,
                                ncols as i32,
                                nrows as i32,
                                b_size as i32,
                                q8_row_stride_bytes as i32,
                            )
                        }
                        .grid((nrows as u32, b_size as u32, 1));

                        // SAFETY: same Candle-owned allocation and stream-ordering argument
                        // as the decode launch above, extended to one output row per RHS row.
                        unsafe { op.async_on(&cutile_stream) }.map_err(cutile_err)?;
                    }
                }
                _ => unreachable!("unsupported cuTile dtype checked above"),
            }
        }

        drop(out_write);
        drop(qweight_read);
        drop(scratch_write);
    }

    let mut out_shape = rhs_l.shape().dims().to_vec();
    out_shape.pop();
    out_shape.push(nrows);
    Ok(Some((
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        out_shape.into(),
    )))
}

/// Launches a tiny cuTile raw-pointer kernel on Candle's CUDA stream.
///
/// This is intentionally only an interop smoke test. It proves the production
/// backend path can borrow Candle's CUDA handles, enqueue cuTile work on
/// Candle's stream, write into a Candle-owned allocation, then let Candle handle
/// synchronization and subsequent CUDA work.
#[allow(dead_code)]
pub(crate) fn borrowed_candle_stream_smoke(cuda: &CudaDevice) -> Result<Vec<f32>> {
    use cudarc::driver::DevicePtrMut;
    use cutile::cuda_async::device_buffer::DevicePointer;
    use cutile::cuda_async::device_operation::DeviceOp;
    use cutile::tile_kernel::TileKernel;

    const LEN: usize = 257;
    const VALUE: f32 = 3.25;

    let mut out = unsafe { cuda.alloc::<f32>(LEN)? };

    {
        let candle_stream = cuda.cuda_stream();
        let (out_ptr, record_out_write) = out.device_ptr_mut(&candle_stream);
        let cutile_out = unsafe {
            DevicePointer::<f32>::from_cu_deviceptr(out_ptr as cutile::cuda_core::sys::CUdeviceptr)
        };

        {
            let (_cutile_device, cutile_stream) = borrow_candle_cuda_handles(cuda)?;
            let blocks = (LEN as u32).div_ceil(128);
            let op = unsafe {
                candle_stream_smoke_kernel::fill_constant_f32(cutile_out, VALUE, LEN as i32)
            }
            .grid((blocks, 1, 1));

            // SAFETY: `cutile_out` points at `out`, a live Candle-owned CUDA
            // allocation of LEN f32 values. The kernel writes each element at
            // most once. `record_out_write` keeps Candle/cudarc's mutable
            // access record alive until after the launch is enqueued.
            unsafe { op.async_on(&cutile_stream) }.map_err(cutile_err)?;
        }

        // Record Candle/cudarc's write event after the cuTile kernel launch has
        // been queued on the same underlying CUDA stream.
        drop(record_out_write);
    }

    // Synchronize through Candle/cudarc, not through cuTile. Candle remains the
    // synchronization owner in the production path.
    cuda.synchronize()?;
    let values = cuda.clone_dtoh(&out)?;

    // Prove dropping borrowed cuTile wrappers did not destroy Candle's handles:
    // after the wrappers are gone, submit another Candle-owned CUDA allocation
    // and copy it back successfully.
    let probe = cuda.alloc_zeros::<f32>(4)?;
    let probe_values = cuda.clone_dtoh(&probe)?;
    if probe_values != [0.0; 4] {
        crate::bail!("Candle CUDA probe after cuTile wrapper drop failed: {probe_values:?}");
    }

    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{quantized, Device, Module, Tensor};

    static QUANT_KERNEL_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct EnvVarGuard {
        name: &'static str,
        value: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn save(name: &'static str) -> Self {
            Self {
                name,
                value: std::env::var_os(name),
            }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(value) = &self.value {
                std::env::set_var(self.name, value);
            } else {
                std::env::remove_var(self.name);
            }
        }
    }

    #[test]
    #[ignore = "requires CUDA 13.2+/cuTile runtime and a CUDA device"]
    fn cutile_borrowed_candle_stream_smoke_fills_candle_allocation() {
        let device = Device::new_cuda(0).unwrap();
        let cuda = device.as_cuda_device().unwrap();

        let values = borrowed_candle_stream_smoke(cuda).unwrap();
        assert_eq!(values.len(), 257);
        for (index, value) in values.iter().enumerate() {
            assert_eq!(*value, 3.25, "unexpected value at index {index}");
        }

        // A second run exercises the same Candle context/stream after all
        // borrowed cuTile wrappers from the first run have been dropped.
        let values = borrowed_candle_stream_smoke(cuda).unwrap();
        assert!(values.iter().all(|value| *value == 3.25));
    }

    fn run_cutile_qk_q8_1_b1_matches_candle_cuda_matvec(dtype: GgmlDType) {
        let _env_guard = QUANT_KERNEL_ENV_LOCK.lock().unwrap();
        std::thread::Builder::new()
            .name("pi-ai-candle-worker".to_string())
            .spawn(move || {
                std::env::set_var("PI_CANDLE_QUANT_KERNEL", "candle");
                std::env::remove_var("PI_CANDLE_QUANT_FALLBACK");

                let device = Device::new_cuda(0).unwrap();
                let cuda = device.as_cuda_device().unwrap();
                let nrows = 9728usize;
                let ncols = 2560usize;

                let weights = (0..nrows * ncols)
                    .map(|i| {
                        let phase = (i % 251) as f32;
                        (phase * 0.013).sin() * 4.0 + ((i / ncols) as f32 - 8.0) * 0.07
                    })
                    .collect::<Vec<_>>();
                let x = (0..ncols)
                    .map(|i| ((i % 97) as f32 * 0.021).cos() * 3.0 - 1.5)
                    .collect::<Vec<_>>();

                let weights = Tensor::from_slice(&weights, (nrows, ncols), &device).unwrap();
                let x = Tensor::from_slice(&x, (1usize, 1usize, ncols), &device).unwrap();
                let qtensor = quantized::QTensor::quantize(&weights, dtype).unwrap();
                let matmul = quantized::QMatMul::from_qtensor(qtensor).unwrap();

                let expected = matmul.forward(&x).unwrap();
                cuda.synchronize().unwrap();
                let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();

                std::env::set_var("PI_CANDLE_QUANT_KERNEL", "cutile");
                let actual = matmul.forward(&x).unwrap();
                cuda.synchronize().unwrap();
                let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                std::env::set_var("PI_CANDLE_QUANT_KERNEL", "candle");

                for (index, (expected, actual)) in expected.iter().zip(actual.iter()).enumerate() {
                    let abs_err = (expected - actual).abs();
                    let rel_err = abs_err / expected.abs().max(1.0);
                    assert!(
                        abs_err <= 2.0e-2 || rel_err <= 2.0e-4,
                        "unexpected {dtype:?} mismatch at index {index}: candle={expected} cutile={actual} abs_err={abs_err} rel_err={rel_err}"
                    );
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    fn run_cutile_qk_q8_1_batched_matches_candle_cuda_matmul(
        dtype: GgmlDType,
        nrows: usize,
        ncols: usize,
        b_size: usize,
    ) {
        let _env_guard = QUANT_KERNEL_ENV_LOCK.lock().unwrap();
        std::thread::Builder::new()
            .name("pi-ai-candle-worker".to_string())
            .spawn(move || {
                std::env::set_var("PI_CANDLE_QUANT_KERNEL", "candle");
                std::env::remove_var("PI_CANDLE_QUANT_FALLBACK");

                let device = Device::new_cuda(0).unwrap();
                let cuda = device.as_cuda_device().unwrap();

                let weights = (0..nrows * ncols)
                    .map(|i| {
                        let phase = (i % 257) as f32;
                        (phase * 0.011).sin() * 3.0 + ((i / ncols) as f32 - 4.0) * 0.03
                    })
                    .collect::<Vec<_>>();
                let x = (0..b_size * ncols)
                    .map(|i| ((i % 131) as f32 * 0.017).cos() * 2.0 - 0.75)
                    .collect::<Vec<_>>();

                let weights = Tensor::from_slice(&weights, (nrows, ncols), &device).unwrap();
                let x = Tensor::from_slice(&x, (1usize, b_size, ncols), &device).unwrap();
                let qtensor = quantized::QTensor::quantize(&weights, dtype).unwrap();
                let matmul = quantized::QMatMul::from_qtensor(qtensor).unwrap();

                let expected = matmul.forward(&x).unwrap();
                cuda.synchronize().unwrap();
                let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();

                std::env::set_var("PI_CANDLE_QUANT_KERNEL", "cutile");
                let actual = matmul.forward(&x).unwrap();
                cuda.synchronize().unwrap();
                let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                std::env::set_var("PI_CANDLE_QUANT_KERNEL", "candle");

                assert_eq!(expected.len(), b_size * nrows);
                assert_eq!(actual.len(), expected.len());
                let mut max_abs_err = 0.0f32;
                let mut max_abs_index = 0usize;
                let mut max_rel_err = 0.0f32;
                let mut max_rel_index = 0usize;
                let mut sum_sq_err = 0.0f64;
                let mut sum_sq_expected = 0.0f64;
                for (index, (expected, actual)) in expected.iter().zip(actual.iter()).enumerate() {
                    let abs_err = (expected - actual).abs();
                    let rel_err = abs_err / expected.abs().max(1.0);
                    if abs_err > max_abs_err {
                        max_abs_err = abs_err;
                        max_abs_index = index;
                    }
                    if rel_err > max_rel_err {
                        max_rel_err = rel_err;
                        max_rel_index = index;
                    }
                    sum_sq_err += f64::from(abs_err) * f64::from(abs_err);
                    sum_sq_expected += f64::from(*expected) * f64::from(*expected);
                }
                let rmse = (sum_sq_err / expected.len() as f64).sqrt();
                let expected_rms = (sum_sq_expected / expected.len() as f64).sqrt();
                let normalized_rmse = rmse / expected_rms.max(1.0e-12);
                assert!(
                    normalized_rmse <= 1.0e-2,
                    "unexpected {dtype:?} batched mismatch: max_abs_err={max_abs_err} max_abs_index={max_abs_index} max_rel_err={max_rel_err} max_rel_index={max_rel_index} rmse={rmse} expected_rms={expected_rms} normalized_rmse={normalized_rmse}"
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    #[ignore = "requires CUDA 13.2+/cuTile runtime and a CUDA device"]
    fn cutile_missing_cuda_toolkit_path_guard_is_clear() {
        let _env_guard = QUANT_KERNEL_ENV_LOCK.lock().unwrap();
        let _kernel_guard = EnvVarGuard::save("PI_CANDLE_QUANT_KERNEL");
        let _fallback_guard = EnvVarGuard::save("PI_CANDLE_QUANT_FALLBACK");
        let _cuda_toolkit_guard = EnvVarGuard::save("CUDA_TOOLKIT_PATH");

        std::thread::Builder::new()
            .name("pi-ai-candle-worker".to_string())
            .spawn(|| {
                std::env::set_var("PI_CANDLE_QUANT_KERNEL", "candle");
                std::env::remove_var("PI_CANDLE_QUANT_FALLBACK");
                std::env::remove_var("CUDA_TOOLKIT_PATH");

                let device = Device::new_cuda(0).unwrap();
                let cuda = device.as_cuda_device().unwrap();
                let nrows = 32usize;
                let ncols = 256usize;

                let weights = (0..nrows * ncols)
                    .map(|i| ((i % 113) as f32 * 0.019).sin() * 2.0 - 0.5)
                    .collect::<Vec<_>>();
                let x = (0..ncols)
                    .map(|i| ((i % 67) as f32 * 0.031).cos() * 1.5)
                    .collect::<Vec<_>>();

                let weights = Tensor::from_slice(&weights, (nrows, ncols), &device).unwrap();
                let x = Tensor::from_slice(&x, (1usize, 1usize, ncols), &device).unwrap();
                let qtensor = quantized::QTensor::quantize(&weights, GgmlDType::Q4K).unwrap();
                let matmul = quantized::QMatMul::from_qtensor(qtensor).unwrap();

                let expected = matmul.forward(&x).unwrap();
                cuda.synchronize().unwrap();
                let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();

                std::env::set_var("PI_CANDLE_QUANT_KERNEL", "cutile");
                std::env::remove_var("PI_CANDLE_QUANT_FALLBACK");
                let error = match matmul.forward(&x) {
                    Ok(_) => panic!("strict cuTile unexpectedly ran without CUDA_TOOLKIT_PATH"),
                    Err(error) => error.to_string(),
                };
                assert!(
                    error.contains("requires CUDA_TOOLKIT_PATH"),
                    "unexpected strict missing-CUDA_TOOLKIT_PATH error: {error}"
                );

                std::env::set_var("PI_CANDLE_QUANT_FALLBACK", "candle");
                let actual = matmul.forward(&x).unwrap();
                cuda.synchronize().unwrap();
                let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();

                for (index, (expected, actual)) in expected.iter().zip(actual.iter()).enumerate() {
                    assert_eq!(
                        expected, actual,
                        "fallback-to-Candle mismatch at index {index}: candle={expected} fallback={actual}"
                    );
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    #[ignore = "requires CUDA 13.2+/cuTile runtime and a CUDA device"]
    fn cutile_q4k_q8_1_b1_matches_candle_cuda_matvec() {
        run_cutile_qk_q8_1_b1_matches_candle_cuda_matvec(GgmlDType::Q4K);
    }

    #[test]
    #[ignore = "requires CUDA 13.2+/cuTile runtime and a CUDA device"]
    fn cutile_q4k_q8_1_batched_matches_candle_cuda_matmul() {
        run_cutile_qk_q8_1_batched_matches_candle_cuda_matmul(GgmlDType::Q4K, 4096, 2560, 372);
    }

    #[test]
    #[ignore = "requires CUDA 13.2+/cuTile runtime and a CUDA device"]
    fn cutile_q6k_q8_1_batched_matches_candle_cuda_matmul() {
        run_cutile_qk_q8_1_batched_matches_candle_cuda_matmul(GgmlDType::Q6K, 1024, 2560, 372);
    }

    #[test]
    #[ignore = "requires CUDA 13.2+/cuTile runtime and a CUDA device"]
    fn cutile_q6k_q8_1_b1_matches_candle_cuda_matvec() {
        run_cutile_qk_q8_1_b1_matches_candle_cuda_matvec(GgmlDType::Q6K);
    }
}
