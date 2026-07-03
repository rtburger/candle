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
    cuda_backend::DeviceId,
    CudaDevice, CudaStorage, DType, Result, Shape,
};
use cudarc::driver::CudaSlice;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const QK_K: usize = 256;
const Q8_1_BLOCK_SIZE: usize = 32;
const Q8_1_BLOCK_BYTES: usize = 36;
const Q8_1_MMQ_BLOCK_SIZE: usize = 4 * Q8_1_BLOCK_SIZE;
const Q8_1_MMQ_BLOCK_BYTES: usize = 4 * Q8_1_BLOCK_BYTES;

#[inline]
fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

fn tiled_prefill_enabled() -> bool {
    matches!(
        std::env::var("PI_CANDLE_QUANT_CUTILE_TILED")
            .ok()
            .as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

fn mmai_prefill_enabled() -> bool {
    matches!(
        std::env::var("PI_CANDLE_QUANT_CUTILE_MMAI").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

fn q4k_b1_rows4_enabled() -> bool {
    matches!(
        std::env::var("PI_CANDLE_QUANT_CUTILE_Q4K_B1_ROWS4")
            .ok()
            .as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

fn q6k_b1_rows4_enabled() -> bool {
    matches!(
        std::env::var("PI_CANDLE_QUANT_CUTILE_Q6K_B1_ROWS4")
            .ok()
            .as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

fn q6k_b1_rows8_enabled() -> bool {
    matches!(
        std::env::var("PI_CANDLE_QUANT_CUTILE_Q6K_B1_ROWS8")
            .ok()
            .as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

fn cutile_trace_enabled() -> bool {
    matches!(
        std::env::var("PI_CANDLE_QUANT_TRACE").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

fn trace_cutile_measurement(message: impl std::fmt::Display) {
    if cutile_trace_enabled() {
        eprintln!("candle quant-kernel {message}");
    }
}

fn record_timing_event(
    stream: &cudarc::driver::CudaStream,
    enabled: bool,
) -> Result<Option<cudarc::driver::CudaEvent>> {
    if !enabled {
        return Ok(None);
    }
    use crate::cuda_backend::WrapErr;
    let event = stream
        .context()
        .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .w()?;
    event.record(stream).w()?;
    Ok(Some(event))
}

fn elapsed_timing_ms(
    events: &Option<(cudarc::driver::CudaEvent, cudarc::driver::CudaEvent)>,
) -> Result<Option<f32>> {
    use crate::cuda_backend::WrapErr;
    match events {
        Some((start, end)) => Ok(Some(start.elapsed_ms(end).w()?)),
        None => Ok(None),
    }
}

fn fmt_timing_ms(value: Option<f32>) -> String {
    match value {
        Some(value) => format!("{value:.3}"),
        None => "n/a".to_string(),
    }
}

fn fmt_host_ms(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{value:.3}"),
        None => "n/a".to_string(),
    }
}

fn fmt_usize(value: Option<usize>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "n/a".to_string(),
    }
}

fn fmt_gbps(value: Option<f32>) -> String {
    match value {
        Some(value) => format!("{value:.2}"),
        None => "n/a".to_string(),
    }
}

fn estimated_decode_weight_bytes(dtype: GgmlDType, nrows: usize, ncols: usize) -> Option<usize> {
    if !ncols.is_multiple_of(QK_K) {
        return None;
    }
    let block_bytes = match dtype {
        GgmlDType::Q4K => 144usize,
        GgmlDType::Q6K => 210usize,
        _ => return None,
    };
    Some(nrows * (ncols / QK_K) * block_bytes)
}

fn estimated_gbps(bytes: Option<usize>, gpu_ms: Option<f32>) -> Option<f32> {
    let bytes = bytes?;
    let gpu_ms = gpu_ms?;
    if gpu_ms <= 0.0 {
        return None;
    }
    Some(bytes as f32 / (gpu_ms / 1_000.0) / 1.0e9)
}

struct Q8ScratchWorkspaceSlot {
    slice: CudaSlice<u8>,
    cap: usize,
}

type Q8ScratchWorkspaceMap = Mutex<HashMap<DeviceId, &'static Mutex<Q8ScratchWorkspaceSlot>>>;

static Q8_SCRATCH_WORKSPACE: OnceLock<Q8ScratchWorkspaceMap> = OnceLock::new();

fn q8_scratch_workspace_ensure(
    dev: &CudaDevice,
    bytes: usize,
) -> Result<std::sync::MutexGuard<'static, Q8ScratchWorkspaceSlot>> {
    let map = Q8_SCRATCH_WORKSPACE.get_or_init(|| Mutex::new(HashMap::new()));
    let device_key = dev.id();
    let device_mtx: &'static Mutex<Q8ScratchWorkspaceSlot> = {
        let mut guard = map.lock().unwrap();
        match guard.get(&device_key).copied() {
            Some(mtx) => mtx,
            None => {
                let slice = unsafe { dev.alloc::<u8>(bytes.max(1))? };
                let leaked = Box::leak(Box::new(Mutex::new(Q8ScratchWorkspaceSlot {
                    slice,
                    cap: bytes.max(1),
                })));
                guard.insert(device_key, leaked);
                leaked
            }
        }
    };
    let mut slot = device_mtx.lock().unwrap();
    if slot.cap < bytes {
        slot.slice = unsafe { dev.alloc::<u8>(bytes.max(1))? };
        slot.cap = bytes.max(1);
    }
    Ok(slot)
}

#[cutile::module]
mod mmai_probe_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn mmai_i8_16x16_probe(out_ptr: *mut i32) {
        let lhs: Tile<i8, { [16, 32] }> = constant(1i8, const_shape![16, 32]);
        let rhs: Tile<i8, { [32, 16] }> = constant(1i8, const_shape![32, 16]);
        let acc: Tile<i32, { [16, 16] }> = constant(0i32, const_shape![16, 16]);
        let result: Tile<i32, { [16, 16] }> =
            mmai(lhs, rhs, acc, signedness::Unsigned, signedness::Signed);

        let out_base: PointerTile<*mut i32, { [] }> = pointer_to_tile(out_ptr);
        let out_1: PointerTile<*mut i32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_256: PointerTile<*mut i32, { [256] }> = out_1.broadcast(const_shape![256]);
        let offsets: Tile<i32, { [256] }> = iota(const_shape![256]);
        let out_dst: PointerTile<*mut i32, { [256] }> = out_256.offset_tile(offsets);
        store_ptr_tko(
            out_dst,
            result.reshape(const_shape![256]),
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<0>,
        );
    }
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
mod q4k_q8_1_matvec_rows4_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn q4k_q8_1_matvec_b1_rows4_f32(
        q4_ptr: *mut u8,
        q8_ptr: *mut u8,
        out_ptr: *mut f32,
        ncols: i32,
        nrows: i32,
    ) {
        let pid = get_tile_block_id();
        let row_base: i32 = pid.0 * 4i32;
        if row_base + 3i32 >= nrows {
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

        let row_lanes4: Tile<i32, { [4] }> = iota(const_shape![4]);
        let col_lanes32: Tile<i32, { [32] }> = iota(const_shape![32]);
        let col_lanes432: Tile<i32, { [4, 32] }> = col_lanes32
            .reshape(const_shape![1, 32])
            .broadcast(const_shape![4, 32]);
        let c15_128: Tile<i32, { [128] }> = constant(15i32, const_shape![128]);
        let c255_128: Tile<i32, { [128] }> = constant(255i32, const_shape![128]);
        let c4_4: Tile<i32, { [4] }> = constant(4i32, const_shape![4]);
        let c6_4: Tile<i32, { [4] }> = constant(6i32, const_shape![4]);
        let c15_4: Tile<i32, { [4] }> = constant(15i32, const_shape![4]);
        let c63_4: Tile<i32, { [4] }> = constant(63i32, const_shape![4]);
        let c255_4: Tile<i32, { [4] }> = constant(255i32, const_shape![4]);

        let mut acc4: Tile<f32, { [4] }> = constant(0.0f32, const_shape![4]);
        let blocks_per_row: i32 = ncols / 256i32;
        let rows4: Tile<i32, { [4] }> = row_lanes4 + broadcast_scalar(row_base, const_shape![4]);
        let q4_row_bases4: Tile<i32, { [4] }> =
            rows4 * broadcast_scalar(blocks_per_row * 144i32, const_shape![4]);

        for block in 0i32..blocks_per_row {
            let q4_block_bases4: Tile<i32, { [4] }> =
                q4_row_bases4 + broadcast_scalar(block * 144i32, const_shape![4]);
            let q4_block_bases432: Tile<i32, { [4, 32] }> = q4_block_bases4
                .reshape(const_shape![4, 1])
                .broadcast(const_shape![4, 32]);

            let d_offsets4: Tile<i32, { [4] }> = q4_block_bases4 / constant(2i32, const_shape![4]);
            let q4_half_4: PointerTile<*mut f16, { [4] }> = q4_half_1.broadcast(const_shape![4]);
            let d_ptrs: PointerTile<*mut f16, { [4] }> = q4_half_4.offset_tile(d_offsets4);
            let (d_h4, _d_tok): (Tile<f16, { [4] }>, Token) = load_ptr_tko(
                d_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let dmin_offsets4: Tile<i32, { [4] }> = d_offsets4 + constant(1i32, const_shape![4]);
            let dmin_ptrs: PointerTile<*mut f16, { [4] }> = q4_half_4.offset_tile(dmin_offsets4);
            let (dmin_h4, _dmin_tok): (Tile<f16, { [4] }>, Token) = load_ptr_tko(
                dmin_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let d4: Tile<f32, { [4] }> = ftof(d_h4, rounding::NearestEven);
            let dmin4: Tile<f32, { [4] }> = ftof(dmin_h4, rounding::NearestEven);
            let d432: Tile<f32, { [4, 32] }> = d4
                .reshape(const_shape![4, 1])
                .broadcast(const_shape![4, 32]);
            let dmin432: Tile<f32, { [4, 32] }> = dmin4
                .reshape(const_shape![4, 1])
                .broadcast(const_shape![4, 32]);

            for group in 0i32..8i32 {
                let q4_offsets128: Tile<i32, { [128] }> = (q4_block_bases432
                    + col_lanes432
                    + broadcast_scalar(16i32 + (group / 2i32) * 32i32, const_shape![4, 32]))
                .reshape(const_shape![128]);
                let q4_i8_128: PointerTile<*mut i8, { [128] }> =
                    q4_i8_1.broadcast(const_shape![128]);
                let q4_ptrs: PointerTile<*mut i8, { [128] }> = q4_i8_128.offset_tile(q4_offsets128);
                let (q4_bytes, _q4_tok): (Tile<i8, { [128] }>, Token) = load_ptr_tko(
                    q4_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let q4_byte_i32: Tile<i32, { [128] }> = exti(q4_bytes) & c255_128;
                let shift128: Tile<i32, { [128] }> =
                    broadcast_scalar((group % 2i32) * 4i32, const_shape![128]);
                let q4_i32: Tile<i32, { [128] }> = shri(q4_byte_i32, shift128) & c15_128;
                let q4_f32_128: Tile<f32, { [128] }> = convert_tile(q4_i32);
                let q4_f32: Tile<f32, { [4, 32] }> = q4_f32_128.reshape(const_shape![4, 32]);

                #[allow(unused_assignments)]
                let mut scale_i32: Tile<i32, { [4] }> = constant(0i32, const_shape![4]);
                #[allow(unused_assignments)]
                let mut min_i32: Tile<i32, { [4] }> = constant(0i32, const_shape![4]);
                let q4_i8_4: PointerTile<*mut i8, { [4] }> = q4_i8_1.broadcast(const_shape![4]);
                if group < 4i32 {
                    let scale_offsets4: Tile<i32, { [4] }> =
                        q4_block_bases4 + broadcast_scalar(4i32 + group, const_shape![4]);
                    let scale_ptrs: PointerTile<*mut i8, { [4] }> =
                        q4_i8_4.offset_tile(scale_offsets4);
                    let (scale_b4, _scale_tok): (Tile<i8, { [4] }>, Token) = load_ptr_tko(
                        scale_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let min_offsets4: Tile<i32, { [4] }> =
                        q4_block_bases4 + broadcast_scalar(8i32 + group, const_shape![4]);
                    let min_ptrs: PointerTile<*mut i8, { [4] }> = q4_i8_4.offset_tile(min_offsets4);
                    let (min_b4, _min_tok): (Tile<i8, { [4] }>, Token) = load_ptr_tko(
                        min_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    scale_i32 = exti(scale_b4) & c63_4;
                    min_i32 = exti(min_b4) & c63_4;
                } else {
                    let packed_offsets4: Tile<i32, { [4] }> =
                        q4_block_bases4 + broadcast_scalar(8i32 + group, const_shape![4]);
                    let packed_ptrs: PointerTile<*mut i8, { [4] }> =
                        q4_i8_4.offset_tile(packed_offsets4);
                    let (packed_b4, _packed_tok): (Tile<i8, { [4] }>, Token) = load_ptr_tko(
                        packed_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let scale_hi_offsets4: Tile<i32, { [4] }> =
                        q4_block_bases4 + broadcast_scalar(group, const_shape![4]);
                    let scale_hi_ptrs: PointerTile<*mut i8, { [4] }> =
                        q4_i8_4.offset_tile(scale_hi_offsets4);
                    let (scale_hi_b4, _scale_hi_tok): (Tile<i8, { [4] }>, Token) = load_ptr_tko(
                        scale_hi_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let min_hi_offsets4: Tile<i32, { [4] }> =
                        q4_block_bases4 + broadcast_scalar(4i32 + group, const_shape![4]);
                    let min_hi_ptrs: PointerTile<*mut i8, { [4] }> =
                        q4_i8_4.offset_tile(min_hi_offsets4);
                    let (min_hi_b4, _min_hi_tok): (Tile<i8, { [4] }>, Token) = load_ptr_tko(
                        min_hi_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let packed: Tile<i32, { [4] }> = exti(packed_b4) & c255_4;
                    let scale_hi: Tile<i32, { [4] }> = exti(scale_hi_b4) & c255_4;
                    let min_hi: Tile<i32, { [4] }> = exti(min_hi_b4) & c255_4;
                    scale_i32 =
                        (packed & c15_4) | shli(shri(scale_hi, c6_4), c4_4, overflow::NoWrap);
                    min_i32 = shri(packed, c4_4) | shli(shri(min_hi, c6_4), c4_4, overflow::NoWrap);
                }

                let scale_f32: Tile<f32, { [4] }> = convert_tile(scale_i32);
                let min_f32: Tile<f32, { [4] }> = convert_tile(min_i32);
                let scale432: Tile<f32, { [4, 32] }> = scale_f32
                    .reshape(const_shape![4, 1])
                    .broadcast(const_shape![4, 32]);
                let min432: Tile<f32, { [4, 32] }> = min_f32
                    .reshape(const_shape![4, 1])
                    .broadcast(const_shape![4, 32]);

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
                let d8_32: Tile<f32, { [32] }> = d8_1.broadcast(const_shape![32]);

                let q8_offsets32: Tile<i32, { [32] }> =
                    col_lanes32 + broadcast_scalar(q8_block_base + 4i32, const_shape![32]);
                let q8_i8_32: PointerTile<*mut i8, { [32] }> = q8_i8_1.broadcast(const_shape![32]);
                let q8_ptrs: PointerTile<*mut i8, { [32] }> = q8_i8_32.offset_tile(q8_offsets32);
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
                let q8_val32: Tile<f32, { [32] }> = d8_32 * q8_f32;
                let q8_val432: Tile<f32, { [4, 32] }> = q8_val32
                    .reshape(const_shape![1, 32])
                    .broadcast(const_shape![4, 32]);

                let q4_val: Tile<f32, { [4, 32] }> = d432 * scale432 * q4_f32 - dmin432 * min432;
                let prod: Tile<f32, { [4, 32] }> = q4_val * q8_val432;
                let partial4: Tile<f32, { [4] }> = reduce_sum(prod, 1i32);
                acc4 = acc4 + partial4;
            }
        }

        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_1: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_4: PointerTile<*mut f32, { [4] }> = out_1.broadcast(const_shape![4]);
        let out_offsets: Tile<i32, { [4] }> =
            row_lanes4 + broadcast_scalar(row_base, const_shape![4]);
        let out_dst: PointerTile<*mut f32, { [4] }> = out_4.offset_tile(out_offsets);
        store_ptr_tko(
            out_dst,
            acc4,
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
mod q4k_q8_1_mmq_matmul_tiled_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn q4k_q8_1_mmq_matmul_tiled_f32(
        q4_ptr: *mut u8,
        q8_ptr: *mut u8,
        out_ptr: *mut f32,
        ncols: i32,
        nrows: i32,
        b_size: i32,
        batch_start: i32,
    ) {
        let pid = get_tile_block_id();
        let row_tile: i32 = pid.0;
        let batch_tile: i32 = pid.1;

        // Flat tile layout: 16 outputs = 4 weight rows × 4 RHS rows.
        // Each K-group expands to 16 × 32 lane products, then reduces the
        // K-lane axis back to one contribution per output.
        let outs16: Tile<i32, { [16] }> = iota(const_shape![16]);
        let row_local16: Tile<i32, { [16] }> = outs16 / constant(4i32, const_shape![16]);
        let batch_local16: Tile<i32, { [16] }> =
            outs16 - row_local16 * constant(4i32, const_shape![16]);
        let rows16: Tile<i32, { [16] }> =
            row_local16 + broadcast_scalar(row_tile * 4i32, const_shape![16]);
        let batches16: Tile<i32, { [16] }> =
            batch_local16 + broadcast_scalar(batch_start + batch_tile * 4i32, const_shape![16]);

        let lanes512: Tile<i32, { [512] }> = iota(const_shape![512]);
        let out_index512: Tile<i32, { [512] }> = lanes512 / constant(32i32, const_shape![512]);
        let lane512: Tile<i32, { [512] }> =
            lanes512 - out_index512 * constant(32i32, const_shape![512]);
        let row_local512: Tile<i32, { [512] }> = out_index512 / constant(4i32, const_shape![512]);
        let batch_local512: Tile<i32, { [512] }> =
            out_index512 - row_local512 * constant(4i32, const_shape![512]);
        let rows512: Tile<i32, { [512] }> =
            row_local512 + broadcast_scalar(row_tile * 4i32, const_shape![512]);
        let batches512: Tile<i32, { [512] }> =
            batch_local512 + broadcast_scalar(batch_start + batch_tile * 4i32, const_shape![512]);

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

        let mut acc: Tile<f32, { [16] }> = constant(0.0f32, const_shape![16]);
        let blocks_per_row: i32 = ncols / 256i32;
        let q4_row_stride_bytes: i32 = blocks_per_row * 144i32;
        let q4_row_bases16: Tile<i32, { [16] }> =
            rows16 * broadcast_scalar(q4_row_stride_bytes, const_shape![16]);
        let q4_row_bases512: Tile<i32, { [512] }> =
            rows512 * broadcast_scalar(q4_row_stride_bytes, const_shape![512]);

        for block in 0i32..blocks_per_row {
            let q4_block_bases16: Tile<i32, { [16] }> =
                q4_row_bases16 + broadcast_scalar(block * 144i32, const_shape![16]);
            let q4_block_bases512: Tile<i32, { [512] }> =
                q4_row_bases512 + broadcast_scalar(block * 144i32, const_shape![512]);

            let d_offsets: Tile<i32, { [16] }> =
                q4_block_bases16 / constant(2i32, const_shape![16]);
            let q4_half_16: PointerTile<*mut f16, { [16] }> = q4_half_1.broadcast(const_shape![16]);
            let d_ptrs: PointerTile<*mut f16, { [16] }> = q4_half_16.offset_tile(d_offsets);
            let (d_h, _d_tok): (Tile<f16, { [16] }>, Token) = load_ptr_tko(
                d_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let dmin_ptrs: PointerTile<*mut f16, { [16] }> =
                q4_half_16.offset_tile(d_offsets + constant(1i32, const_shape![16]));
            let (dmin_h, _dmin_tok): (Tile<f16, { [16] }>, Token) = load_ptr_tko(
                dmin_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let d16: Tile<f32, { [16] }> = ftof(d_h, rounding::NearestEven);
            let dmin16: Tile<f32, { [16] }> = ftof(dmin_h, rounding::NearestEven);

            for group in 0i32..8i32 {
                let q4_offsets: Tile<i32, { [512] }> = q4_block_bases512
                    + broadcast_scalar(16i32 + (group / 2i32) * 32i32, const_shape![512])
                    + lane512;
                let q4_i8_512: PointerTile<*mut i8, { [512] }> =
                    q4_i8_1.broadcast(const_shape![512]);
                let q4_ptrs: PointerTile<*mut i8, { [512] }> = q4_i8_512.offset_tile(q4_offsets);
                let (q4_bytes, _q4_tok): (Tile<i8, { [512] }>, Token) = load_ptr_tko(
                    q4_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let q4_byte_i32: Tile<i32, { [512] }> =
                    exti(q4_bytes) & constant(255i32, const_shape![512]);
                let q4_i32: Tile<i32, { [512] }> = shri(
                    q4_byte_i32,
                    broadcast_scalar((group % 2i32) * 4i32, const_shape![512]),
                ) & constant(15i32, const_shape![512]);
                let q4_f32: Tile<f32, { [512] }> = convert_tile(q4_i32);

                #[allow(unused_assignments)]
                let mut scale_i32: Tile<i32, { [16] }> = constant(0i32, const_shape![16]);
                #[allow(unused_assignments)]
                let mut min_i32: Tile<i32, { [16] }> = constant(0i32, const_shape![16]);
                let q4_i8_16: PointerTile<*mut i8, { [16] }> = q4_i8_1.broadcast(const_shape![16]);
                if group < 4i32 {
                    let scale_offsets: Tile<i32, { [16] }> =
                        q4_block_bases16 + broadcast_scalar(4i32 + group, const_shape![16]);
                    let scale_ptrs: PointerTile<*mut i8, { [16] }> =
                        q4_i8_16.offset_tile(scale_offsets);
                    let (scale_b, _scale_tok): (Tile<i8, { [16] }>, Token) = load_ptr_tko(
                        scale_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let min_offsets: Tile<i32, { [16] }> =
                        q4_block_bases16 + broadcast_scalar(8i32 + group, const_shape![16]);
                    let min_ptrs: PointerTile<*mut i8, { [16] }> =
                        q4_i8_16.offset_tile(min_offsets);
                    let (min_b, _min_tok): (Tile<i8, { [16] }>, Token) = load_ptr_tko(
                        min_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    scale_i32 = exti(scale_b) & constant(63i32, const_shape![16]);
                    min_i32 = exti(min_b) & constant(63i32, const_shape![16]);
                } else {
                    let packed_offsets: Tile<i32, { [16] }> =
                        q4_block_bases16 + broadcast_scalar(8i32 + group, const_shape![16]);
                    let packed_ptrs: PointerTile<*mut i8, { [16] }> =
                        q4_i8_16.offset_tile(packed_offsets);
                    let (packed_b, _packed_tok): (Tile<i8, { [16] }>, Token) = load_ptr_tko(
                        packed_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let scale_hi_offsets: Tile<i32, { [16] }> =
                        q4_block_bases16 + broadcast_scalar(group, const_shape![16]);
                    let scale_hi_ptrs: PointerTile<*mut i8, { [16] }> =
                        q4_i8_16.offset_tile(scale_hi_offsets);
                    let (scale_hi_b, _scale_hi_tok): (Tile<i8, { [16] }>, Token) = load_ptr_tko(
                        scale_hi_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let min_hi_offsets: Tile<i32, { [16] }> =
                        q4_block_bases16 + broadcast_scalar(4i32 + group, const_shape![16]);
                    let min_hi_ptrs: PointerTile<*mut i8, { [16] }> =
                        q4_i8_16.offset_tile(min_hi_offsets);
                    let (min_hi_b, _min_hi_tok): (Tile<i8, { [16] }>, Token) = load_ptr_tko(
                        min_hi_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let c255: Tile<i32, { [16] }> = constant(255i32, const_shape![16]);
                    let packed: Tile<i32, { [16] }> = exti(packed_b) & c255;
                    let scale_hi: Tile<i32, { [16] }> = exti(scale_hi_b) & c255;
                    let min_hi: Tile<i32, { [16] }> = exti(min_hi_b) & c255;
                    let c4: Tile<i32, { [16] }> = constant(4i32, const_shape![16]);
                    let c6: Tile<i32, { [16] }> = constant(6i32, const_shape![16]);
                    let c15: Tile<i32, { [16] }> = constant(15i32, const_shape![16]);
                    scale_i32 = (packed & c15) | shli(shri(scale_hi, c6), c4, overflow::NoWrap);
                    min_i32 = shri(packed, c4) | shli(shri(min_hi, c6), c4, overflow::NoWrap);
                }

                let q8_block_index: i32 = block * 8i32 + group;
                let q8_mmq_block_index: i32 = q8_block_index / 4i32;
                let q8_mmq_group: i32 = q8_block_index % 4i32;
                let q8_block_bases16: Tile<i32, { [16] }> = (batches16
                    + broadcast_scalar(q8_mmq_block_index * b_size, const_shape![16]))
                    * broadcast_scalar(144i32, const_shape![16]);
                let q8_block_bases512: Tile<i32, { [512] }> = (batches512
                    + broadcast_scalar(q8_mmq_block_index * b_size, const_shape![512]))
                    * broadcast_scalar(144i32, const_shape![512]);

                let d8_byte_offsets16: Tile<i32, { [16] }> =
                    q8_block_bases16 + broadcast_scalar(q8_mmq_group * 4i32, const_shape![16]);
                let d8_offsets16: Tile<i32, { [16] }> =
                    d8_byte_offsets16 / constant(2i32, const_shape![16]);
                let q8_half_16: PointerTile<*mut f16, { [16] }> =
                    q8_half_1.broadcast(const_shape![16]);
                let d8_ptrs: PointerTile<*mut f16, { [16] }> = q8_half_16.offset_tile(d8_offsets16);
                let (d8_h, _d8_tok): (Tile<f16, { [16] }>, Token) = load_ptr_tko(
                    d8_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None,
                    Latency::<0>,
                );
                let sum8_ptrs: PointerTile<*mut f16, { [16] }> =
                    q8_half_16.offset_tile(d8_offsets16 + constant(1i32, const_shape![16]));
                let (sum8_h, _sum8_tok): (Tile<f16, { [16] }>, Token) = load_ptr_tko(
                    sum8_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None,
                    Latency::<0>,
                );
                let d8_16: Tile<f32, { [16] }> = ftof(d8_h, rounding::NearestEven);
                let sum8_16: Tile<f32, { [16] }> = ftof(sum8_h, rounding::NearestEven);

                let q8_offsets: Tile<i32, { [512] }> = q8_block_bases512
                    + broadcast_scalar(16i32 + q8_mmq_group * 32i32, const_shape![512])
                    + lane512;
                let q8_i8_512: PointerTile<*mut i8, { [512] }> =
                    q8_i8_1.broadcast(const_shape![512]);
                let q8_ptrs: PointerTile<*mut i8, { [512] }> = q8_i8_512.offset_tile(q8_offsets);
                let (q8_bytes, _q8_tok): (Tile<i8, { [512] }>, Token) = load_ptr_tko(
                    q8_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let q8_i32: Tile<i32, { [512] }> = exti(q8_bytes);
                let q8_f32: Tile<f32, { [512] }> = convert_tile(q8_i32);

                let prod: Tile<f32, { [512] }> = q4_f32 * q8_f32;
                let dot: Tile<f32, { [16] }> = reduce_sum(prod.reshape(const_shape![16, 32]), 1i32);
                let scale16: Tile<f32, { [16] }> = convert_tile(scale_i32);
                let min16: Tile<f32, { [16] }> = convert_tile(min_i32);
                acc = acc + d16 * scale16 * d8_16 * dot - dmin16 * min16 * sum8_16;
            }
        }

        let out_lanes16: Tile<i32, { [16] }> = iota(const_shape![16]);
        let out_batch_local16: Tile<i32, { [16] }> = out_lanes16 / constant(4i32, const_shape![16]);
        let out_row_local16: Tile<i32, { [16] }> =
            out_lanes16 - out_batch_local16 * constant(4i32, const_shape![16]);
        let out_rows16: Tile<i32, { [16] }> =
            out_row_local16 + broadcast_scalar(row_tile * 4i32, const_shape![16]);
        let out_batches16: Tile<i32, { [16] }> =
            out_batch_local16 + broadcast_scalar(batch_start + batch_tile * 4i32, const_shape![16]);
        let out_offsets: Tile<i32, { [16] }> =
            out_rows16 + out_batches16 * broadcast_scalar(nrows, const_shape![16]);
        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_1: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_16: PointerTile<*mut f32, { [16] }> = out_1.broadcast(const_shape![16]);
        let out_dst: PointerTile<*mut f32, { [16] }> = out_16.offset_tile(out_offsets);
        let acc_4x4: Tile<f32, { [4, 4] }> = acc.reshape(const_shape![4, 4]);
        let acc_batch_major: Tile<f32, { [16] }> = acc_4x4.transpose().reshape(const_shape![16]);
        store_ptr_tko(
            out_dst,
            acc_batch_major,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<0>,
        );
    }
}

#[cutile::module]
mod q4k_q8_1_mmq_matmul_mmai_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn q4k_q8_1_mmq_matmul_mmai_16x16_f32(
        q4_ptr: *mut u8,
        q8_ptr: *mut u8,
        out_ptr: *mut f32,
        ncols: i32,
        nrows: i32,
        b_size: i32,
        batch_start: i32,
    ) {
        let pid = get_tile_block_id();
        let row_tile: i32 = pid.0;
        let batch_tile: i32 = pid.1;
        let row_base: i32 = row_tile * 16i32;
        let batch_base: i32 = batch_start + batch_tile * 16i32;
        if row_base >= nrows || batch_base >= b_size {
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

        let lanes16: Tile<i32, { [16] }> = iota(const_shape![16]);
        let rows16: Tile<i32, { [16] }> = lanes16 + broadcast_scalar(row_base, const_shape![16]);
        let batches16: Tile<i32, { [16] }> =
            lanes16 + broadcast_scalar(batch_base, const_shape![16]);

        let q4_lanes512: Tile<i32, { [512] }> = iota(const_shape![512]);
        let q4_row_local512: Tile<i32, { [512] }> =
            q4_lanes512 / constant(32i32, const_shape![512]);
        let q4_k_lane512: Tile<i32, { [512] }> =
            q4_lanes512 - q4_row_local512 * constant(32i32, const_shape![512]);
        let q4_rows512: Tile<i32, { [512] }> =
            q4_row_local512 + broadcast_scalar(row_base, const_shape![512]);

        let q8_lanes512: Tile<i32, { [512] }> = iota(const_shape![512]);
        let q8_k_lane512: Tile<i32, { [512] }> = q8_lanes512 / constant(16i32, const_shape![512]);
        let q8_batch_local512: Tile<i32, { [512] }> =
            q8_lanes512 - q8_k_lane512 * constant(16i32, const_shape![512]);
        let q8_batches512: Tile<i32, { [512] }> =
            q8_batch_local512 + broadcast_scalar(batch_base, const_shape![512]);

        let mut acc: Tile<f32, { [16, 16] }> = constant(0.0f32, const_shape![16, 16]);
        let blocks_per_row: i32 = ncols / 256i32;
        let q4_row_stride_bytes: i32 = blocks_per_row * 144i32;
        let q4_row_bases16: Tile<i32, { [16] }> =
            rows16 * broadcast_scalar(q4_row_stride_bytes, const_shape![16]);
        let q4_row_bases512: Tile<i32, { [512] }> =
            q4_rows512 * broadcast_scalar(q4_row_stride_bytes, const_shape![512]);

        let q4_i8_16: PointerTile<*mut i8, { [16] }> = q4_i8_1.broadcast(const_shape![16]);
        let q4_i8_512: PointerTile<*mut i8, { [512] }> = q4_i8_1.broadcast(const_shape![512]);
        let q8_i8_512: PointerTile<*mut i8, { [512] }> = q8_i8_1.broadcast(const_shape![512]);
        let q4_half_16: PointerTile<*mut f16, { [16] }> = q4_half_1.broadcast(const_shape![16]);
        let q8_half_16: PointerTile<*mut f16, { [16] }> = q8_half_1.broadcast(const_shape![16]);

        for block in 0i32..blocks_per_row {
            let q4_block_bases16: Tile<i32, { [16] }> =
                q4_row_bases16 + broadcast_scalar(block * 144i32, const_shape![16]);
            let q4_block_bases512: Tile<i32, { [512] }> =
                q4_row_bases512 + broadcast_scalar(block * 144i32, const_shape![512]);

            let d_offsets16: Tile<i32, { [16] }> =
                q4_block_bases16 / constant(2i32, const_shape![16]);
            let d_ptrs: PointerTile<*mut f16, { [16] }> = q4_half_16.offset_tile(d_offsets16);
            let (d_h, _d_tok): (Tile<f16, { [16] }>, Token) = load_ptr_tko(
                d_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let dmin_ptrs: PointerTile<*mut f16, { [16] }> =
                q4_half_16.offset_tile(d_offsets16 + constant(1i32, const_shape![16]));
            let (dmin_h, _dmin_tok): (Tile<f16, { [16] }>, Token) = load_ptr_tko(
                dmin_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let d16: Tile<f32, { [16] }> = ftof(d_h, rounding::NearestEven);
            let dmin16: Tile<f32, { [16] }> = ftof(dmin_h, rounding::NearestEven);
            let d1616: Tile<f32, { [16, 16] }> = d16
                .reshape(const_shape![16, 1])
                .broadcast(const_shape![16, 16]);
            let dmin1616: Tile<f32, { [16, 16] }> = dmin16
                .reshape(const_shape![16, 1])
                .broadcast(const_shape![16, 16]);

            for pair in 0i32..4i32 {
                let q4_offsets: Tile<i32, { [512] }> = q4_block_bases512
                    + broadcast_scalar(16i32 + pair * 32i32, const_shape![512])
                    + q4_k_lane512;
                let q4_ptrs: PointerTile<*mut i8, { [512] }> = q4_i8_512.offset_tile(q4_offsets);
                let (q4_bytes, _q4_tok): (Tile<i8, { [512] }>, Token) = load_ptr_tko(
                    q4_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let c4_512: Tile<i32, { [512] }> = constant(4i32, const_shape![512]);
                let c15_512: Tile<i32, { [512] }> = constant(15i32, const_shape![512]);
                let c255_512: Tile<i32, { [512] }> = constant(255i32, const_shape![512]);
                let q4_byte_i32: Tile<i32, { [512] }> = exti(q4_bytes) & c255_512;
                let q4_low_i32: Tile<i32, { [512] }> = q4_byte_i32 & c15_512;
                let q4_high_i32: Tile<i32, { [512] }> = shri(q4_byte_i32, c4_512) & c15_512;
                let q4_low_i8_flat: Tile<i8, { [512] }> = trunci(q4_low_i32, overflow::NoWrap);
                let q4_high_i8_flat: Tile<i8, { [512] }> = trunci(q4_high_i32, overflow::NoWrap);
                let q4_low_i8: Tile<i8, { [16, 32] }> =
                    q4_low_i8_flat.reshape(const_shape![16, 32]);
                let q4_high_i8: Tile<i8, { [16, 32] }> =
                    q4_high_i8_flat.reshape(const_shape![16, 32]);

                let group0: i32 = pair * 2i32;
                let group1: i32 = group0 + 1i32;

                #[allow(unused_assignments)]
                let mut scale0_i32: Tile<i32, { [16] }> = constant(0i32, const_shape![16]);
                #[allow(unused_assignments)]
                let mut min0_i32: Tile<i32, { [16] }> = constant(0i32, const_shape![16]);
                if group0 < 4i32 {
                    let scale_offsets: Tile<i32, { [16] }> =
                        q4_block_bases16 + broadcast_scalar(4i32 + group0, const_shape![16]);
                    let scale_ptrs: PointerTile<*mut i8, { [16] }> =
                        q4_i8_16.offset_tile(scale_offsets);
                    let (scale_b, _scale_tok): (Tile<i8, { [16] }>, Token) = load_ptr_tko(
                        scale_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let min_offsets: Tile<i32, { [16] }> =
                        q4_block_bases16 + broadcast_scalar(8i32 + group0, const_shape![16]);
                    let min_ptrs: PointerTile<*mut i8, { [16] }> =
                        q4_i8_16.offset_tile(min_offsets);
                    let (min_b, _min_tok): (Tile<i8, { [16] }>, Token) = load_ptr_tko(
                        min_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    scale0_i32 = exti(scale_b) & constant(63i32, const_shape![16]);
                    min0_i32 = exti(min_b) & constant(63i32, const_shape![16]);
                } else {
                    let packed_offsets: Tile<i32, { [16] }> =
                        q4_block_bases16 + broadcast_scalar(8i32 + group0, const_shape![16]);
                    let packed_ptrs: PointerTile<*mut i8, { [16] }> =
                        q4_i8_16.offset_tile(packed_offsets);
                    let (packed_b, _packed_tok): (Tile<i8, { [16] }>, Token) = load_ptr_tko(
                        packed_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let scale_hi_offsets: Tile<i32, { [16] }> =
                        q4_block_bases16 + broadcast_scalar(group0, const_shape![16]);
                    let scale_hi_ptrs: PointerTile<*mut i8, { [16] }> =
                        q4_i8_16.offset_tile(scale_hi_offsets);
                    let (scale_hi_b, _scale_hi_tok): (Tile<i8, { [16] }>, Token) = load_ptr_tko(
                        scale_hi_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let min_hi_offsets: Tile<i32, { [16] }> =
                        q4_block_bases16 + broadcast_scalar(4i32 + group0, const_shape![16]);
                    let min_hi_ptrs: PointerTile<*mut i8, { [16] }> =
                        q4_i8_16.offset_tile(min_hi_offsets);
                    let (min_hi_b, _min_hi_tok): (Tile<i8, { [16] }>, Token) = load_ptr_tko(
                        min_hi_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let c255: Tile<i32, { [16] }> = constant(255i32, const_shape![16]);
                    let packed: Tile<i32, { [16] }> = exti(packed_b) & c255;
                    let scale_hi: Tile<i32, { [16] }> = exti(scale_hi_b) & c255;
                    let min_hi: Tile<i32, { [16] }> = exti(min_hi_b) & c255;
                    let c4: Tile<i32, { [16] }> = constant(4i32, const_shape![16]);
                    let c6: Tile<i32, { [16] }> = constant(6i32, const_shape![16]);
                    let c15: Tile<i32, { [16] }> = constant(15i32, const_shape![16]);
                    scale0_i32 = (packed & c15) | shli(shri(scale_hi, c6), c4, overflow::NoWrap);
                    min0_i32 = shri(packed, c4) | shli(shri(min_hi, c6), c4, overflow::NoWrap);
                }

                let q8_block_index0: i32 = block * 8i32 + group0;
                let q8_mmq_block_index0: i32 = q8_block_index0 / 4i32;
                let q8_mmq_group0: i32 = q8_block_index0 % 4i32;
                let q8_block_bases16_0: Tile<i32, { [16] }> = (batches16
                    + broadcast_scalar(q8_mmq_block_index0 * b_size, const_shape![16]))
                    * broadcast_scalar(144i32, const_shape![16]);
                let q8_block_bases512_0: Tile<i32, { [512] }> = (q8_batches512
                    + broadcast_scalar(q8_mmq_block_index0 * b_size, const_shape![512]))
                    * broadcast_scalar(144i32, const_shape![512]);
                let d8_byte_offsets16_0: Tile<i32, { [16] }> =
                    q8_block_bases16_0 + broadcast_scalar(q8_mmq_group0 * 4i32, const_shape![16]);
                let d8_offsets16_0: Tile<i32, { [16] }> =
                    d8_byte_offsets16_0 / constant(2i32, const_shape![16]);
                let d8_ptrs0: PointerTile<*mut f16, { [16] }> =
                    q8_half_16.offset_tile(d8_offsets16_0);
                let (d8_h0, _d8_tok0): (Tile<f16, { [16] }>, Token) = load_ptr_tko(
                    d8_ptrs0,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None,
                    Latency::<0>,
                );
                let sum8_ptrs0: PointerTile<*mut f16, { [16] }> =
                    q8_half_16.offset_tile(d8_offsets16_0 + constant(1i32, const_shape![16]));
                let (sum8_h0, _sum8_tok0): (Tile<f16, { [16] }>, Token) = load_ptr_tko(
                    sum8_ptrs0,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None,
                    Latency::<0>,
                );
                let d8_16_0: Tile<f32, { [16] }> = ftof(d8_h0, rounding::NearestEven);
                let sum8_16_0: Tile<f32, { [16] }> = ftof(sum8_h0, rounding::NearestEven);
                let q8_offsets0: Tile<i32, { [512] }> = q8_block_bases512_0
                    + broadcast_scalar(16i32 + q8_mmq_group0 * 32i32, const_shape![512])
                    + q8_k_lane512;
                let q8_ptrs0: PointerTile<*mut i8, { [512] }> = q8_i8_512.offset_tile(q8_offsets0);
                let (q8_bytes0, _q8_tok0): (Tile<i8, { [512] }>, Token) = load_ptr_tko(
                    q8_ptrs0,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let q8_i8_0: Tile<i8, { [32, 16] }> = q8_bytes0.reshape(const_shape![32, 16]);
                let zero_acc0: Tile<i32, { [16, 16] }> = constant(0i32, const_shape![16, 16]);
                let dot0_i32: Tile<i32, { [16, 16] }> = mmai(
                    q4_low_i8,
                    q8_i8_0,
                    zero_acc0,
                    signedness::Unsigned,
                    signedness::Signed,
                );
                let dot0_f32: Tile<f32, { [16, 16] }> = convert_tile(dot0_i32);
                let scale0_f32: Tile<f32, { [16] }> = convert_tile(scale0_i32);
                let min0_f32: Tile<f32, { [16] }> = convert_tile(min0_i32);
                let scale0_1616: Tile<f32, { [16, 16] }> = scale0_f32
                    .reshape(const_shape![16, 1])
                    .broadcast(const_shape![16, 16]);
                let min0_1616: Tile<f32, { [16, 16] }> = min0_f32
                    .reshape(const_shape![16, 1])
                    .broadcast(const_shape![16, 16]);
                let d8_1616_0: Tile<f32, { [16, 16] }> = d8_16_0
                    .reshape(const_shape![1, 16])
                    .broadcast(const_shape![16, 16]);
                let sum8_1616_0: Tile<f32, { [16, 16] }> = sum8_16_0
                    .reshape(const_shape![1, 16])
                    .broadcast(const_shape![16, 16]);
                acc = acc + d1616 * scale0_1616 * d8_1616_0 * dot0_f32
                    - dmin1616 * min0_1616 * sum8_1616_0;

                #[allow(unused_assignments)]
                let mut scale1_i32: Tile<i32, { [16] }> = constant(0i32, const_shape![16]);
                #[allow(unused_assignments)]
                let mut min1_i32: Tile<i32, { [16] }> = constant(0i32, const_shape![16]);
                if group1 < 4i32 {
                    let scale_offsets: Tile<i32, { [16] }> =
                        q4_block_bases16 + broadcast_scalar(4i32 + group1, const_shape![16]);
                    let scale_ptrs: PointerTile<*mut i8, { [16] }> =
                        q4_i8_16.offset_tile(scale_offsets);
                    let (scale_b, _scale_tok): (Tile<i8, { [16] }>, Token) = load_ptr_tko(
                        scale_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let min_offsets: Tile<i32, { [16] }> =
                        q4_block_bases16 + broadcast_scalar(8i32 + group1, const_shape![16]);
                    let min_ptrs: PointerTile<*mut i8, { [16] }> =
                        q4_i8_16.offset_tile(min_offsets);
                    let (min_b, _min_tok): (Tile<i8, { [16] }>, Token) = load_ptr_tko(
                        min_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    scale1_i32 = exti(scale_b) & constant(63i32, const_shape![16]);
                    min1_i32 = exti(min_b) & constant(63i32, const_shape![16]);
                } else {
                    let packed_offsets: Tile<i32, { [16] }> =
                        q4_block_bases16 + broadcast_scalar(8i32 + group1, const_shape![16]);
                    let packed_ptrs: PointerTile<*mut i8, { [16] }> =
                        q4_i8_16.offset_tile(packed_offsets);
                    let (packed_b, _packed_tok): (Tile<i8, { [16] }>, Token) = load_ptr_tko(
                        packed_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let scale_hi_offsets: Tile<i32, { [16] }> =
                        q4_block_bases16 + broadcast_scalar(group1, const_shape![16]);
                    let scale_hi_ptrs: PointerTile<*mut i8, { [16] }> =
                        q4_i8_16.offset_tile(scale_hi_offsets);
                    let (scale_hi_b, _scale_hi_tok): (Tile<i8, { [16] }>, Token) = load_ptr_tko(
                        scale_hi_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let min_hi_offsets: Tile<i32, { [16] }> =
                        q4_block_bases16 + broadcast_scalar(4i32 + group1, const_shape![16]);
                    let min_hi_ptrs: PointerTile<*mut i8, { [16] }> =
                        q4_i8_16.offset_tile(min_hi_offsets);
                    let (min_hi_b, _min_hi_tok): (Tile<i8, { [16] }>, Token) = load_ptr_tko(
                        min_hi_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let c255: Tile<i32, { [16] }> = constant(255i32, const_shape![16]);
                    let packed: Tile<i32, { [16] }> = exti(packed_b) & c255;
                    let scale_hi: Tile<i32, { [16] }> = exti(scale_hi_b) & c255;
                    let min_hi: Tile<i32, { [16] }> = exti(min_hi_b) & c255;
                    let c4: Tile<i32, { [16] }> = constant(4i32, const_shape![16]);
                    let c6: Tile<i32, { [16] }> = constant(6i32, const_shape![16]);
                    let c15: Tile<i32, { [16] }> = constant(15i32, const_shape![16]);
                    scale1_i32 = (packed & c15) | shli(shri(scale_hi, c6), c4, overflow::NoWrap);
                    min1_i32 = shri(packed, c4) | shli(shri(min_hi, c6), c4, overflow::NoWrap);
                }

                let q8_block_index1: i32 = block * 8i32 + group1;
                let q8_mmq_block_index1: i32 = q8_block_index1 / 4i32;
                let q8_mmq_group1: i32 = q8_block_index1 % 4i32;
                let q8_block_bases16_1: Tile<i32, { [16] }> = (batches16
                    + broadcast_scalar(q8_mmq_block_index1 * b_size, const_shape![16]))
                    * broadcast_scalar(144i32, const_shape![16]);
                let q8_block_bases512_1: Tile<i32, { [512] }> = (q8_batches512
                    + broadcast_scalar(q8_mmq_block_index1 * b_size, const_shape![512]))
                    * broadcast_scalar(144i32, const_shape![512]);
                let d8_byte_offsets16_1: Tile<i32, { [16] }> =
                    q8_block_bases16_1 + broadcast_scalar(q8_mmq_group1 * 4i32, const_shape![16]);
                let d8_offsets16_1: Tile<i32, { [16] }> =
                    d8_byte_offsets16_1 / constant(2i32, const_shape![16]);
                let d8_ptrs1: PointerTile<*mut f16, { [16] }> =
                    q8_half_16.offset_tile(d8_offsets16_1);
                let (d8_h1, _d8_tok1): (Tile<f16, { [16] }>, Token) = load_ptr_tko(
                    d8_ptrs1,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None,
                    Latency::<0>,
                );
                let sum8_ptrs1: PointerTile<*mut f16, { [16] }> =
                    q8_half_16.offset_tile(d8_offsets16_1 + constant(1i32, const_shape![16]));
                let (sum8_h1, _sum8_tok1): (Tile<f16, { [16] }>, Token) = load_ptr_tko(
                    sum8_ptrs1,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None,
                    Latency::<0>,
                );
                let d8_16_1: Tile<f32, { [16] }> = ftof(d8_h1, rounding::NearestEven);
                let sum8_16_1: Tile<f32, { [16] }> = ftof(sum8_h1, rounding::NearestEven);
                let q8_offsets1: Tile<i32, { [512] }> = q8_block_bases512_1
                    + broadcast_scalar(16i32 + q8_mmq_group1 * 32i32, const_shape![512])
                    + q8_k_lane512;
                let q8_ptrs1: PointerTile<*mut i8, { [512] }> = q8_i8_512.offset_tile(q8_offsets1);
                let (q8_bytes1, _q8_tok1): (Tile<i8, { [512] }>, Token) = load_ptr_tko(
                    q8_ptrs1,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let q8_i8_1_tile: Tile<i8, { [32, 16] }> = q8_bytes1.reshape(const_shape![32, 16]);
                let zero_acc1: Tile<i32, { [16, 16] }> = constant(0i32, const_shape![16, 16]);
                let dot1_i32: Tile<i32, { [16, 16] }> = mmai(
                    q4_high_i8,
                    q8_i8_1_tile,
                    zero_acc1,
                    signedness::Unsigned,
                    signedness::Signed,
                );
                let dot1_f32: Tile<f32, { [16, 16] }> = convert_tile(dot1_i32);
                let scale1_f32: Tile<f32, { [16] }> = convert_tile(scale1_i32);
                let min1_f32: Tile<f32, { [16] }> = convert_tile(min1_i32);
                let scale1_1616: Tile<f32, { [16, 16] }> = scale1_f32
                    .reshape(const_shape![16, 1])
                    .broadcast(const_shape![16, 16]);
                let min1_1616: Tile<f32, { [16, 16] }> = min1_f32
                    .reshape(const_shape![16, 1])
                    .broadcast(const_shape![16, 16]);
                let d8_1616_1: Tile<f32, { [16, 16] }> = d8_16_1
                    .reshape(const_shape![1, 16])
                    .broadcast(const_shape![16, 16]);
                let sum8_1616_1: Tile<f32, { [16, 16] }> = sum8_16_1
                    .reshape(const_shape![1, 16])
                    .broadcast(const_shape![16, 16]);
                acc = acc + d1616 * scale1_1616 * d8_1616_1 * dot1_f32
                    - dmin1616 * min1_1616 * sum8_1616_1;
            }
        }

        let out_lanes256: Tile<i32, { [256] }> = iota(const_shape![256]);
        let out_batch_local256: Tile<i32, { [256] }> =
            out_lanes256 / constant(16i32, const_shape![256]);
        let out_row_local256: Tile<i32, { [256] }> =
            out_lanes256 - out_batch_local256 * constant(16i32, const_shape![256]);
        let out_rows256: Tile<i32, { [256] }> =
            out_row_local256 + broadcast_scalar(row_base, const_shape![256]);
        let out_batches256: Tile<i32, { [256] }> =
            out_batch_local256 + broadcast_scalar(batch_base, const_shape![256]);
        let out_offsets: Tile<i32, { [256] }> =
            out_rows256 + out_batches256 * broadcast_scalar(nrows, const_shape![256]);
        let out_base_ptr: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_1: PointerTile<*mut f32, { [1] }> = out_base_ptr.reshape(const_shape![1]);
        let out_256: PointerTile<*mut f32, { [256] }> = out_1.broadcast(const_shape![256]);
        let out_dst: PointerTile<*mut f32, { [256] }> = out_256.offset_tile(out_offsets);
        let acc_batch_major: Tile<f32, { [256] }> = acc.transpose().reshape(const_shape![256]);
        store_ptr_tko(
            out_dst,
            acc_batch_major,
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
mod q6k_q8_1_matvec_rows4_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn q6k_q8_1_matvec_b1_rows4_f32(
        q6_ptr: *mut u8,
        q8_ptr: *mut u8,
        out_ptr: *mut f32,
        ncols: i32,
        nrows: i32,
    ) {
        let pid = get_tile_block_id();
        let row_base: i32 = pid.0 * 4i32;
        if row_base + 3i32 >= nrows {
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

        let row_lanes4: Tile<i32, { [4] }> = iota(const_shape![4]);
        let q_lanes4: Tile<i32, { [4] }> = iota(const_shape![4]);
        let q_lanes16: Tile<i32, { [16] }> = iota(const_shape![16]);
        let c3_16: Tile<i32, { [16] }> = constant(3i32, const_shape![16]);
        let c4_16: Tile<i32, { [16] }> = constant(4i32, const_shape![16]);
        let c15_16: Tile<i32, { [16] }> = constant(15i32, const_shape![16]);
        let c32_16: Tile<i32, { [16] }> = constant(32i32, const_shape![16]);
        let c255_16: Tile<i32, { [16] }> = constant(255i32, const_shape![16]);
        let lane_lanes16: Tile<i32, { [16] }> = q_lanes16 - (q_lanes16 / c4_16) * c4_16;

        let mut acc4: Tile<f32, { [4] }> = constant(0.0f32, const_shape![4]);
        let blocks_per_row: i32 = ncols / 256i32;
        let rows4: Tile<i32, { [4] }> = row_lanes4 + broadcast_scalar(row_base, const_shape![4]);
        let q6_row_bases4: Tile<i32, { [4] }> =
            rows4 * broadcast_scalar(blocks_per_row * 210i32, const_shape![4]);

        for block in 0i32..blocks_per_row {
            let q6_block_bases4: Tile<i32, { [4] }> =
                q6_row_bases4 + broadcast_scalar(block * 210i32, const_shape![4]);
            let q6_block_bases16: Tile<i32, { [16] }> = q6_block_bases4
                .reshape(const_shape![4, 1])
                .broadcast(const_shape![4, 4])
                .reshape(const_shape![16]);

            let d_offsets4: Tile<i32, { [4] }> = (q6_block_bases4
                + constant(208i32, const_shape![4]))
                / constant(2i32, const_shape![4]);
            let q6_half_4: PointerTile<*mut f16, { [4] }> = q6_half_1.broadcast(const_shape![4]);
            let d_ptrs: PointerTile<*mut f16, { [4] }> = q6_half_4.offset_tile(d_offsets4);
            let (d_h4, _d_tok): (Tile<f16, { [4] }>, Token) = load_ptr_tko(
                d_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let d4: Tile<f32, { [4] }> = ftof(d_h4, rounding::NearestEven);
            let d16: Tile<f32, { [16] }> = d4
                .reshape(const_shape![4, 1])
                .broadcast(const_shape![4, 4])
                .reshape(const_shape![16]);

            for iqs in 0i32..32i32 {
                let bq8_offset: i32 = 4i32 * (iqs / 16i32) + (iqs % 16i32) / 8i32;
                let scale_offset: i32 = 8i32 * (iqs / 16i32) + (iqs % 16i32) / 4i32;
                let vh_shift: i32 = 2i32 * ((iqs % 16i32) / 8i32);

                let ql_offsets16: Tile<i32, { [16] }> = q6_block_bases16
                    + lane_lanes16
                    + broadcast_scalar(4i32 * iqs, const_shape![16]);
                let q6_i8_16: PointerTile<*mut i8, { [16] }> = q6_i8_1.broadcast(const_shape![16]);
                let ql_ptrs: PointerTile<*mut i8, { [16] }> = q6_i8_16.offset_tile(ql_offsets16);
                let (ql_bytes, _ql_tok): (Tile<i8, { [16] }>, Token) = load_ptr_tko(
                    ql_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let ql_i32: Tile<i32, { [16] }> = exti(ql_bytes) & c255_16;

                let qh_index: i32 = 8i32 * (iqs / 16i32) + iqs % 8i32;
                let qh_offsets16: Tile<i32, { [16] }> = q6_block_bases16
                    + lane_lanes16
                    + broadcast_scalar(128i32 + 4i32 * qh_index, const_shape![16]);
                let qh_ptrs: PointerTile<*mut i8, { [16] }> = q6_i8_16.offset_tile(qh_offsets16);
                let (qh_bytes, _qh_tok): (Tile<i8, { [16] }>, Token) = load_ptr_tko(
                    qh_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let qh_i32: Tile<i32, { [16] }> = exti(qh_bytes) & c255_16;

                for half in 0i32..2i32 {
                    let ql_shift: Tile<i32, { [16] }> =
                        broadcast_scalar(4i32 * half, const_shape![16]);
                    let qh_shift: Tile<i32, { [16] }> =
                        broadcast_scalar(vh_shift + 4i32 * half, const_shape![16]);
                    let ql_part: Tile<i32, { [16] }> = shri(ql_i32, ql_shift) & c15_16;
                    let qh_part: Tile<i32, { [16] }> = shri(qh_i32, qh_shift) & c3_16;
                    let q6_i32: Tile<i32, { [16] }> =
                        (ql_part | shli(qh_part, c4_16, overflow::NoWrap)) - c32_16;
                    let q6_f32: Tile<f32, { [16] }> = convert_tile(q6_i32);

                    let scale_index: i32 = scale_offset + 4i32 * half;
                    let scale_offsets4: Tile<i32, { [4] }> =
                        q6_block_bases4 + broadcast_scalar(192i32 + scale_index, const_shape![4]);
                    let q6_i8_4: PointerTile<*mut i8, { [4] }> = q6_i8_1.broadcast(const_shape![4]);
                    let scale_ptrs: PointerTile<*mut i8, { [4] }> =
                        q6_i8_4.offset_tile(scale_offsets4);
                    let (scale_i8, _scale_tok): (Tile<i8, { [4] }>, Token) = load_ptr_tko(
                        scale_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let scale_i32: Tile<i32, { [4] }> = exti(scale_i8);
                    let scale_f32: Tile<f32, { [4] }> = convert_tile(scale_i32);
                    let scale16: Tile<f32, { [16] }> = scale_f32
                        .reshape(const_shape![4, 1])
                        .broadcast(const_shape![4, 4])
                        .reshape(const_shape![16]);

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
                    let d8_16: Tile<f32, { [16] }> = d8_1.broadcast(const_shape![16]);

                    let q8_offsets4: Tile<i32, { [4] }> = q_lanes4
                        + broadcast_scalar(
                            q8_block_base + 4i32 + 4i32 * (iqs % 8i32),
                            const_shape![4],
                        );
                    let q8_i8_4: PointerTile<*mut i8, { [4] }> = q8_i8_1.broadcast(const_shape![4]);
                    let q8_ptrs: PointerTile<*mut i8, { [4] }> = q8_i8_4.offset_tile(q8_offsets4);
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
                    let q8_16: Tile<f32, { [16] }> = q8_f32
                        .reshape(const_shape![1, 4])
                        .broadcast(const_shape![4, 4])
                        .reshape(const_shape![16]);

                    let prod16: Tile<f32, { [16] }> = d16 * scale16 * q6_f32 * d8_16 * q8_16;
                    let prod44: Tile<f32, { [4, 4] }> = prod16.reshape(const_shape![4, 4]);
                    let partial4: Tile<f32, { [4] }> = reduce_sum(prod44, 1i32);
                    acc4 = acc4 + partial4;
                }
            }
        }

        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_1: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_4: PointerTile<*mut f32, { [4] }> = out_1.broadcast(const_shape![4]);
        let out_offsets: Tile<i32, { [4] }> =
            row_lanes4 + broadcast_scalar(row_base, const_shape![4]);
        let out_dst: PointerTile<*mut f32, { [4] }> = out_4.offset_tile(out_offsets);
        store_ptr_tko(
            out_dst,
            acc4,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<0>,
        );
    }
}

#[cutile::module]
mod q6k_q8_1_matvec_rows8_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn q6k_q8_1_matvec_b1_rows8_f32(
        q6_ptr: *mut u8,
        q8_ptr: *mut u8,
        out_ptr: *mut f32,
        ncols: i32,
        nrows: i32,
    ) {
        let pid = get_tile_block_id();
        let row_base: i32 = pid.0 * 8i32;
        if row_base + 7i32 >= nrows {
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

        let row_lanes8: Tile<i32, { [8] }> = iota(const_shape![8]);
        let q_lanes4: Tile<i32, { [4] }> = iota(const_shape![4]);
        let q_lanes32: Tile<i32, { [32] }> = iota(const_shape![32]);
        let c3_32: Tile<i32, { [32] }> = constant(3i32, const_shape![32]);
        let c4_32: Tile<i32, { [32] }> = constant(4i32, const_shape![32]);
        let c15_32: Tile<i32, { [32] }> = constant(15i32, const_shape![32]);
        let c32_32: Tile<i32, { [32] }> = constant(32i32, const_shape![32]);
        let c255_32: Tile<i32, { [32] }> = constant(255i32, const_shape![32]);
        let lane_lanes32: Tile<i32, { [32] }> = q_lanes32 - (q_lanes32 / c4_32) * c4_32;

        let mut acc8: Tile<f32, { [8] }> = constant(0.0f32, const_shape![8]);
        let blocks_per_row: i32 = ncols / 256i32;
        let rows8: Tile<i32, { [8] }> = row_lanes8 + broadcast_scalar(row_base, const_shape![8]);
        let q6_row_bases8: Tile<i32, { [8] }> =
            rows8 * broadcast_scalar(blocks_per_row * 210i32, const_shape![8]);

        for block in 0i32..blocks_per_row {
            let q6_block_bases8: Tile<i32, { [8] }> =
                q6_row_bases8 + broadcast_scalar(block * 210i32, const_shape![8]);
            let q6_block_bases32: Tile<i32, { [32] }> = q6_block_bases8
                .reshape(const_shape![8, 1])
                .broadcast(const_shape![8, 4])
                .reshape(const_shape![32]);

            let d_offsets8: Tile<i32, { [8] }> = (q6_block_bases8
                + constant(208i32, const_shape![8]))
                / constant(2i32, const_shape![8]);
            let q6_half_8: PointerTile<*mut f16, { [8] }> = q6_half_1.broadcast(const_shape![8]);
            let d_ptrs: PointerTile<*mut f16, { [8] }> = q6_half_8.offset_tile(d_offsets8);
            let (d_h8, _d_tok): (Tile<f16, { [8] }>, Token) = load_ptr_tko(
                d_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let d8_rows: Tile<f32, { [8] }> = ftof(d_h8, rounding::NearestEven);
            let d32: Tile<f32, { [32] }> = d8_rows
                .reshape(const_shape![8, 1])
                .broadcast(const_shape![8, 4])
                .reshape(const_shape![32]);

            for iqs in 0i32..32i32 {
                let bq8_offset: i32 = 4i32 * (iqs / 16i32) + (iqs % 16i32) / 8i32;
                let scale_offset: i32 = 8i32 * (iqs / 16i32) + (iqs % 16i32) / 4i32;
                let vh_shift: i32 = 2i32 * ((iqs % 16i32) / 8i32);

                let ql_offsets32: Tile<i32, { [32] }> = q6_block_bases32
                    + lane_lanes32
                    + broadcast_scalar(4i32 * iqs, const_shape![32]);
                let q6_i8_32: PointerTile<*mut i8, { [32] }> = q6_i8_1.broadcast(const_shape![32]);
                let ql_ptrs: PointerTile<*mut i8, { [32] }> = q6_i8_32.offset_tile(ql_offsets32);
                let (ql_bytes, _ql_tok): (Tile<i8, { [32] }>, Token) = load_ptr_tko(
                    ql_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let ql_i32: Tile<i32, { [32] }> = exti(ql_bytes) & c255_32;

                let qh_index: i32 = 8i32 * (iqs / 16i32) + iqs % 8i32;
                let qh_offsets32: Tile<i32, { [32] }> = q6_block_bases32
                    + lane_lanes32
                    + broadcast_scalar(128i32 + 4i32 * qh_index, const_shape![32]);
                let qh_ptrs: PointerTile<*mut i8, { [32] }> = q6_i8_32.offset_tile(qh_offsets32);
                let (qh_bytes, _qh_tok): (Tile<i8, { [32] }>, Token) = load_ptr_tko(
                    qh_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    Some(0i8),
                    None,
                    Latency::<0>,
                );
                let qh_i32: Tile<i32, { [32] }> = exti(qh_bytes) & c255_32;

                for half in 0i32..2i32 {
                    let ql_shift: Tile<i32, { [32] }> =
                        broadcast_scalar(4i32 * half, const_shape![32]);
                    let qh_shift: Tile<i32, { [32] }> =
                        broadcast_scalar(vh_shift + 4i32 * half, const_shape![32]);
                    let ql_part: Tile<i32, { [32] }> = shri(ql_i32, ql_shift) & c15_32;
                    let qh_part: Tile<i32, { [32] }> = shri(qh_i32, qh_shift) & c3_32;
                    let q6_i32: Tile<i32, { [32] }> =
                        (ql_part | shli(qh_part, c4_32, overflow::NoWrap)) - c32_32;
                    let q6_f32: Tile<f32, { [32] }> = convert_tile(q6_i32);

                    let scale_index: i32 = scale_offset + 4i32 * half;
                    let scale_offsets8: Tile<i32, { [8] }> =
                        q6_block_bases8 + broadcast_scalar(192i32 + scale_index, const_shape![8]);
                    let q6_i8_8: PointerTile<*mut i8, { [8] }> = q6_i8_1.broadcast(const_shape![8]);
                    let scale_ptrs: PointerTile<*mut i8, { [8] }> =
                        q6_i8_8.offset_tile(scale_offsets8);
                    let (scale_i8, _scale_tok): (Tile<i8, { [8] }>, Token) = load_ptr_tko(
                        scale_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let scale_i32: Tile<i32, { [8] }> = exti(scale_i8);
                    let scale_f32: Tile<f32, { [8] }> = convert_tile(scale_i32);
                    let scale32: Tile<f32, { [32] }> = scale_f32
                        .reshape(const_shape![8, 1])
                        .broadcast(const_shape![8, 4])
                        .reshape(const_shape![32]);

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
                    let d8_32: Tile<f32, { [32] }> = d8_1.broadcast(const_shape![32]);

                    let q8_offsets4: Tile<i32, { [4] }> = q_lanes4
                        + broadcast_scalar(
                            q8_block_base + 4i32 + 4i32 * (iqs % 8i32),
                            const_shape![4],
                        );
                    let q8_i8_4: PointerTile<*mut i8, { [4] }> = q8_i8_1.broadcast(const_shape![4]);
                    let q8_ptrs: PointerTile<*mut i8, { [4] }> = q8_i8_4.offset_tile(q8_offsets4);
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
                    let q8_32: Tile<f32, { [32] }> = q8_f32
                        .reshape(const_shape![1, 4])
                        .broadcast(const_shape![8, 4])
                        .reshape(const_shape![32]);

                    let prod32: Tile<f32, { [32] }> = d32 * scale32 * q6_f32 * d8_32 * q8_32;
                    let prod84: Tile<f32, { [8, 4] }> = prod32.reshape(const_shape![8, 4]);
                    let partial8: Tile<f32, { [8] }> = reduce_sum(prod84, 1i32);
                    acc8 = acc8 + partial8;
                }
            }
        }

        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_1: PointerTile<*mut f32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_8: PointerTile<*mut f32, { [8] }> = out_1.broadcast(const_shape![8]);
        let out_offsets: Tile<i32, { [8] }> =
            row_lanes8 + broadcast_scalar(row_base, const_shape![8]);
        let out_dst: PointerTile<*mut f32, { [8] }> = out_8.offset_tile(out_offsets);
        store_ptr_tko(
            out_dst,
            acc8,
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
    let mut scratch_guard = q8_scratch_workspace_ensure(dev, scratch_bytes)?;
    let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };

    let timing_enabled = cutile_trace_enabled();
    let total_host_start = std::time::Instant::now();
    let total_start_event = record_timing_event(&stream, timing_enabled)?;
    let total_end_event;
    let q8_quant_events;
    let mut q4k_mmai_main_events = None;
    let mut q4k_mmai_tail_events = None;
    let mut q4k_scalar_mmq_events = None;
    let mut q4k_tiled_mmq_events = None;
    let mut q4k_matvec_b1_rows4_events = None;
    let mut q6k_mmq_events = None;
    let mut q6k_matvec_b1_rows4_events = None;
    let mut q6k_matvec_b1_rows8_events = None;
    let mut q4k_mmai_main_host_ms = None;
    let mut q4k_mmai_tail_host_ms = None;
    let mut q4k_scalar_mmq_host_ms = None;
    let mut q4k_tiled_mmq_host_ms = None;
    let mut q4k_matvec_b1_rows4_host_ms = None;
    let mut q6k_mmq_host_ms = None;
    let mut q6k_matvec_b1_rows4_host_ms = None;
    let mut q6k_matvec_b1_rows8_host_ms = None;
    let trace_main_kernel: &'static str;
    let trace_tail_kernel: &'static str;
    let trace_main_b_size: usize;
    let trace_tail_b_size: usize;

    {
        let (scratch_ptr, scratch_write) = scratch_guard.slice.device_ptr_mut(&stream);
        let q8_quant_start_event = record_timing_event(&stream, timing_enabled)?;
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
        let q8_quant_end_event = record_timing_event(&stream, timing_enabled)?;
        q8_quant_events = q8_quant_start_event.zip(q8_quant_end_event);

        let (qweight_ptr, qweight_read) = qstorage.device_ptr_with_guard(&stream)?;
        let (out_ptr, out_write) = out.device_ptr_mut(&stream);

        {
            let (_cutile_device, cutile_stream) = borrow_candle_cuda_handles(dev)?;
            match w_dtype {
                GgmlDType::Q4K => {
                    if b_size == 1 {
                        trace_tail_kernel = "none";
                        trace_main_b_size = b_size;
                        trace_tail_b_size = 0;
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
                        if q4k_b1_rows4_enabled() && nrows.is_multiple_of(4) {
                            trace_main_kernel = "q4k_q8_1_matvec_b1_rows4_f32";
                            let q4k_rows4_start_event =
                                record_timing_event(&stream, timing_enabled)?;
                            let op = unsafe {
                                q4k_q8_1_matvec_rows4_kernel::q4k_q8_1_matvec_b1_rows4_f32(
                                    qweight_cutile,
                                    q8_cutile,
                                    out_cutile,
                                    ncols as i32,
                                    nrows as i32,
                                )
                            }
                            .grid(((nrows / 4) as u32, 1, 1));

                            // SAFETY: all pointers refer to live Candle-owned CUDA allocations.
                            // Each cuTile program owns four contiguous output rows, while the
                            // Q4K storage and Q8 scratch are read-only and stream-ordered after
                            // Q8 quantization on Candle's stream.
                            let q4k_rows4_host_start = std::time::Instant::now();
                            unsafe { op.async_on(&cutile_stream) }.map_err(cutile_err)?;
                            q4k_matvec_b1_rows4_host_ms =
                                Some(q4k_rows4_host_start.elapsed().as_secs_f64() * 1000.0);
                            let q4k_rows4_end_event = record_timing_event(&stream, timing_enabled)?;
                            q4k_matvec_b1_rows4_events =
                                q4k_rows4_start_event.zip(q4k_rows4_end_event);
                        } else {
                            trace_main_kernel = "q4k_q8_1_matvec_b1_f32";
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
                        }
                    } else if use_mmq_q8_layout
                        && mmai_prefill_enabled()
                        && nrows.is_multiple_of(16)
                        && b_size >= 16
                        && b_size.is_multiple_of(4)
                    {
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
                        let main_b_size = (b_size / 16) * 16;
                        let tail_b_size = b_size - main_b_size;
                        trace_main_kernel = "q4k_q8_1_mmq_matmul_mmai_16x16_f32";
                        trace_tail_kernel = if tail_b_size != 0 {
                            "q4k_q8_1_mmq_matmul_tiled_f32"
                        } else {
                            "none"
                        };
                        trace_main_b_size = main_b_size;
                        trace_tail_b_size = tail_b_size;

                        let q4k_mmai_main_start_event =
                            record_timing_event(&stream, timing_enabled)?;
                        let op = unsafe {
                            q4k_q8_1_mmq_matmul_mmai_kernel::q4k_q8_1_mmq_matmul_mmai_16x16_f32(
                                qweight_cutile,
                                q8_cutile,
                                out_cutile,
                                ncols as i32,
                                nrows as i32,
                                b_size as i32,
                                0,
                            )
                        }
                        .grid((
                            (nrows / 16) as u32,
                            (main_b_size / 16) as u32,
                            1,
                        ));

                        // SAFETY: same Candle-owned allocation and stream-ordering argument
                        // as the scalar baseline, but each cuTile program owns a disjoint
                        // 16-row by 16-RHS output tile and the optional tail below writes
                        // non-overlapping RHS columns.
                        let q4k_mmai_main_host_start = std::time::Instant::now();
                        unsafe { op.async_on(&cutile_stream) }.map_err(cutile_err)?;
                        q4k_mmai_main_host_ms =
                            Some(q4k_mmai_main_host_start.elapsed().as_secs_f64() * 1000.0);
                        let q4k_mmai_main_end_event = record_timing_event(&stream, timing_enabled)?;
                        q4k_mmai_main_events =
                            q4k_mmai_main_start_event.zip(q4k_mmai_main_end_event);

                        if tail_b_size != 0 {
                            let q4k_mmai_tail_start_event =
                                record_timing_event(&stream, timing_enabled)?;
                            let op = unsafe {
                                q4k_q8_1_mmq_matmul_tiled_kernel::q4k_q8_1_mmq_matmul_tiled_f32(
                                    qweight_cutile,
                                    q8_cutile,
                                    out_cutile,
                                    ncols as i32,
                                    nrows as i32,
                                    b_size as i32,
                                    main_b_size as i32,
                                )
                            }
                            .grid((
                                (nrows / 4) as u32,
                                (tail_b_size / 4) as u32,
                                1,
                            ));

                            // SAFETY: tail launch covers only RHS columns [main_b_size, b_size)
                            // using the same MMQ Q8_1 scratch layout and output allocation.
                            let q4k_mmai_tail_host_start = std::time::Instant::now();
                            unsafe { op.async_on(&cutile_stream) }.map_err(cutile_err)?;
                            q4k_mmai_tail_host_ms =
                                Some(q4k_mmai_tail_host_start.elapsed().as_secs_f64() * 1000.0);
                            let q4k_mmai_tail_end_event =
                                record_timing_event(&stream, timing_enabled)?;
                            q4k_mmai_tail_events =
                                q4k_mmai_tail_start_event.zip(q4k_mmai_tail_end_event);
                        }
                    } else if use_mmq_q8_layout
                        && tiled_prefill_enabled()
                        && nrows.is_multiple_of(4)
                        && b_size.is_multiple_of(4)
                    {
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
                        trace_main_kernel = "q4k_q8_1_mmq_matmul_tiled_f32";
                        trace_tail_kernel = "none";
                        trace_main_b_size = b_size;
                        trace_tail_b_size = 0;
                        let q4k_tiled_mmq_start_event =
                            record_timing_event(&stream, timing_enabled)?;
                        let op = unsafe {
                            q4k_q8_1_mmq_matmul_tiled_kernel::q4k_q8_1_mmq_matmul_tiled_f32(
                                qweight_cutile,
                                q8_cutile,
                                out_cutile,
                                ncols as i32,
                                nrows as i32,
                                b_size as i32,
                                0,
                            )
                        }
                        .grid(((nrows / 4) as u32, (b_size / 4) as u32, 1));

                        // SAFETY: same Candle-owned allocation and stream-ordering argument
                        // as the scalar baseline, but each cuTile program owns a disjoint
                        // 4-row by 4-RHS output tile.
                        let q4k_tiled_mmq_host_start = std::time::Instant::now();
                        unsafe { op.async_on(&cutile_stream) }.map_err(cutile_err)?;
                        q4k_tiled_mmq_host_ms =
                            Some(q4k_tiled_mmq_host_start.elapsed().as_secs_f64() * 1000.0);
                        let q4k_tiled_mmq_end_event = record_timing_event(&stream, timing_enabled)?;
                        q4k_tiled_mmq_events =
                            q4k_tiled_mmq_start_event.zip(q4k_tiled_mmq_end_event);
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
                        trace_main_kernel = "q4k_q8_1_mmq_matmul_batched_f32";
                        trace_tail_kernel = "none";
                        trace_main_b_size = b_size;
                        trace_tail_b_size = 0;
                        let q4k_scalar_mmq_start_event =
                            record_timing_event(&stream, timing_enabled)?;
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
                        let q4k_scalar_mmq_host_start = std::time::Instant::now();
                        unsafe { op.async_on(&cutile_stream) }.map_err(cutile_err)?;
                        q4k_scalar_mmq_host_ms =
                            Some(q4k_scalar_mmq_host_start.elapsed().as_secs_f64() * 1000.0);
                        let q4k_scalar_mmq_end_event =
                            record_timing_event(&stream, timing_enabled)?;
                        q4k_scalar_mmq_events =
                            q4k_scalar_mmq_start_event.zip(q4k_scalar_mmq_end_event);
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
                        trace_main_kernel = "q4k_q8_1_matmul_batched_f32";
                        trace_tail_kernel = "none";
                        trace_main_b_size = b_size;
                        trace_tail_b_size = 0;
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
                        trace_tail_kernel = "none";
                        trace_main_b_size = b_size;
                        trace_tail_b_size = 0;
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
                        if q6k_b1_rows8_enabled() && nrows.is_multiple_of(8) {
                            trace_main_kernel = "q6k_q8_1_matvec_b1_rows8_f32";
                            let q6k_rows8_start_event =
                                record_timing_event(&stream, timing_enabled)?;
                            let op = unsafe {
                                q6k_q8_1_matvec_rows8_kernel::q6k_q8_1_matvec_b1_rows8_f32(
                                    qweight_cutile,
                                    q8_cutile,
                                    out_cutile,
                                    ncols as i32,
                                    nrows as i32,
                                )
                            }
                            .grid(((nrows / 8) as u32, 1, 1));

                            // SAFETY: each cuTile program owns eight contiguous output rows.
                            // The Q6K weights and Q8 scratch are read-only and all work is
                            // enqueued on Candle's stream after Q8 quantization.
                            let q6k_rows8_host_start = std::time::Instant::now();
                            unsafe { op.async_on(&cutile_stream) }.map_err(cutile_err)?;
                            q6k_matvec_b1_rows8_host_ms =
                                Some(q6k_rows8_host_start.elapsed().as_secs_f64() * 1000.0);
                            let q6k_rows8_end_event = record_timing_event(&stream, timing_enabled)?;
                            q6k_matvec_b1_rows8_events =
                                q6k_rows8_start_event.zip(q6k_rows8_end_event);
                        } else if q6k_b1_rows4_enabled() && nrows.is_multiple_of(4) {
                            trace_main_kernel = "q6k_q8_1_matvec_b1_rows4_f32";
                            let q6k_rows4_start_event =
                                record_timing_event(&stream, timing_enabled)?;
                            let op = unsafe {
                                q6k_q8_1_matvec_rows4_kernel::q6k_q8_1_matvec_b1_rows4_f32(
                                    qweight_cutile,
                                    q8_cutile,
                                    out_cutile,
                                    ncols as i32,
                                    nrows as i32,
                                )
                            }
                            .grid(((nrows / 4) as u32, 1, 1));

                            // SAFETY: each cuTile program owns four contiguous output rows.
                            // The Q6K weights and Q8 scratch are read-only and all work is
                            // enqueued on Candle's stream after Q8 quantization.
                            let q6k_rows4_host_start = std::time::Instant::now();
                            unsafe { op.async_on(&cutile_stream) }.map_err(cutile_err)?;
                            q6k_matvec_b1_rows4_host_ms =
                                Some(q6k_rows4_host_start.elapsed().as_secs_f64() * 1000.0);
                            let q6k_rows4_end_event = record_timing_event(&stream, timing_enabled)?;
                            q6k_matvec_b1_rows4_events =
                                q6k_rows4_start_event.zip(q6k_rows4_end_event);
                        } else {
                            trace_main_kernel = "q6k_q8_1_matvec_b1_f32";
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
                        }
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
                        trace_main_kernel = "q6k_q8_1_mmq_matmul_batched_f32";
                        trace_tail_kernel = "none";
                        trace_main_b_size = b_size;
                        trace_tail_b_size = 0;
                        let q6k_mmq_start_event = record_timing_event(&stream, timing_enabled)?;
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
                        let q6k_mmq_host_start = std::time::Instant::now();
                        unsafe { op.async_on(&cutile_stream) }.map_err(cutile_err)?;
                        q6k_mmq_host_ms = Some(q6k_mmq_host_start.elapsed().as_secs_f64() * 1000.0);
                        let q6k_mmq_end_event = record_timing_event(&stream, timing_enabled)?;
                        q6k_mmq_events = q6k_mmq_start_event.zip(q6k_mmq_end_event);
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
                        trace_main_kernel = "q6k_q8_1_matmul_batched_f32";
                        trace_tail_kernel = "none";
                        trace_main_b_size = b_size;
                        trace_tail_b_size = 0;
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

        total_end_event = record_timing_event(&stream, timing_enabled)?;

        drop(out_write);
        drop(qweight_read);
        drop(scratch_write);
    }

    if timing_enabled {
        let total_gpu_events = total_start_event.zip(total_end_event);
        let q8_quant_ms = elapsed_timing_ms(&q8_quant_events)?;
        let q4k_mmai_main_ms = elapsed_timing_ms(&q4k_mmai_main_events)?;
        let q4k_mmai_tail_ms = elapsed_timing_ms(&q4k_mmai_tail_events)?;
        let q4k_scalar_mmq_ms = elapsed_timing_ms(&q4k_scalar_mmq_events)?;
        let q4k_tiled_mmq_ms = elapsed_timing_ms(&q4k_tiled_mmq_events)?;
        let q4k_matvec_b1_rows4_ms = elapsed_timing_ms(&q4k_matvec_b1_rows4_events)?;
        let q6k_mmq_ms = elapsed_timing_ms(&q6k_mmq_events)?;
        let q6k_matvec_b1_rows4_ms = elapsed_timing_ms(&q6k_matvec_b1_rows4_events)?;
        let q6k_matvec_b1_rows8_ms = elapsed_timing_ms(&q6k_matvec_b1_rows8_events)?;
        let total_gpu_ms = elapsed_timing_ms(&total_gpu_events)?;
        let decode_matvec_gpu_ms = if b_size == 1 {
            q4k_matvec_b1_rows4_ms
                .or(q6k_matvec_b1_rows8_ms)
                .or(q6k_matvec_b1_rows4_ms)
                .or(total_gpu_ms)
        } else {
            None
        };
        let decode_matvec_kernel = if b_size == 1 {
            trace_main_kernel
        } else {
            "n/a"
        };
        let estimated_weight_bytes = if b_size == 1 {
            estimated_decode_weight_bytes(w_dtype, nrows, ncols)
        } else {
            None
        };
        let estimated_bandwidth_gbps = estimated_gbps(estimated_weight_bytes, decode_matvec_gpu_ms);
        let total_host_ms = total_host_start.elapsed().as_secs_f64() * 1000.0;
        trace_cutile_measurement(format_args!(
            "cutile_timing dtype={:?} rhs_dtype={:?} nrows={} ncols={} b_size={} main_kernel={} tail_kernel={} main_b_size={} tail_b_size={} q8_quant_ms={} q4k_mmai_main_ms={} q4k_mmai_tail_ms={} q4k_scalar_mmq_ms={} q4k_tiled_mmq_ms={} q4k_matvec_b1_rows4_ms={} q6k_mmq_ms={} q6k_matvec_b1_rows4_ms={} q6k_matvec_b1_rows8_ms={} total_gpu_ms={} decode_matvec_kernel={} decode_matvec_gpu_ms={} estimated_weight_bytes={} estimated_gbps={} q4k_mmai_main_host_submit_or_jit_ms={} q4k_mmai_tail_host_submit_or_jit_ms={} q4k_scalar_mmq_host_submit_or_jit_ms={} q4k_tiled_mmq_host_submit_or_jit_ms={} q4k_matvec_b1_rows4_host_submit_or_jit_ms={} q6k_mmq_host_submit_or_jit_ms={} q6k_matvec_b1_rows4_host_submit_or_jit_ms={} q6k_matvec_b1_rows8_host_submit_or_jit_ms={} total_host_submit_or_jit_ms={:.3}",
            w_dtype,
            rhs.dtype(),
            nrows,
            ncols,
            b_size,
            trace_main_kernel,
            trace_tail_kernel,
            trace_main_b_size,
            trace_tail_b_size,
            fmt_timing_ms(q8_quant_ms),
            fmt_timing_ms(q4k_mmai_main_ms),
            fmt_timing_ms(q4k_mmai_tail_ms),
            fmt_timing_ms(q4k_scalar_mmq_ms),
            fmt_timing_ms(q4k_tiled_mmq_ms),
            fmt_timing_ms(q4k_matvec_b1_rows4_ms),
            fmt_timing_ms(q6k_mmq_ms),
            fmt_timing_ms(q6k_matvec_b1_rows4_ms),
            fmt_timing_ms(q6k_matvec_b1_rows8_ms),
            fmt_timing_ms(total_gpu_ms),
            decode_matvec_kernel,
            fmt_timing_ms(decode_matvec_gpu_ms),
            fmt_usize(estimated_weight_bytes),
            fmt_gbps(estimated_bandwidth_gbps),
            fmt_host_ms(q4k_mmai_main_host_ms),
            fmt_host_ms(q4k_mmai_tail_host_ms),
            fmt_host_ms(q4k_scalar_mmq_host_ms),
            fmt_host_ms(q4k_tiled_mmq_host_ms),
            fmt_host_ms(q4k_matvec_b1_rows4_host_ms),
            fmt_host_ms(q6k_mmq_host_ms),
            fmt_host_ms(q6k_matvec_b1_rows4_host_ms),
            fmt_host_ms(q6k_matvec_b1_rows8_host_ms),
            total_host_ms,
        ));
    }

    let mut out_shape = rhs_l.shape().dims().to_vec();
    out_shape.pop();
    out_shape.push(nrows);
    Ok(Some((
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        out_shape.into(),
    )))
}

/// Launches a tiny cuTile `mmai` kernel on Candle's CUDA stream.
///
/// The kernel computes a synthetic `u8/s8` 16x16x32 dot product using the same
/// `mmai` shape as the Q4K MMQ main-body kernel. Each output should be 32.
#[allow(dead_code)]
pub(crate) fn borrowed_candle_stream_mmai_probe(cuda: &CudaDevice) -> Result<Vec<i32>> {
    use cudarc::driver::DevicePtrMut;
    use cutile::cuda_async::device_buffer::DevicePointer;
    use cutile::cuda_async::device_operation::DeviceOp;
    use cutile::tile_kernel::TileKernel;

    const LEN: usize = 16 * 16;

    let mut out = unsafe { cuda.alloc::<i32>(LEN)? };

    {
        let candle_stream = cuda.cuda_stream();
        let (out_ptr, record_out_write) = out.device_ptr_mut(&candle_stream);
        let cutile_out = unsafe {
            DevicePointer::<i32>::from_cu_deviceptr(out_ptr as cutile::cuda_core::sys::CUdeviceptr)
        };

        {
            let (_cutile_device, cutile_stream) = borrow_candle_cuda_handles(cuda)?;
            let op = unsafe { mmai_probe_kernel::mmai_i8_16x16_probe(cutile_out) }.grid((1, 1, 1));

            // SAFETY: `cutile_out` points at `out`, a live Candle-owned CUDA
            // allocation of LEN i32 values. The probe writes each element once.
            unsafe { op.async_on(&cutile_stream) }.map_err(cutile_err)?;
        }

        drop(record_out_write);
    }

    cuda.synchronize()?;
    Ok(cuda.clone_dtoh(&out)?)
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

    #[test]
    #[ignore = "requires CUDA 13.2+/cuTile runtime and a CUDA device"]
    fn cutile_borrowed_candle_stream_mmai_probe_works() {
        let device = Device::new_cuda(0).unwrap();
        let cuda = device.as_cuda_device().unwrap();

        let values = borrowed_candle_stream_mmai_probe(cuda).unwrap();
        assert_eq!(values.len(), 16 * 16);
        for (index, value) in values.iter().enumerate() {
            assert_eq!(*value, 32, "unexpected mmai probe value at index {index}");
        }
    }

    fn run_cutile_qk_q8_1_b1_matches_candle_cuda_matvec(
        dtype: GgmlDType,
        rows4: bool,
        rows8: bool,
    ) {
        let _env_guard = QUANT_KERNEL_ENV_LOCK.lock().unwrap();
        let _q4k_rows4_guard = EnvVarGuard::save("PI_CANDLE_QUANT_CUTILE_Q4K_B1_ROWS4");
        let _q6k_rows4_guard = EnvVarGuard::save("PI_CANDLE_QUANT_CUTILE_Q6K_B1_ROWS4");
        let _q6k_rows8_guard = EnvVarGuard::save("PI_CANDLE_QUANT_CUTILE_Q6K_B1_ROWS8");
        std::thread::Builder::new()
            .name("pi-ai-candle-worker".to_string())
            .spawn(move || {
                std::env::set_var("PI_CANDLE_QUANT_KERNEL", "candle");
                std::env::remove_var("PI_CANDLE_QUANT_FALLBACK");
                std::env::remove_var("PI_CANDLE_QUANT_CUTILE_Q4K_B1_ROWS4");
                std::env::remove_var("PI_CANDLE_QUANT_CUTILE_Q6K_B1_ROWS4");
                std::env::remove_var("PI_CANDLE_QUANT_CUTILE_Q6K_B1_ROWS8");
                if rows4 {
                    match dtype {
                        GgmlDType::Q4K => {
                            std::env::set_var("PI_CANDLE_QUANT_CUTILE_Q4K_B1_ROWS4", "1")
                        }
                        GgmlDType::Q6K => {
                            std::env::set_var("PI_CANDLE_QUANT_CUTILE_Q6K_B1_ROWS4", "1")
                        }
                        _ => {}
                    }
                }
                if rows8 && dtype == GgmlDType::Q6K {
                    std::env::set_var("PI_CANDLE_QUANT_CUTILE_Q6K_B1_ROWS8", "1");
                }

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
        tiled_prefill: bool,
        mmai_prefill: bool,
    ) {
        let _env_guard = QUANT_KERNEL_ENV_LOCK.lock().unwrap();
        let _tiled_guard = EnvVarGuard::save("PI_CANDLE_QUANT_CUTILE_TILED");
        let _mmai_guard = EnvVarGuard::save("PI_CANDLE_QUANT_CUTILE_MMAI");
        std::thread::Builder::new()
            .name("pi-ai-candle-worker".to_string())
            .spawn(move || {
                std::env::set_var("PI_CANDLE_QUANT_KERNEL", "candle");
                std::env::remove_var("PI_CANDLE_QUANT_FALLBACK");
                if tiled_prefill {
                    std::env::set_var("PI_CANDLE_QUANT_CUTILE_TILED", "1");
                } else {
                    std::env::remove_var("PI_CANDLE_QUANT_CUTILE_TILED");
                }
                if mmai_prefill {
                    std::env::set_var("PI_CANDLE_QUANT_CUTILE_MMAI", "1");
                } else {
                    std::env::remove_var("PI_CANDLE_QUANT_CUTILE_MMAI");
                }

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
        run_cutile_qk_q8_1_b1_matches_candle_cuda_matvec(GgmlDType::Q4K, false, false);
    }

    #[test]
    #[ignore = "requires CUDA 13.2+/cuTile runtime and a CUDA device"]
    fn cutile_q4k_q8_1_b1_rows4_matches_candle_cuda_matvec() {
        run_cutile_qk_q8_1_b1_matches_candle_cuda_matvec(GgmlDType::Q4K, true, false);
    }

    #[test]
    #[ignore = "requires CUDA 13.2+/cuTile runtime and a CUDA device"]
    fn cutile_q4k_q8_1_batched_matches_candle_cuda_matmul() {
        run_cutile_qk_q8_1_batched_matches_candle_cuda_matmul(
            GgmlDType::Q4K,
            4096,
            2560,
            372,
            false,
            false,
        );
    }

    #[test]
    #[ignore = "requires CUDA 13.2+/cuTile runtime and a CUDA device"]
    fn cutile_q4k_q8_1_mmq_tiled_matches_candle_cuda_matmul() {
        run_cutile_qk_q8_1_batched_matches_candle_cuda_matmul(
            GgmlDType::Q4K,
            4096,
            2560,
            372,
            true,
            false,
        );
    }

    #[test]
    #[ignore = "requires CUDA 13.2+/cuTile runtime and a CUDA device"]
    fn cutile_q4k_q8_1_mmq_mmai_small_matches_candle_cuda_matmul() {
        run_cutile_qk_q8_1_batched_matches_candle_cuda_matmul(
            GgmlDType::Q4K,
            64,
            512,
            16,
            false,
            true,
        );
    }

    #[test]
    #[ignore = "requires CUDA 13.2+/cuTile runtime and a CUDA device"]
    fn cutile_q4k_q8_1_mmq_mmai_matches_candle_cuda_matmul() {
        run_cutile_qk_q8_1_batched_matches_candle_cuda_matmul(
            GgmlDType::Q4K,
            4096,
            2560,
            372,
            false,
            true,
        );
    }

    fn timed_forward_ms(matmul: &quantized::QMatMul, x: &Tensor, cuda: &CudaDevice) -> f64 {
        let start = std::time::Instant::now();
        let _ = matmul.forward(x).unwrap();
        cuda.synchronize().unwrap();
        start.elapsed().as_secs_f64() * 1000.0
    }

    #[test]
    #[ignore = "requires CUDA 13.2+/cuTile runtime and a CUDA device"]
    fn cutile_q4k_q8_1_benchmark_variants() {
        let _env_guard = QUANT_KERNEL_ENV_LOCK.lock().unwrap();
        let _kernel_guard = EnvVarGuard::save("PI_CANDLE_QUANT_KERNEL");
        let _fallback_guard = EnvVarGuard::save("PI_CANDLE_QUANT_FALLBACK");
        let _tiled_guard = EnvVarGuard::save("PI_CANDLE_QUANT_CUTILE_TILED");
        let _mmai_guard = EnvVarGuard::save("PI_CANDLE_QUANT_CUTILE_MMAI");

        std::thread::Builder::new()
            .name("pi-ai-candle-worker".to_string())
            .spawn(|| {
                std::env::remove_var("PI_CANDLE_QUANT_FALLBACK");

                let warm_iters = std::env::var("PI_CANDLE_QUANT_BENCH_ITERS")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(3)
                    .max(1);

                let device = Device::new_cuda(0).unwrap();
                let cuda = device.as_cuda_device().unwrap();
                let nrows = 4096usize;
                let ncols = 2560usize;
                let b_size = 372usize;

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
                let qtensor = quantized::QTensor::quantize(&weights, GgmlDType::Q4K).unwrap();
                let matmul = quantized::QMatMul::from_qtensor(qtensor).unwrap();

                for (label, backend, tiled, mmai) in [
                    ("candle-cuda", "candle", false, false),
                    ("cutile-scalar", "cutile", false, false),
                    ("cutile-tiled-4x4", "cutile", true, false),
                    ("cutile-mmai-16x16", "cutile", false, true),
                ] {
                    std::env::set_var("PI_CANDLE_QUANT_KERNEL", backend);
                    if tiled {
                        std::env::set_var("PI_CANDLE_QUANT_CUTILE_TILED", "1");
                    } else {
                        std::env::remove_var("PI_CANDLE_QUANT_CUTILE_TILED");
                    }
                    if mmai {
                        std::env::set_var("PI_CANDLE_QUANT_CUTILE_MMAI", "1");
                    } else {
                        std::env::remove_var("PI_CANDLE_QUANT_CUTILE_MMAI");
                    }

                    let cold_ms = timed_forward_ms(&matmul, &x, cuda);
                    let mut warm_ms = Vec::with_capacity(warm_iters);
                    for _ in 0..warm_iters {
                        warm_ms.push(timed_forward_ms(&matmul, &x, cuda));
                    }
                    let warm_mean_ms = warm_ms.iter().sum::<f64>() / warm_ms.len() as f64;
                    let warm_min_ms = warm_ms.iter().copied().fold(f64::INFINITY, f64::min);
                    eprintln!(
                        "candle quant-kernel benchmark variant={label} cold_ms={cold_ms:.3} warm_mean_ms={warm_mean_ms:.3} warm_min_ms={warm_min_ms:.3} warm_iters={warm_iters} nrows={nrows} ncols={ncols} b_size={b_size}"
                    );
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    #[ignore = "requires CUDA 13.2+/cuTile runtime and a CUDA device"]
    fn cutile_q6k_q8_1_batched_matches_candle_cuda_matmul() {
        run_cutile_qk_q8_1_batched_matches_candle_cuda_matmul(
            GgmlDType::Q6K,
            1024,
            2560,
            372,
            false,
            false,
        );
    }

    #[test]
    #[ignore = "requires CUDA 13.2+/cuTile runtime and a CUDA device"]
    fn cutile_q6k_q8_1_b1_matches_candle_cuda_matvec() {
        run_cutile_qk_q8_1_b1_matches_candle_cuda_matvec(GgmlDType::Q6K, false, false);
    }

    #[test]
    #[ignore = "requires CUDA 13.2+/cuTile runtime and a CUDA device"]
    fn cutile_q6k_q8_1_b1_rows4_matches_candle_cuda_matvec() {
        run_cutile_qk_q8_1_b1_matches_candle_cuda_matvec(GgmlDType::Q6K, true, false);
    }

    #[test]
    #[ignore = "requires CUDA 13.2+/cuTile runtime and a CUDA device"]
    fn cutile_q6k_q8_1_b1_rows8_matches_candle_cuda_matvec() {
        run_cutile_qk_q8_1_b1_matches_candle_cuda_matvec(GgmlDType::Q6K, false, true);
    }
}
