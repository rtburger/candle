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

    if qstorage.dtype() != GgmlDType::Q4K || rhs.dtype() != DType::F32 {
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
    if b_size != 1 || k != ncols {
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
    let q8_blocks_per_row = ncols_padded / Q8_1_BLOCK_SIZE;
    let scratch_bytes = q8_blocks_per_row * Q8_1_BLOCK_BYTES;
    let mut scratch = unsafe { dev.alloc::<u8>(scratch_bytes)? };
    let mut out = unsafe { dev.alloc::<f32>(nrows)? };

    {
        let (scratch_ptr, scratch_write) = scratch.device_ptr_mut(&stream);
        unsafe {
            ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                rhs_ptr,
                scratch_ptr as *mut std::ffi::c_void,
                ncols as i32,
                ncols_padded as i32,
                1,
                stream_ptr,
            );
        }

        let (q4_ptr, q4_read) = qstorage.device_ptr_with_guard(&stream)?;
        let (out_ptr, out_write) = out.device_ptr_mut(&stream);

        let q4_cutile = unsafe {
            DevicePointer::<u8>::from_cu_deviceptr(q4_ptr as cutile::cuda_core::sys::CUdeviceptr)
        };
        let q8_cutile = unsafe {
            DevicePointer::<u8>::from_cu_deviceptr(
                scratch_ptr as cutile::cuda_core::sys::CUdeviceptr,
            )
        };
        let out_cutile = unsafe {
            DevicePointer::<f32>::from_cu_deviceptr(out_ptr as cutile::cuda_core::sys::CUdeviceptr)
        };

        {
            let (_cutile_device, cutile_stream) = borrow_candle_cuda_handles(dev)?;
            let op = unsafe {
                q4k_q8_1_matvec_kernel::q4k_q8_1_matvec_b1_f32(
                    q4_cutile,
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

        drop(out_write);
        drop(q4_read);
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
    fn cutile_q4k_q8_1_b1_matches_candle_cuda_matvec() {
        let _env_guard = QUANT_KERNEL_ENV_LOCK.lock().unwrap();
        std::thread::Builder::new()
            .name("pi-ai-candle-worker".to_string())
            .spawn(|| {
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
        let qtensor = quantized::QTensor::quantize(&weights, GgmlDType::Q4K).unwrap();
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
                        "unexpected mismatch at index {index}: candle={expected} cutile={actual} abs_err={abs_err} rel_err={rel_err}"
                    );
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
