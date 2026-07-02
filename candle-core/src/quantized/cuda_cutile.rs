//! Experimental CUDA/cuTile interop for Candle quantized kernels.
//!
//! This module lives inside `candle-core` rather than `pi-ai-candle` because the
//! useful backend seam is in Candle's private quantized CUDA path. Candle stays
//! the owner of the CUDA device, context, stream, allocations, synchronization,
//! and fallback policy; cuTile only borrows Candle's raw handles and receives
//! Candle-owned device pointers.

use crate::{backend::BackendDevice, CudaDevice, Result};

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
    use crate::Device;

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
}
