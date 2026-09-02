#pragma once

// FlashAttention's forward-only kernel parameter ABI carries a Philox state
// even when dropout is compiled out.  Diffusion Compiler never enables
// dropout in inference attention, so the native build needs only an inert
// type with no framework dependency.
namespace at {
struct PhiloxCudaState {};
} // namespace at
