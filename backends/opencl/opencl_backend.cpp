#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include "dif/backend/abi.h"
#include "dif/compiler/memory_plan.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

void set_error(dif_backend_error *error, const std::string &message) {
  if (error && error->data && error->capacity)
    std::snprintf(error->data, error->capacity, "%s", message.c_str());
}

void check(cl_int status, const char *operation) {
  if (status != CL_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed with OpenCL " +
                             std::to_string(status));
}

std::string kernel_name(const dif::ir::Operation &operation) {
  return "dif_op_" + std::to_string(operation.id);
}

bool graph_float(dif::ir::DType dtype) {
  return dtype == dif::ir::DType::F32 || dtype == dif::ir::DType::BF16;
}

void emit_header(std::ostringstream &source) {
  source << R"CLC(
typedef float dif_f32;
typedef ushort dif_bf16;
inline float dif_load_f32(__global const dif_f32* value, ulong index) { return value[index]; }
inline void dif_store_f32(__global dif_f32* value, ulong index, float input) { value[index] = input; }
inline float dif_round_f32(float input) { return input; }
inline float dif_load_bf16(__global const dif_bf16* value, ulong index) {
  return as_float(((uint)value[index]) << 16U);
}
inline void dif_store_bf16(__global dif_bf16* value, ulong index, float input) {
  uint bits = as_uint(input);
  uint rounding = 0x7fffU + ((bits >> 16U) & 1U);
  value[index] = (ushort)((bits + rounding) >> 16U);
}
inline float dif_round_bf16(float input) {
  uint bits = as_uint(input);
  uint rounding = 0x7fffU + ((bits >> 16U) & 1U);
  return as_float(((bits + rounding) >> 16U) << 16U);
}
inline float dif_silu(float input) { return input / (1.0f + exp(-input)); }
)CLC";
}

dif::ir::DType operation_float_dtype(const dif::ir::Program &program,
                                     const dif::ir::Operation &operation) {
  for (const auto &ids : {&operation.outputs, &operation.inputs}) {
    for (const auto id : *ids) {
      const auto dtype = program.tensor(id)->dtype;
      if (graph_float(dtype))
        return dtype;
      if (dtype == dif::ir::DType::F16)
        throw std::runtime_error(
            "OpenCL backend does not yet admit f16 operations");
    }
  }
  throw std::runtime_error("OpenCL operation has no supported float tensor");
}

void begin_float_operation(std::ostringstream &source,
                           dif::ir::DType dtype) {
  if (dtype == dif::ir::DType::F32) {
    source << "#define dif_scalar dif_f32\n#define dif_load dif_load_f32\n"
              "#define dif_store dif_store_f32\n#define dif_round dif_round_f32\n";
    return;
  }
  if (dtype == dif::ir::DType::BF16) {
    source << "#define dif_scalar dif_bf16\n#define dif_load dif_load_bf16\n"
              "#define dif_store dif_store_bf16\n#define dif_round dif_round_bf16\n";
    return;
  }
  throw std::runtime_error("OpenCL backend admits f32 or bf16 operations");
}

void end_float_operation(std::ostringstream &source) {
  source << "#undef dif_scalar\n#undef dif_load\n#undef dif_store\n"
            "#undef dif_round\n";
}

void emit_cast(std::ostringstream &source, const dif::ir::Program &program,
               const dif::ir::Operation &operation) {
  const auto *input = program.tensor(operation.inputs[0]);
  const auto *output = program.tensor(operation.outputs[0]);
  const auto type = [](dif::ir::DType dtype) -> const char * {
    if (dtype == dif::ir::DType::F32)
      return "dif_f32";
    if (dtype == dif::ir::DType::BF16)
      return "dif_bf16";
    throw std::runtime_error("OpenCL Cast admits f32 and bf16 storage");
  };
  const auto load = input->dtype == dif::ir::DType::F32 ? "dif_load_f32"
                                                         : "dif_load_bf16";
  const auto store = output->dtype == dif::ir::DType::F32 ? "dif_store_f32"
                                                           : "dif_store_bf16";
  source << "__kernel void " << kernel_name(operation) << "(__global const "
         << type(input->dtype) << "* x,__global " << type(output->dtype)
         << "* y){ulong i=get_global_id(0);if(i<" << output->element_count()
         << "UL)" << store << "(y,i," << load << "(x,i));}\n";
}

void emit_elementwise(std::ostringstream &source,
                      const dif::ir::Program &program,
                      const dif::ir::Operation &operation,
                      const char *expression) {
  const auto count = program.tensor(operation.outputs[0])->element_count();
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* a,__global const dif_scalar* b,"
            "__global dif_scalar* y){ulong i=get_global_id(0);if(i<"
         << count << "UL)dif_store(y,i," << expression << ");}\n";
}

void emit_affine_last_dim(std::ostringstream &source,
                          const dif::ir::Program &program,
                          const dif::ir::Operation &operation) {
  const auto count = program.tensor(operation.outputs[0])->element_count();
  const auto width = program.tensor(operation.inputs[1])->dims[0];
  source << "__kernel void " << kernel_name(operation)
         << (operation.inputs.size() == 3U
                 ? "(__global const dif_scalar* x,__global const dif_scalar* scale,__global const dif_scalar* bias,__global dif_scalar* y){"
                 : "(__global const dif_scalar* x,__global const dif_scalar* scale,__global dif_scalar* y){")
         << "ulong i=get_global_id(0);if(i<" << count
         << "UL){ulong col=i%" << width
         << "UL;float value=dif_round(dif_load(x,i)*dif_load(scale,col));"
         << (operation.inputs.size() == 3U
                 ? "value=dif_round(value+dif_load(bias,col));"
                 : "")
         << "dif_store(y,i,value);}}\n";
}

void emit_clamp(std::ostringstream &source, const dif::ir::Program &program,
                const dif::ir::Operation &operation) {
  const auto count = program.tensor(operation.outputs[0])->element_count();
  const auto lower = static_cast<float>(
      operation.f64(dif::ir::AttrKey::Lower, 0.0));
  const auto upper = static_cast<float>(
      operation.f64(dif::ir::AttrKey::Upper, 1.0));
  source << std::scientific << std::setprecision(9) << "__kernel void "
         << kernel_name(operation)
         << "(__global const dif_scalar* x,__global dif_scalar* y){ulong i="
            "get_global_id(0);if(i<"
         << count << "UL)dif_store(y,i,fmin(" << upper << "f,fmax(" << lower
         << "f,dif_load(x,i))));}\n" << std::defaultfloat;
}

void emit_silu(std::ostringstream &source, const dif::ir::Program &program,
               const dif::ir::Operation &operation) {
  const auto count = program.tensor(operation.outputs[0])->element_count();
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* x,__global dif_scalar* y){ulong i="
            "get_global_id(0);if(i<"
         << count << "UL)dif_store(y,i,dif_silu(dif_load(x,i)));}\n";
}

void emit_mse_loss(std::ostringstream &source,
                   const dif::ir::Program &program,
                   const dif::ir::Operation &operation) {
  const auto count = program.tensor(operation.inputs[0])->element_count();
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* prediction,__global const dif_scalar* target,__global dif_scalar* loss){"
            "if(get_global_id(0)==0UL){float sum=0.0f;for(ulong i=0UL;i<"
         << count
         << "UL;++i){float d=dif_load(prediction,i)-dif_load(target,i);sum=fma(d,d,sum);}"
            "dif_store(loss,0UL,sum/"
         << count << ".0f);}}\n";
}

void emit_mse_loss_backward(std::ostringstream &source,
                            const dif::ir::Program &program,
                            const dif::ir::Operation &operation) {
  const auto count = program.tensor(operation.outputs[0])->element_count();
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* prediction,__global const dif_scalar* target,__global const dif_scalar* grad_loss,__global dif_scalar* grad_prediction){"
            "ulong i=get_global_id(0);if(i<"
         << count
         << "UL){float factor=2.0f*dif_load(grad_loss,0UL)/" << count
         << ".0f;dif_store(grad_prediction,i,(dif_load(prediction,i)-dif_load(target,i))*factor);}}\n";
}

void emit_linear_backward_input(std::ostringstream &source,
                                const dif::ir::Program &program,
                                const dif::ir::Operation &operation) {
  const auto *weight = program.tensor(operation.inputs[1]);
  const auto count = program.tensor(operation.outputs[0])->element_count();
  const auto inner = weight->dims[1];
  const auto outputs = weight->dims[0];
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* grad_output,__global const dif_scalar* weight,__global dif_scalar* grad_input){"
            "ulong i=get_global_id(0);if(i<"
         << count << "UL){ulong row=i/" << inner << "UL,column=i%" << inner
         << "UL;float value=0.0f;for(ulong output=0UL;output<" << outputs
         << "UL;++output)value=fma(dif_load(grad_output,row*" << outputs
         << "UL+output),dif_load(weight,output*" << inner
         << "UL+column),value);dif_store(grad_input,i,value);}}\n";
}

void emit_linear_backward_weight(std::ostringstream &source,
                                 const dif::ir::Program &program,
                                 const dif::ir::Operation &operation) {
  const auto *input = program.tensor(operation.inputs[1]);
  const auto *grad_output = program.tensor(operation.inputs[0]);
  const auto count = program.tensor(operation.outputs[0])->element_count();
  const auto inner = input->dims.back();
  const auto rows = input->element_count() / inner;
  const auto outputs = grad_output->dims.back();
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* grad_output,__global const dif_scalar* input,__global dif_scalar* grad_weight){"
            "ulong i=get_global_id(0);if(i<"
         << count << "UL){ulong output=i/" << inner << "UL,column=i%" << inner
         << "UL;float value=0.0f;for(ulong row=0UL;row<" << rows
         << "UL;++row)value=fma(dif_load(grad_output,row*" << outputs
         << "UL+output),dif_load(input,row*" << inner
         << "UL+column),value);dif_store(grad_weight,i,value);}}\n";
}

void emit_bias_backward(std::ostringstream &source,
                        const dif::ir::Program &program,
                        const dif::ir::Operation &operation) {
  const auto *grad_output = program.tensor(operation.inputs[0]);
  const auto width = program.tensor(operation.outputs[0])->dims[0];
  const auto rows = grad_output->element_count() / width;
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* grad_output,__global dif_scalar* grad_bias){"
            "ulong column=get_global_id(0);if(column<"
         << width << "UL){float value=0.0f;for(ulong row=0UL;row<" << rows
         << "UL;++row)value+=dif_load(grad_output,row*" << width
         << "UL+column);dif_store(grad_bias,column,value);}}\n";
}

void emit_silu_backward(std::ostringstream &source,
                        const dif::ir::Program &program,
                        const dif::ir::Operation &operation) {
  const auto count = program.tensor(operation.outputs[0])->element_count();
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* x,__global const dif_scalar* grad_output,__global dif_scalar* grad_input){"
            "ulong i=get_global_id(0);if(i<"
         << count
         << "UL){float value=dif_load(x,i),sigmoid=1.0f/(1.0f+exp(-value));"
            "float derivative=sigmoid*(1.0f+value*(1.0f-sigmoid));"
            "dif_store(grad_input,i,dif_load(grad_output,i)*derivative);}}\n";
}

void emit_adamw_update(std::ostringstream &source,
                       const dif::ir::Program &program,
                       const dif::ir::Operation &operation) {
  const auto count = program.tensor(operation.outputs[0])->element_count();
  const auto learning_rate = static_cast<float>(
      operation.f64(dif::ir::AttrKey::LearningRate, 1.0e-3));
  const auto beta1 =
      static_cast<float>(operation.f64(dif::ir::AttrKey::Beta1, 0.9));
  const auto beta2 =
      static_cast<float>(operation.f64(dif::ir::AttrKey::Beta2, 0.999));
  const auto epsilon =
      static_cast<float>(operation.f64(dif::ir::AttrKey::Epsilon, 1.0e-8));
  const auto weight_decay = static_cast<float>(
      operation.f64(dif::ir::AttrKey::WeightDecay, 0.0));
  source << std::scientific << std::setprecision(9) << "__kernel void "
         << kernel_name(operation)
         << "(__global const dif_scalar* parameter,__global const dif_scalar* gradient,__global const dif_scalar* first,__global const dif_scalar* second,__global const int* completed_steps,__global dif_scalar* updated,__global dif_scalar* updated_first,__global dif_scalar* updated_second){"
            "ulong i=get_global_id(0);if(i<"
         << count << "UL){float step=(float)(completed_steps[0]+1),beta1="
         << beta1 << "f,beta2=" << beta2
         << "f;float grad=dif_load(gradient,i);float m=beta1*dif_load(first,i)+(1.0f-beta1)*grad;"
            "float v=beta2*dif_load(second,i)+(1.0f-beta2)*grad*grad;float bias1=1.0f-pow(beta1,step);"
            "float bias2_sqrt=sqrt(1.0f-pow(beta2,step));float decayed=dif_load(parameter,i)*(1.0f-"
         << learning_rate << "f*" << weight_decay
         << "f);float denominator=sqrt(v)/bias2_sqrt+" << epsilon
         << "f;float value=decayed-(" << learning_rate
         << "f/bias1)*m/denominator;dif_store(updated,i,value);dif_store(updated_first,i,m);"
            "dif_store(updated_second,i,v);}}\n"
         << std::defaultfloat;
}

void emit_fill(std::ostringstream &source, const dif::ir::Program &program,
               const dif::ir::Operation &operation) {
  const auto count = program.tensor(operation.outputs[0])->element_count();
  const auto value = static_cast<float>(
      operation.f64(dif::ir::AttrKey::Value, 0.0));
  source << std::setprecision(17) << "__kernel void "
         << kernel_name(operation)
         << "(__global dif_scalar* y){ulong i=get_global_id(0);if(i<"
         << count << "UL)dif_store(y,i," << std::scientific
         << std::setprecision(9) << value << "f);}\n" << std::defaultfloat;
}

void emit_gather_rows(std::ostringstream &source,
                      const dif::ir::Program &program,
                      const dif::ir::Operation &operation) {
  const auto *input = program.tensor(operation.inputs[0]);
  const auto count = program.tensor(operation.outputs[0])->element_count();
  const auto input_rows = input->dims[0];
  const auto row_width = input->element_count() / input_rows;
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* x,__global const int* indices,"
            "__global dif_scalar* y){ulong i=get_global_id(0);if(i<"
         << count << "UL){ulong row=i/" << row_width << "UL,col=i%"
         << row_width << "UL;int from=indices[row];if(from>=0&&from<"
         << input_rows
         << ")dif_store(y,i,dif_load(x,(ulong)from*" << row_width
         << "UL+col));else dif_store(y,i,as_float(0x7fffffffU));}}\n";
}

void emit_indexed_update_rows(std::ostringstream &source,
                              const dif::ir::Program &program,
                              const dif::ir::Operation &operation) {
  const auto *base = program.tensor(operation.inputs[0]);
  const auto *updates = program.tensor(operation.inputs[1]);
  const auto count = program.tensor(operation.outputs[0])->element_count();
  const auto rows = base->dims[0];
  const auto update_rows = updates->dims[0];
  const auto row_width = base->element_count() / rows;
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* base,__global const dif_scalar* "
            "updates,__global const int* map,__global dif_scalar* y){ulong i="
            "get_global_id(0);if(i<"
         << count << "UL){ulong row=i/" << row_width << "UL,col=i%"
         << row_width
         << "UL;int from=map[row];if(from==-1)dif_store(y,i,dif_load(base,i));"
            "else if(from>=0&&from<"
         << update_rows
         << ")dif_store(y,i,dif_load(updates,(ulong)from*" << row_width
         << "UL+col));else dif_store(y,i,as_float(0x7fffffffU));}}\n";
}

void emit_select_row_chunks(std::ostringstream &source,
                            const dif::ir::Program &program,
                            const dif::ir::Operation &operation) {
  const auto *values = program.tensor(operation.inputs[0]);
  const auto *output = program.tensor(operation.outputs[0]);
  const auto count = output->element_count();
  const auto rows = values->dims[0];
  const auto width = output->dims[1];
  const auto source_width = values->dims[1];
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* values,__global const int* indices";
  for (std::size_t chunk = 0; chunk < operation.outputs.size(); ++chunk)
    source << ",__global dif_scalar* o" << chunk;
  source << "){ulong i=get_global_id(0);if(i<" << count
         << "UL){ulong row=i/" << width << "UL,col=i%" << width
         << "UL;int from=indices[row];";
  for (std::size_t chunk = 0; chunk < operation.outputs.size(); ++chunk) {
    source << "if(from>=0&&from<" << rows << ")dif_store(o" << chunk
           << ",i,dif_load(values,(ulong)from*" << source_width << "UL+"
           << chunk * width << "UL+col));else dif_store(o" << chunk
           << ",i,as_float(0x7fffffffU));";
  }
  source << "}}\n";
}

void emit_sinusoidal_timestep(std::ostringstream &source,
                              const dif::ir::Program &program,
                              const dif::ir::Operation &operation) {
  const auto *output = program.tensor(operation.outputs[0]);
  const auto count = output->element_count();
  const auto width = output->dims[1];
  const auto half = width / 2U;
  const auto flip =
      operation.boolean(dif::ir::AttrKey::FlipSinToCos, false);
  const auto shift = static_cast<float>(
      operation.f64(dif::ir::AttrKey::DownscaleFreqShift, 1.0));
  const auto scale =
      static_cast<float>(operation.f64(dif::ir::AttrKey::Scale, 1.0));
  const auto max_period =
      static_cast<float>(operation.f64(dif::ir::AttrKey::MaxPeriod, 10000.0));
  const auto log_period = static_cast<float>(std::log(max_period));
  const auto denominator = static_cast<float>(half) - shift;
  source << std::scientific << std::setprecision(9) << "__kernel void "
         << kernel_name(operation)
         << "(__global const dif_f32* timesteps,__global dif_f32* y){ulong i="
            "get_global_id(0);if(i<"
         << count << "UL){ulong row=i/" << width << "UL,column=i%" << width
         << "UL;if(column>=" << 2U * half
         << "UL){dif_store_f32(y,i,0.0f);return;}ulong component=column%"
         << half << "UL;float exponent=(-" << log_period
         << "f*(float)component)/" << denominator
         << "f;float frequency=exp(exponent);float angle=" << scale
         << "f*(dif_load_f32(timesteps,row)*frequency);float s=sin(angle),"
            "c=cos(angle);dif_store_f32(y,i,"
         << (flip ? "(column<" + std::to_string(half) + "UL?c:s)"
                  : "(column<" + std::to_string(half) + "UL?s:c)")
         << ");}}\n" << std::defaultfloat;
}

void emit_rotary_position(std::ostringstream &source,
                          const dif::ir::Program &program,
                          const dif::ir::Operation &operation) {
  const auto *positions = program.tensor(operation.inputs[0]);
  const auto *inv_freq = program.tensor(operation.inputs[1]);
  const auto *output = program.tensor(operation.outputs[0]);
  const auto count = output->element_count();
  const auto axes = positions->dims[1];
  const auto frequencies = inv_freq->dims[0];
  const auto unrepeated_width = axes * frequencies;
  const auto width = 2U * unrepeated_width;
  const auto output_type = output->dtype == dif::ir::DType::F32 ? "dif_f32"
                                                                 : "dif_bf16";
  const auto store = output->dtype == dif::ir::DType::F32 ? "dif_store_f32"
                                                           : "dif_store_bf16";
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_f32* positions,__global const dif_f32* "
            "inv_freq,__global "
         << output_type << "* cosine,__global " << output_type
         << "* sine){ulong i=get_global_id(0);if(i<" << count
         << "UL){ulong row=i/" << width << "UL,column=i%" << width
         << "UL,component=column%" << unrepeated_width
         << "UL,axis=component/" << frequencies
         << "UL,frequency=component%" << frequencies
         << "UL;float angle=dif_load_f32(positions,row*" << axes
         << "UL+axis)*dif_load_f32(inv_freq,frequency);" << store
         << "(cosine,i,cos(angle));" << store << "(sine,i,sin(angle));}}\n";
}

void emit_linear_blend(std::ostringstream &source,
                       const dif::ir::Program &program,
                       const dif::ir::Operation &operation) {
  const auto count = program.tensor(operation.outputs[0])->element_count();
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* left,__global const dif_scalar* right,"
            "__global const dif_f32* factor,__global dif_scalar* output){ulong "
            "i=get_global_id(0);if(i<"
         << count
         << "UL){float f=dif_load_f32(factor,0UL),left_value=dif_load(left,i),"
            "right_value=dif_load(right,i);volatile float complement=1.0f-f;"
            "volatile float weighted_left=f*left_value;volatile float "
            "weighted_right=complement*right_value;volatile float blended="
            "weighted_left+weighted_right;dif_store(output,i,blended);}}\n";
}

void emit_flow_euler_step(std::ostringstream &source,
                          const dif::ir::Program &program,
                          const dif::ir::Operation &operation) {
  const auto count = program.tensor(operation.outputs[0])->element_count();
  const auto step = operation.u64(dif::ir::AttrKey::StepIndex, 0U);
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* sample,__global const dif_scalar* "
            "velocity,__global const dif_f32* timesteps,__global const dif_f32* "
            "sigmas,__global dif_scalar* output){ulong i=get_global_id(0);if(i<"
         << count << "UL){float timestep=dif_load_f32(timesteps," << step
         << "UL),sigma=dif_load_f32(sigmas," << step
         << "UL),sigma_next=dif_load_f32(sigmas," << step + 1U
         << "UL),sample_value=dif_load(sample,i),velocity_value=dif_load(velocity,"
            "i),sigma_from_timestep=1.0f-timestep,ratio=sigma_next/sigma;volatile "
            "float velocity_delta=sigma_from_timestep*velocity_value;volatile "
            "float denoised=sample_value+velocity_delta;volatile float complement="
            "1.0f-ratio;volatile float weighted_sample=ratio*sample_value;volatile "
            "float weighted_denoised=complement*denoised;volatile float blended="
            "weighted_sample+weighted_denoised;dif_store(output,i,blended);}}\n";
}

void emit_patchify_3d(std::ostringstream &source,
                      const dif::ir::Program &program,
                      const dif::ir::Operation &operation, bool inverse) {
  const auto *volume = program.tensor(
      inverse ? operation.outputs[0] : operation.inputs[0]);
  const auto *rows = program.tensor(
      inverse ? operation.inputs[0] : operation.outputs[0]);
  const auto patch_t = operation.u64(dif::ir::AttrKey::PatchT, 0U);
  const auto patch_h = operation.u64(dif::ir::AttrKey::PatchH, 0U);
  const auto patch_w = operation.u64(dif::ir::AttrKey::PatchW, 0U);
  const auto channels = volume->dims[1];
  const auto frames = volume->dims[2];
  const auto height = volume->dims[3];
  const auto width = volume->dims[4];
  const auto output_frames = frames / patch_t;
  const auto output_height = height / patch_h;
  const auto output_width = width / patch_w;
  const auto count = rows->element_count();
  const auto row_width = rows->dims[1];
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* input,__global dif_scalar* output){"
            "ulong i=get_global_id(0);if(i<"
         << count << "UL){ulong row=i/" << row_width << "UL,column=i%"
         << row_width << "UL,outer=row,patch_x=outer%" << output_width
         << "UL;outer/=" << output_width << "UL;ulong patch_y=outer%"
         << output_height << "UL;outer/=" << output_height
         << "UL;ulong patch_frame=outer%" << output_frames
         << "UL,batch=outer/" << output_frames
         << "UL,inner=column,offset_x=inner%" << patch_w << "UL;inner/="
         << patch_w << "UL;ulong offset_y=inner%" << patch_h
         << "UL;inner/=" << patch_h << "UL;ulong offset_t=inner%" << patch_t
         << "UL;inner/=" << patch_t
         << "UL;ulong channel=inner,frame=patch_frame*" << patch_t
         << "UL+offset_t,y=patch_y*" << patch_h
         << "UL+offset_y,x=patch_x*" << patch_w
         << "UL+offset_x,volume_index=((((batch*" << channels
         << "UL+channel)*" << frames << "UL+frame)*" << height
         << "UL+y)*" << width << "UL+x);"
         << (inverse
                 ? "dif_store(output,volume_index,dif_load(input,i));"
                 : "dif_store(output,i,dif_load(input,volume_index));")
         << "}}\n";
}

void emit_rms_norm(std::ostringstream &source,
                   const dif::ir::Program &program,
                   const dif::ir::Operation &operation) {
  const auto *input = program.tensor(operation.inputs[0]);
  const auto columns = input->dims.back();
  const auto rows = input->element_count() / columns;
  const auto epsilon = operation.f64(dif::ir::AttrKey::Epsilon, 1.0e-5);
  source << std::setprecision(17) << "__kernel void "
         << kernel_name(operation)
         << "(__global const dif_scalar* x,__global const dif_scalar* weight,"
            "__global dif_scalar* y){ulong row=get_global_id(0);if(row<"
         << rows << "UL){float sum=0.0f;for(ulong col=0;col<" << columns
         << "UL;++col){float v=dif_load(x,row*" << columns
         << "UL+col);sum=fma(v,v,sum);}float inv=rsqrt(sum/" << columns
         << ".0f+" << static_cast<float>(epsilon)
         << "f);for(ulong col=0;col<" << columns
         << "UL;++col){ulong i=row*" << columns
         << "UL+col;dif_store(y,i,dif_load(x,i)*inv*dif_load(weight,col));}}}\n";
}

void emit_layer_norm(std::ostringstream &source,
                     const dif::ir::Program &program,
                     const dif::ir::Operation &operation) {
  const auto *input = program.tensor(operation.inputs[0]);
  const auto columns = input->dims.back();
  const auto rows = input->element_count() / columns;
  const auto epsilon = operation.f64(dif::ir::AttrKey::Epsilon, 1.0e-5);
  source << std::setprecision(17) << "__kernel void "
         << kernel_name(operation)
         << "(__global const dif_scalar* x,__global const dif_scalar* weight,"
            "__global const dif_scalar* bias,__global dif_scalar* y){ulong "
            "row=get_global_id(0);if(row<"
         << rows << "UL){float mean=0.0f;for(ulong col=0;col<" << columns
         << "UL;++col)mean+=dif_load(x,row*" << columns << "UL+col);mean/="
         << columns << ".0f;float variance=0.0f;for(ulong col=0;col<"
         << columns << "UL;++col){float centered=dif_load(x,row*" << columns
         << "UL+col)-mean;variance+=centered*centered;}float inv=rsqrt(variance/"
         << columns << ".0f+" << static_cast<float>(epsilon)
         << "f);for(ulong col=0;col<" << columns
         << "UL;++col){ulong i=row*" << columns
         << "UL+col;dif_store(y,i,(dif_load(x,i)-mean)*inv*dif_load(weight,col)+"
            "dif_load(bias,col));}}}\n";
}

void emit_rms(std::ostringstream &source, const dif::ir::Program &program,
              const dif::ir::Operation &operation) {
  const auto &dims = program.tensor(operation.inputs[0])->dims;
  const auto rows = dims[0];
  const auto columns = dims[1];
  const auto epsilon = operation.f64(dif::ir::AttrKey::Epsilon, 1.0e-5);
  const bool weighted = operation.inputs.size() == 4U;
  source << std::setprecision(17) << "__kernel void "
         << kernel_name(operation)
         << (weighted
                 ? "(__global const dif_scalar* x,__global const dif_scalar* weight,__global const dif_scalar* scale,__global const dif_scalar* shift,__global dif_scalar* y){"
                 : "(__global const dif_scalar* x,__global const dif_scalar* scale,__global const dif_scalar* shift,__global dif_scalar* y){")
         << "ulong row=get_global_id(0);if(row<" << rows
         << "UL){float sum=0.0f;for(ulong col=0;col<" << columns
         << "UL;++col){float v=dif_load(x,row*" << columns
         << "UL+col);sum=fma(v,v,sum);}float inv=rsqrt(sum/" << columns
         << ".0f+" << static_cast<float>(epsilon)
         << "f);for(ulong col=0;col<" << columns
         << "UL;++col){ulong i=row*" << columns
         << "UL+col;float value=dif_load(x,i)*inv"
         << (weighted ? "*dif_load(weight,col)" : "")
         << ";value=dif_round(value);float modulation=dif_round(1.0f+"
            "dif_load(scale,i));value=dif_round(value*modulation);"
            "dif_store(y,i,value+dif_load(shift,i));}}}\n";
}

void emit_swiglu(std::ostringstream &source, const dif::ir::Program &program,
                 const dif::ir::Operation &operation) {
  const auto count = program.tensor(operation.outputs[0])->element_count();
  const auto width = program.tensor(operation.outputs[0])->dims.back();
  const bool gate_first =
      operation.boolean(dif::ir::AttrKey::GateFirst, false);
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* x,__global dif_scalar* y){ulong i="
            "get_global_id(0);if(i<"
         << count << "UL){ulong row=i/" << width << "UL,col=i%" << width
         << "UL;float value=dif_load(x,row*" << width * 2U << "UL+"
         << (gate_first ? width : 0U)
         << "UL+col);float gate=dif_load(x,row*" << width * 2U << "UL+"
         << (gate_first ? 0U : width)
         << "UL+col);dif_store(y,i,value*dif_round(dif_silu(gate)));}}\n";
}

void emit_residual(std::ostringstream &source,
                   const dif::ir::Program &program,
                   const dif::ir::Operation &operation) {
  const auto count = program.tensor(operation.outputs[0])->element_count();
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* residual,__global const dif_scalar* "
            "branch,__global const dif_scalar* gate,__global dif_scalar* y){"
            "ulong i=get_global_id(0);if(i<"
         << count
         << "UL)dif_store(y,i,dif_load(residual,i)+dif_round(dif_load(gate,i)*"
            "dif_load(branch,i)));}\n";
}

void emit_linear(std::ostringstream &source, const dif::ir::Program &program,
                 const dif::ir::Operation &operation) {
  const auto *input = program.tensor(operation.inputs[0]);
  const auto *weight = program.tensor(operation.inputs[1]);
  const auto rows = input->element_count() / input->dims[0] == 0U
                        ? 0U
                        : input->dims[0];
  const auto inner = input->element_count() / rows;
  const auto outputs = weight->dims[0];
  const auto count = rows * outputs;
  const bool biased = operation.inputs.size() == 3U;
  source << "__kernel void " << kernel_name(operation)
         << (biased
                 ? "(__global const dif_scalar* x,__global const dif_scalar* weight,__global const dif_scalar* bias,__global dif_scalar* y){"
                 : "(__global const dif_scalar* x,__global const dif_scalar* weight,__global dif_scalar* y){")
         << "ulong i=get_global_id(0);if(i<" << count << "UL){ulong row=i/"
         << outputs << "UL,column=i%" << outputs << "UL;float acc="
         << (biased ? "dif_load(bias,column)" : "0.0f")
         << ";for(ulong k=0;k<" << inner
         << "UL;++k)acc=fma(dif_load(x,row*" << inner
         << "UL+k),dif_load(weight,column*" << inner
         << "UL+k),acc);dif_store(y,i,acc);}}\n";
}

void emit_qk_norm(std::ostringstream &source,
                  const dif::ir::Program &program,
                  const dif::ir::Operation &operation) {
  const auto &dims = program.tensor(operation.inputs[0])->dims;
  const auto sequence = dims[0];
  const auto heads = dims[1];
  const auto dim = dims[2];
  const auto rotary =
      operation.u64(dif::ir::AttrKey::RotaryDim, dim);
  const auto half = rotary / 2U;
  const auto table_width = program.tensor(operation.inputs[2])->dims[1];
  const auto epsilon = operation.f64(dif::ir::AttrKey::Epsilon, 1.0e-5);
  source << std::setprecision(17) << "__kernel void "
         << kernel_name(operation)
         << "(__global const dif_scalar* x,__global const dif_scalar* weight,"
            "__global const dif_scalar* cosv,__global const dif_scalar* sinv,"
            "__global dif_scalar* y){ulong item=get_global_id(0);if(item<"
         << sequence * heads << "UL){ulong s=item/" << heads
         << "UL,h=item%" << heads << "UL,base=(s*" << heads << "UL+h)*"
         << dim << "UL;float sum=0.0f;for(ulong d=0;d<" << dim
         << "UL;++d){float v=dif_load(x,base+d);sum=fma(v,v,sum);}float inv="
            "rsqrt(sum/"
         << dim << ".0f+" << static_cast<float>(epsilon)
         << "f);for(ulong d=0;d<" << dim
         << "UL;++d){float value=dif_round(dif_load(x,base+d)*inv*dif_load("
            "weight,d));float result=value;if(d<"
         << half << "UL){float other=dif_round(dif_load(x,base+d+" << half
         << "UL)*inv*dif_load(weight,d+" << half
         << "UL));result=dif_round(dif_round(value*dif_load(cosv,s*"
         << table_width
         << "UL+d))-dif_round(other*dif_load(sinv,s*" << table_width
         << "UL+d)));}else if(d<" << rotary << "UL){ulong r=d-" << half
         << "UL;float other=dif_round(dif_load(x,base+r)*inv*dif_load(weight,r));"
            "ulong ti="
         << (table_width == rotary ? "d" : "r")
         << ";result=dif_round(dif_round(value*dif_load(cosv,s*"
         << table_width
         << "UL+ti))+dif_round(other*dif_load(sinv,s*" << table_width
         << "UL+ti)));}dif_store(y,base+d,result);}}}\n";
}

void emit_attention(std::ostringstream &source,
                    const dif::ir::Program &program,
                    const dif::ir::Operation &operation) {
  const auto &dims = program.tensor(operation.inputs[0])->dims;
  const auto sequence = dims[0];
  const auto heads = dims[1];
  const auto dim = dims[2];
  if (sequence > 256U)
    throw std::runtime_error(
        "portable OpenCL reference attention currently admits S<=256");
  const auto scale = operation.f64(
      dif::ir::AttrKey::AttentionScale,
      1.0 / std::sqrt(static_cast<double>(dim)));
  const bool causal =
      operation.boolean(dif::ir::AttrKey::Causal, false);
  source << std::setprecision(17) << "__kernel void "
         << kernel_name(operation)
         << "(__global const dif_scalar* q,__global const dif_scalar* k,"
            "__global const dif_scalar* v,__global dif_scalar* y){ulong item="
            "get_global_id(0);if(item<"
         << sequence * heads << "UL){ulong qs=item/" << heads
         << "UL,h=item%" << heads << "UL,kend="
         << (causal ? "qs+1UL" : std::to_string(sequence) + "UL")
         << ";float probabilities[" << sequence
         << "];float maximum=-3.402823466e+38f;for(ulong ks=0;ks<kend;++ks){"
            "float score=0.0f;for(ulong d=0;d<"
         << dim << "UL;++d)score=fma(dif_load(q,(qs*" << heads
         << "UL+h)*" << dim << "UL+d),dif_load(k,(ks*" << heads
         << "UL+h)*" << dim << "UL+d),score);score*="
         << static_cast<float>(scale)
         << "f;probabilities[ks]=score;maximum=fmax(maximum,score);}float denom="
            "0.0f;for(ulong ks=0;ks<kend;++ks){probabilities[ks]=exp("
            "probabilities[ks]-maximum);denom+=probabilities[ks];}for(ulong d=0;"
            "d<"
         << dim
         << "UL;++d){float acc=0.0f;for(ulong ks=0;ks<kend;++ks)acc=fma("
            "probabilities[ks]/denom,dif_load(v,(ks*"
         << heads << "UL+h)*" << dim << "UL+d),acc);dif_store(y,(qs*"
         << heads << "UL+h)*" << dim << "UL+d,acc);}}}\n";
}

void emit_bias(std::ostringstream &source, const dif::ir::Program &program,
               const dif::ir::Operation &operation) {
  const auto count = program.tensor(operation.outputs[0])->element_count();
  const auto width = program.tensor(operation.inputs[1])->dims[0];
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* x,__global const dif_scalar* bias,"
            "__global dif_scalar* y){ulong i=get_global_id(0);if(i<"
         << count << "UL)dif_store(y,i,dif_load(x,i)+dif_load(bias,i%" << width
         << "UL));}\n";
}

void emit_adaln_select(std::ostringstream &source,
                       const dif::ir::Program &program,
                       const dif::ir::Operation &operation) {
  const auto &dims = program.tensor(operation.outputs[0])->dims;
  const auto count = program.tensor(operation.outputs[0])->element_count();
  const auto hidden = dims[1];
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* projected,__global const int* indices,"
            "__global dif_scalar* o0,__global dif_scalar* o1,__global dif_scalar* "
            "o2,__global dif_scalar* o3,__global dif_scalar* o4,__global "
            "dif_scalar* o5){ulong i=get_global_id(0);if(i<"
         << count << "UL){ulong row=i/" << hidden << "UL,col=i%" << hidden
         << "UL,table=(ulong)indices[row];"
            "dif_store(o0,i,dif_load(projected,(table*6UL+0UL)*"
         << hidden
         << "UL+col));dif_store(o1,i,dif_load(projected,(table*6UL+1UL)*"
         << hidden
         << "UL+col));dif_store(o2,i,dif_load(projected,(table*6UL+2UL)*"
         << hidden
         << "UL+col));dif_store(o3,i,dif_load(projected,(table*6UL+3UL)*"
         << hidden
         << "UL+col));dif_store(o4,i,dif_load(projected,(table*6UL+4UL)*"
         << hidden
         << "UL+col));dif_store(o5,i,dif_load(projected,(table*6UL+5UL)*"
         << hidden << "UL+col));}}\n";
}

void emit_deinterleave(std::ostringstream &source,
                       const dif::ir::Program &program,
                       const dif::ir::Operation &operation) {
  const auto &dims = program.tensor(operation.outputs[0])->dims;
  const auto count = program.tensor(operation.outputs[0])->element_count();
  const auto heads = dims[1];
  const auto dim = dims[2];
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* x,__global dif_scalar* q,__global "
            "dif_scalar* k,__global dif_scalar* v){ulong i=get_global_id(0);"
            "if(i<"
         << count << "UL){ulong row=i/" << heads * dim << "UL,within=i%"
         << heads * dim << "UL,head=within/" << dim << "UL,d=within%" << dim
         << "UL,base=row*" << 3U * heads * dim << "UL+head*" << 3U * dim
         << "UL+d;dif_store(q,i,dif_load(x,base));dif_store(k,i,dif_load(x,"
            "base+"
         << dim << "UL));dif_store(v,i,dif_load(x,base+" << 2U * dim
         << "UL));}}\n";
}

void emit_deinterleave_weight(std::ostringstream &source,
                              const dif::ir::Program &program,
                              const dif::ir::Operation &operation) {
  const auto &dims = program.tensor(operation.outputs[0])->dims;
  const auto count = program.tensor(operation.outputs[0])->element_count();
  const auto dim = operation.u64(dif::ir::AttrKey::HeadDim, 0U);
  const auto hidden = dims[1];
  source << "__kernel void " << kernel_name(operation)
         << "(__global const dif_scalar* packed,__global dif_scalar* q,__global "
            "dif_scalar* k,__global dif_scalar* v){ulong i=get_global_id(0);"
            "if(i<"
         << count << "UL){ulong row=i/" << hidden << "UL,col=i%" << hidden
         << "UL,head=row/" << dim << "UL,d=row%" << dim
         << "UL,base=((head*3UL)*" << dim << "UL+d)*" << hidden
         << "UL+col;dif_store(q,i,dif_load(packed,base));dif_store(k,i,dif_load("
            "packed,base+"
         << dim * hidden
         << "UL));dif_store(v,i,dif_load(packed,base+" << 2U * dim * hidden
         << "UL));}}\n";
}

void emit_dequantize(std::ostringstream &source,
                     const dif::ir::Program &program,
                     const dif::ir::Operation &operation,
                     std::uint32_t bits) {
  const auto &dims = program.tensor(operation.outputs[0])->dims;
  const auto rows = dims[0];
  const auto columns = dims[1];
  const auto row_bytes = columns * bits / 8U;
  const auto group =
      operation.u64(dif::ir::AttrKey::GroupSize, 64U);
  const auto groups = columns / group;
  const auto count = rows * columns;
  const bool column_scaled = operation.inputs.size() == 3U;
  const bool corrected = operation.inputs.size() == 4U;
  source << "__kernel void " << kernel_name(operation)
         << (column_scaled
                 ? "(__global const uchar* packed,__global const dif_scalar* scales,__global const dif_scalar* column_scales,__global dif_scalar* y){"
                 : corrected
                 ? "(__global const uchar* packed,__global const dif_scalar* scales,__global const uchar* outlier_indices,__global const dif_scalar* outlier_residuals,__global dif_scalar* y){"
                 : "(__global const uchar* packed,__global const dif_scalar* scales,__global dif_scalar* y){")
         << "ulong i=get_global_id(0);if(i<" << count << "UL){ulong row=i/"
         << columns << "UL,col=i%" << columns << "UL,bit=col*" << bits
         << "UL,bi=row*" << row_bytes
         << "UL+bit/8UL;uint shift=(uint)(bit&7UL),word=packed[bi];if(shift+"
         << bits << "U>8U)word|=((uint)packed[bi+1UL])<<8U;uint encoded=(word>>"
            "shift)&"
         << ((1U << bits) - 1U) << "U;int q=encoded<" << (1U << (bits - 1U))
         << "U?(int)encoded:(int)encoded-" << (1U << bits)
         << ";ulong gi=row*" << groups << "UL+col/" << group
         << "UL;float value=(float)q*dif_load(scales,gi);"
         << (corrected
                 ? "if(outlier_indices[gi]==col%" + std::to_string(group) +
                       "UL)value+=dif_load(outlier_residuals,gi);"
                 : "")
         << (column_scaled ? "value*=dif_load(column_scales,col);" : "")
         << "dif_store(y,i,value);}}\n";
}

std::string generate_source(const dif::ir::Program &program) {
  for (const auto &tensor : program.tensors) {
    if (!graph_float(tensor.dtype) && tensor.dtype != dif::ir::DType::I8 &&
        tensor.dtype != dif::ir::DType::I32)
      throw std::runtime_error("OpenCL backend encountered an unsupported dtype");
  }
  std::ostringstream source;
  emit_header(source);
  for (const auto &operation : program.operations) {
    using dif::ir::Opcode;
    if (operation.opcode == Opcode::Barrier)
      continue;
    if (operation.opcode == Opcode::Cast) {
      emit_cast(source, program, operation);
      continue;
    }
    if (operation.opcode == Opcode::RotaryPosition) {
      emit_rotary_position(source, program, operation);
      continue;
    }
    begin_float_operation(source, operation_float_dtype(program, operation));
    switch (operation.opcode) {
    case Opcode::Add:
      emit_elementwise(source, program, operation,
                       "dif_load(a,i)+dif_load(b,i)");
      break;
    case Opcode::Multiply:
      emit_elementwise(source, program, operation,
                       "dif_load(a,i)*dif_load(b,i)");
      break;
    case Opcode::AffineLastDim:
      emit_affine_last_dim(source, program, operation);
      break;
    case Opcode::SiLU:
      emit_silu(source, program, operation);
      break;
    case Opcode::RmsNorm:
      emit_rms_norm(source, program, operation);
      break;
    case Opcode::LayerNorm:
      emit_layer_norm(source, program, operation);
      break;
    case Opcode::Clamp:
      emit_clamp(source, program, operation);
      break;
    case Opcode::MseLoss:
      emit_mse_loss(source, program, operation);
      break;
    case Opcode::MseLossBackward:
      emit_mse_loss_backward(source, program, operation);
      break;
    case Opcode::LinearBackwardInput:
      emit_linear_backward_input(source, program, operation);
      break;
    case Opcode::LinearBackwardWeight:
      emit_linear_backward_weight(source, program, operation);
      break;
    case Opcode::BiasBackward:
      emit_bias_backward(source, program, operation);
      break;
    case Opcode::SiLUBackward:
      emit_silu_backward(source, program, operation);
      break;
    case Opcode::AdamWUpdate:
      emit_adamw_update(source, program, operation);
      break;
    case Opcode::Fill:
      emit_fill(source, program, operation);
      break;
    case Opcode::GatherRows:
      emit_gather_rows(source, program, operation);
      break;
    case Opcode::IndexedUpdateRows:
      emit_indexed_update_rows(source, program, operation);
      break;
    case Opcode::Cast:
      break;
    case Opcode::SelectRowChunks:
      emit_select_row_chunks(source, program, operation);
      break;
    case Opcode::SinusoidalTimestep:
      emit_sinusoidal_timestep(source, program, operation);
      break;
    case Opcode::RotaryPosition:
      break;
    case Opcode::LinearBlend:
      emit_linear_blend(source, program, operation);
      break;
    case Opcode::FlowEulerStep:
      emit_flow_euler_step(source, program, operation);
      break;
    case Opcode::Patchify3D:
      emit_patchify_3d(source, program, operation, false);
      break;
    case Opcode::Unpatchify3D:
      emit_patchify_3d(source, program, operation, true);
      break;
    case Opcode::RmsNormModulate:
      emit_rms(source, program, operation);
      break;
    case Opcode::SwiGlu:
      emit_swiglu(source, program, operation);
      break;
    case Opcode::ResidualGate:
      emit_residual(source, program, operation);
      break;
    case Opcode::Linear:
      if (operation.u64(dif::ir::AttrKey::Implementation, 1U) == 3U)
        dif::fail("OpenCL reference backend does not implement direct packed "
                  "INT5 Linear candidate 3");
      emit_linear(source, program, operation);
      break;
    case Opcode::QkNormPartialRope:
      emit_qk_norm(source, program, operation);
      break;
    case Opcode::Attention:
      emit_attention(source, program, operation);
      break;
    case Opcode::RmsNormBackward:
    case Opcode::RmsNormModulateBackward:
    case Opcode::SwiGluBackward:
    case Opcode::ResidualGateBackward:
    case Opcode::LayerNormBackward:
    case Opcode::QkNormPartialRopeBackward:
    case Opcode::AttentionLse:
    case Opcode::AttentionBackward:
      dif::fail("OpenCL reference backend does not implement the DiT "
                "backward opcodes");
      break;
    case Opcode::Barrier:
      break;
    case Opcode::BiasAdd:
      emit_bias(source, program, operation);
      break;
    case Opcode::H3AdaLNSelect:
      emit_adaln_select(source, program, operation);
      break;
    case Opcode::H3DeinterleaveQkv:
      emit_deinterleave(source, program, operation);
      break;
    case Opcode::H3DeinterleaveQkvWeight:
      emit_deinterleave_weight(source, program, operation);
      break;
    case Opcode::DequantizeInt4:
      emit_dequantize(source, program, operation, 4U);
      break;
    case Opcode::DequantizeInt5:
      emit_dequantize(source, program, operation, 5U);
      break;
    case Opcode::Conv1d:
    case Opcode::SnakeBeta:
      // Recorded gap (BigVGAN decode plan, integrator decision 3): the audio
      // opcodes land on CPU+CUDA first; OpenCL conformance is chunk 8.
      dif::fail("OpenCL reference backend does not implement the audio "
                "decode opcodes");
      break;
    case Opcode::Gelu:
      // NVIDIA is the admitted production backend for K2-A. Keep the OpenCL
      // reference boundary explicit until its tanh transcendental parity is
      // measured instead of silently emitting a missing kernel.
      dif::fail("OpenCL reference backend does not implement tanh GELU");
      break;
    case Opcode::Sigmoid:
    case Opcode::Reshape:
    case Opcode::BroadcastTo:
    case Opcode::Slice:
    case Opcode::RotaryFrequency:
    case Opcode::RotaryApply:
    case Opcode::BooleanMaskToBias:
    case Opcode::EulerVelocityStep:
    case Opcode::Permute:
    case Opcode::Concat:
      // These generic operations currently have CPU and NVIDIA production
      // implementations. Keep OpenCL fail-closed until each lowering has its
      // own parity gate.
      dif::fail("OpenCL reference backend does not implement the Krea-required "
                "generic opcodes");
      break;
    }
    end_float_operation(source);
  }
  return source.str();
}

struct Context {
  cl_device_id device{};
  cl_context context{};
  cl_command_queue queue{};
  std::string device_name;

  ~Context() {
    if (queue)
      (void)clReleaseCommandQueue(queue);
    if (context)
      (void)clReleaseContext(context);
  }
};

struct Dispatch {
  cl_kernel kernel{};
  std::size_t global_size{};
};

struct Executable {
  Context *owner{};
  dif::ir::Program program;
  dif::compiler::MemoryPlan memory_plan;
  cl_program compiled_program{};
  std::vector<cl_mem> slots;
  std::unordered_map<std::uint32_t, cl_mem> tensors;
  std::vector<Dispatch> dispatches;

  ~Executable() {
    for (auto &dispatch : dispatches) {
      if (dispatch.kernel)
        (void)clReleaseKernel(dispatch.kernel);
    }
    if (compiled_program)
      (void)clReleaseProgram(compiled_program);
    for (const auto buffer : slots) {
      if (buffer)
        (void)clReleaseMemObject(buffer);
    }
  }
};

std::vector<cl_device_id> gpu_devices() {
  cl_uint platform_count = 0U;
  check(clGetPlatformIDs(0U, nullptr, &platform_count), "clGetPlatformIDs count");
  std::vector<cl_platform_id> platforms(platform_count);
  check(clGetPlatformIDs(platform_count, platforms.data(), nullptr),
        "clGetPlatformIDs");
  std::vector<cl_device_id> devices;
  for (const auto platform : platforms) {
    cl_uint count = 0U;
    const auto status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0U, nullptr,
                                       &count);
    if (status == CL_DEVICE_NOT_FOUND)
      continue;
    check(status, "clGetDeviceIDs count");
    std::vector<cl_device_id> current(count);
    check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, current.data(),
                         nullptr),
          "clGetDeviceIDs");
    devices.insert(devices.end(), current.begin(), current.end());
  }
  return devices;
}

dif_backend_status create(int32_t ordinal, dif_backend_context *output,
                          dif_backend_error *error) {
  try {
    if (!output || ordinal < 0)
      throw std::runtime_error("OpenCL device ordinal is invalid");
    const auto devices = gpu_devices();
    if (static_cast<std::size_t>(ordinal) >= devices.size()) {
      set_error(error, "OpenCL GPU device ordinal is unavailable");
      return DIF_BACKEND_UNSUPPORTED;
    }
    auto context = std::make_unique<Context>();
    context->device = devices[static_cast<std::size_t>(ordinal)];
    cl_int status = CL_SUCCESS;
    context->context =
        clCreateContext(nullptr, 1U, &context->device, nullptr, nullptr, &status);
    check(status, "clCreateContext");
    context->queue =
        clCreateCommandQueue(context->context, context->device, 0U, &status);
    check(status, "clCreateCommandQueue");
    std::size_t name_size = 0U;
    check(clGetDeviceInfo(context->device, CL_DEVICE_NAME, 0U, nullptr,
                          &name_size),
          "clGetDeviceInfo name size");
    std::vector<char> name(name_size);
    check(clGetDeviceInfo(context->device, CL_DEVICE_NAME, name.size(),
                          name.data(), nullptr),
          "clGetDeviceInfo name");
    context->device_name = name.data();
    *output = context.release();
    return DIF_BACKEND_OK;
  } catch (const std::bad_alloc &) {
    set_error(error, "OpenCL context allocation failed");
    return DIF_BACKEND_OUT_OF_MEMORY;
  } catch (const std::exception &exception) {
    set_error(error, exception.what());
    return DIF_BACKEND_INTERNAL_ERROR;
  }
}

void destroy(dif_backend_context context) {
  delete static_cast<Context *>(context);
}

std::size_t global_size(const dif::ir::Program &program,
                        const dif::ir::Operation &operation) {
  using dif::ir::Opcode;
  if (operation.opcode == Opcode::RmsNormModulate ||
      operation.opcode == Opcode::RmsNorm ||
      operation.opcode == Opcode::LayerNorm) {
    const auto *input = program.tensor(operation.inputs[0]);
    return static_cast<std::size_t>(input->element_count() /
                                    input->dims.back());
  }
  if (operation.opcode == Opcode::QkNormPartialRope ||
      operation.opcode == Opcode::Attention) {
    const auto &dims = program.tensor(operation.inputs[0])->dims;
    return static_cast<std::size_t>(dims[0] * dims[1]);
  }
  return static_cast<std::size_t>(
      program.tensor(operation.outputs[0])->element_count());
}

dif_backend_status compile(
    dif_backend_context opaque_context, const uint8_t *bytes, size_t byte_count,
    const dif_backend_tensor_view *constants, size_t constant_count,
    const dif_backend_compile_options_v2 *options,
    dif_backend_executable *opaque_executable,
    dif_backend_compile_telemetry_v2 *telemetry, dif_backend_error *error) {
  const auto start = std::chrono::steady_clock::now();
  try {
    if (!opaque_context || !bytes || byte_count == 0U || !options ||
        !opaque_executable || !telemetry ||
        (constant_count != 0U && !constants))
      throw std::runtime_error("OpenCL compile received invalid arguments");
    auto *context = static_cast<Context *>(opaque_context);
    auto executable = std::make_unique<Executable>();
    executable->owner = context;
    executable->program = dif::ir::decode(
        std::span<const std::uint8_t>(bytes, byte_count));
    dif::ir::verify(executable->program);

    auto resident_program = executable->program;
    for (auto &tensor : resident_program.tensors)
      tensor.roles &= ~static_cast<std::uint32_t>(dif::ir::TensorRole::Streamed);
    executable->memory_plan = dif::compiler::plan_memory(resident_program);
    cl_ulong global_memory = 0U;
    cl_ulong maximum_allocation = 0U;
    check(clGetDeviceInfo(context->device, CL_DEVICE_GLOBAL_MEM_SIZE,
                          sizeof(global_memory), &global_memory, nullptr),
          "clGetDeviceInfo global memory");
    check(clGetDeviceInfo(context->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                          sizeof(maximum_allocation), &maximum_allocation,
                          nullptr),
          "clGetDeviceInfo maximum allocation");
    if (executable->memory_plan.total_bytes > global_memory ||
        executable->memory_plan.total_bytes + options->minimum_free_bytes >
            global_memory)
      return DIF_BACKEND_OUT_OF_MEMORY;
    executable->slots.resize(executable->memory_plan.slots.size());
    for (const auto &slot : executable->memory_plan.slots) {
      if (slot.bytes > maximum_allocation)
        return DIF_BACKEND_OUT_OF_MEMORY;
      cl_int status = CL_SUCCESS;
      executable->slots[slot.id] =
          clCreateBuffer(context->context, CL_MEM_READ_WRITE,
                         static_cast<std::size_t>(slot.bytes), nullptr, &status);
      check(status, "clCreateBuffer");
    }
    for (const auto &assignment : executable->memory_plan.assignments)
      executable->tensors.emplace(assignment.tensor_id,
                                  executable->slots.at(assignment.slot_id));

    std::unordered_map<std::uint32_t, const dif_backend_tensor_view *>
        constant_views;
    for (std::size_t index = 0; index < constant_count; ++index)
      constant_views.emplace(constants[index].tensor_id, &constants[index]);
    std::size_t expected_constants = 0U;
    for (const auto &description : executable->program.tensors) {
      if (!description.has_role(dif::ir::TensorRole::Constant))
        continue;
      ++expected_constants;
      const auto found = constant_views.find(description.id);
      if (found == constant_views.end() ||
          found->second->dtype != static_cast<std::uint32_t>(description.dtype) ||
          found->second->rank != description.dims.size() ||
          found->second->byte_count != description.byte_count() ||
          !std::equal(description.dims.begin(), description.dims.end(),
                      found->second->dims))
        throw std::runtime_error("OpenCL constant binding mismatch");
      check(clEnqueueWriteBuffer(
                context->queue, executable->tensors.at(description.id), CL_TRUE,
                0U, static_cast<std::size_t>(found->second->byte_count),
                found->second->host_data, 0U, nullptr, nullptr),
            "clEnqueueWriteBuffer constant");
    }
    if (expected_constants != constant_count)
      throw std::runtime_error("OpenCL received an unexpected constant binding");

    const auto generated = generate_source(executable->program);
    const char *source_pointer = generated.c_str();
    const auto source_size = generated.size();
    cl_int status = CL_SUCCESS;
    executable->compiled_program = clCreateProgramWithSource(
        context->context, 1U, &source_pointer, &source_size, &status);
    check(status, "clCreateProgramWithSource");
    status = clBuildProgram(
        executable->compiled_program, 1U, &context->device,
        "-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt", nullptr,
        nullptr);
    if (status != CL_SUCCESS) {
      std::size_t log_size = 0U;
      (void)clGetProgramBuildInfo(executable->compiled_program, context->device,
                                  CL_PROGRAM_BUILD_LOG, 0U, nullptr, &log_size);
      std::vector<char> log(log_size + 1U, '\0');
      (void)clGetProgramBuildInfo(executable->compiled_program, context->device,
                                  CL_PROGRAM_BUILD_LOG, log_size, log.data(),
                                  nullptr);
      throw std::runtime_error("OpenCL source build failed: " +
                               std::string(log.data()));
    }
    for (const auto &operation : executable->program.operations) {
      if (operation.opcode == dif::ir::Opcode::Barrier)
        continue;
      cl_int kernel_status = CL_SUCCESS;
      const auto name = kernel_name(operation);
      auto kernel = clCreateKernel(executable->compiled_program, name.c_str(),
                                   &kernel_status);
      check(kernel_status, "clCreateKernel");
      cl_uint argument = 0U;
      for (const auto id : operation.inputs) {
        const auto buffer = executable->tensors.at(id);
        check(clSetKernelArg(kernel, argument++, sizeof(buffer), &buffer),
              "clSetKernelArg input");
      }
      for (const auto id : operation.outputs) {
        const auto buffer = executable->tensors.at(id);
        check(clSetKernelArg(kernel, argument++, sizeof(buffer), &buffer),
              "clSetKernelArg output");
      }
      executable->dispatches.push_back(
          {kernel, global_size(executable->program, operation)});
    }
    check(clFinish(context->queue), "clFinish compile");
    const auto stop = std::chrono::steady_clock::now();
    telemetry->preparation_milliseconds =
        std::chrono::duration<double, std::milli>(stop - start).count();
    telemetry->resident_bytes = executable->memory_plan.total_bytes;
    telemetry->free_bytes_before = global_memory;
    telemetry->free_bytes_after =
        global_memory - executable->memory_plan.total_bytes;
    std::snprintf(telemetry->device_name, sizeof(telemetry->device_name), "%s",
                  context->device_name.c_str());
    *opaque_executable = executable.release();
    return DIF_BACKEND_OK;
  } catch (const std::bad_alloc &) {
    set_error(error, "OpenCL executable host allocation failed");
    return DIF_BACKEND_OUT_OF_MEMORY;
  } catch (const dif::Error &exception) {
    set_error(error, exception.what());
    return DIF_BACKEND_COMPILE_ERROR;
  } catch (const std::exception &exception) {
    set_error(error, exception.what());
    return DIF_BACKEND_COMPILE_ERROR;
  }
}

void destroy_executable(dif_backend_executable executable) {
  delete static_cast<Executable *>(executable);
}

void validate_view(const dif::ir::TensorDesc &description,
                   const dif_backend_tensor_view &view) {
  if (view.tensor_id != description.id ||
      view.dtype != static_cast<std::uint32_t>(description.dtype) ||
      view.rank != description.dims.size() ||
      view.byte_count != description.byte_count() || !view.host_data ||
      !std::equal(description.dims.begin(), description.dims.end(), view.dims))
    throw std::runtime_error("OpenCL execute tensor binding mismatch");
}

dif_backend_status execute(dif_backend_executable opaque_executable,
                           const dif_backend_tensor_view *inputs,
                           size_t input_count, dif_backend_tensor_view *outputs,
                           size_t output_count,
                           const dif_backend_run_options *options,
                           dif_backend_telemetry *telemetry,
                           dif_backend_error *error) {
  try {
    if (!opaque_executable || (input_count != 0U && !inputs) ||
        (output_count != 0U && !outputs) || !options || !telemetry ||
        options->iterations == 0U)
      throw std::runtime_error("OpenCL execute received invalid arguments");
    auto *executable = static_cast<Executable *>(opaque_executable);
    auto *context = executable->owner;
    std::unordered_map<std::uint32_t, const dif_backend_tensor_view *> input_map;
    for (std::size_t index = 0; index < input_count; ++index)
      input_map.emplace(inputs[index].tensor_id, &inputs[index]);
    std::size_t expected_inputs = 0U;
    for (const auto &description : executable->program.tensors) {
      if (!description.has_role(dif::ir::TensorRole::Input))
        continue;
      ++expected_inputs;
      const auto found = input_map.find(description.id);
      if (found == input_map.end())
        throw std::runtime_error("OpenCL dynamic input is missing");
      validate_view(description, *found->second);
      check(clEnqueueWriteBuffer(
                context->queue, executable->tensors.at(description.id), CL_TRUE,
                0U, static_cast<std::size_t>(description.byte_count()),
                found->second->host_data, 0U, nullptr, nullptr),
            "clEnqueueWriteBuffer input");
    }
    if (expected_inputs != input_count)
      throw std::runtime_error("OpenCL received an unexpected dynamic input");

    auto dispatch = [&]() {
      for (const auto &entry : executable->dispatches) {
        check(clEnqueueNDRangeKernel(context->queue, entry.kernel, 1U, nullptr,
                                     &entry.global_size, nullptr, 0U, nullptr,
                                     nullptr),
              "clEnqueueNDRangeKernel");
      }
      check(clFinish(context->queue), "clFinish execute");
    };
    for (std::uint32_t warmup = 0U; warmup < options->warmups; ++warmup)
      dispatch();
    std::vector<double> elapsed;
    elapsed.reserve(options->iterations);
    for (std::uint32_t iteration = 0U; iteration < options->iterations;
         ++iteration) {
      const auto start = std::chrono::steady_clock::now();
      dispatch();
      const auto stop = std::chrono::steady_clock::now();
      elapsed.push_back(
          std::chrono::duration<double, std::milli>(stop - start).count());
    }

    std::unordered_map<std::uint32_t, dif_backend_tensor_view *> output_map;
    for (std::size_t index = 0; index < output_count; ++index)
      output_map.emplace(outputs[index].tensor_id, &outputs[index]);
    std::size_t expected_outputs = 0U;
    for (const auto &description : executable->program.tensors) {
      if (!description.has_role(dif::ir::TensorRole::Output))
        continue;
      ++expected_outputs;
      const auto found = output_map.find(description.id);
      if (found == output_map.end())
        throw std::runtime_error("OpenCL output is missing");
      validate_view(description, *found->second);
      check(clEnqueueReadBuffer(
                context->queue, executable->tensors.at(description.id), CL_TRUE,
                0U, static_cast<std::size_t>(description.byte_count()),
                found->second->host_data, 0U, nullptr, nullptr),
            "clEnqueueReadBuffer output");
    }
    if (expected_outputs != output_count)
      throw std::runtime_error("OpenCL received an unexpected output");
    telemetry->mean_milliseconds = 0.0;
    for (const auto value : elapsed)
      telemetry->mean_milliseconds += value;
    telemetry->mean_milliseconds /= static_cast<double>(elapsed.size());
    telemetry->minimum_milliseconds =
        *std::min_element(elapsed.begin(), elapsed.end());
    telemetry->maximum_milliseconds =
        *std::max_element(elapsed.begin(), elapsed.end());
    std::snprintf(telemetry->device_name, sizeof(telemetry->device_name), "%s",
                  context->device_name.c_str());
    return DIF_BACKEND_OK;
  } catch (const std::exception &exception) {
    set_error(error, exception.what());
    return DIF_BACKEND_EXECUTION_ERROR;
  }
}

const dif_backend_api_v2 api = {
    sizeof(dif_backend_api_v2), DIF_BACKEND_ABI_VERSION_V2, "opencl-reference-v2",
    DIF_BACKEND_CAP_F32 | DIF_BACKEND_CAP_BF16 | DIF_BACKEND_CAP_I8, create,
    destroy, compile, destroy_executable, execute};

} // namespace

extern "C" DIF_BACKEND_EXPORT const dif_backend_api_v2 *
dif_backend_get_v2(void) {
  return &api;
}
