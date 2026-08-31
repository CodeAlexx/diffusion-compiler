#include "dif/compiler/compiler.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <cmath>
#include <array>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace dif::compiler {
namespace {

std::string function_name(const ir::Operation &op) {
  return "dif_op_" + std::to_string(op.id);
}

struct LowbitLinearFusion {
  const ir::Operation *dequant{};
  const ir::Operation *linear{};
  std::uint64_t qkv_component{3U};
  std::uint64_t qkv_head_dim{};
};

std::unordered_map<std::uint32_t, LowbitLinearFusion>
find_lowbit_linear_fusions(const ir::Program &program,
                           std::unordered_set<std::uint32_t> &skipped) {
  std::unordered_map<std::uint32_t, const ir::Operation *> producer;
  std::unordered_map<std::uint32_t, std::vector<const ir::Operation *>>
      consumers;
  for (const auto &operation : program.operations) {
    for (const auto output : operation.outputs)
      producer.emplace(output, &operation);
    for (const auto input : operation.inputs)
      consumers[input].push_back(&operation);
  }

  std::unordered_map<std::uint32_t, LowbitLinearFusion> fusions;
  // H3 source-shaped QKV is dequantized, deinterleaved, then consumed by three
  // independent Linear operations. Fuse the complete chain only when none of
  // its intermediate values are observable or shared.
  for (const auto &operation : program.operations) {
    if (operation.opcode != ir::Opcode::H3DeinterleaveQkvWeight ||
        operation.inputs.size() != 1U || operation.outputs.size() != 3U)
      continue;
    const auto dequant_found = producer.find(operation.inputs[0]);
    if (dequant_found == producer.end() ||
        dequant_found->second->opcode != ir::Opcode::DequantizeInt5 ||
        consumers[operation.inputs[0]].size() != 1U ||
        program.tensor(operation.inputs[0])->has_role(ir::TensorRole::Output))
      continue;
    bool eligible = true;
    std::array<const ir::Operation *, 3> linears{};
    for (std::size_t component = 0; component < operation.outputs.size();
         ++component) {
      const auto weight = operation.outputs[component];
      const auto &uses = consumers[weight];
      if (uses.size() != 1U || uses.front()->opcode != ir::Opcode::Linear ||
          uses.front()->inputs.size() < 2U || uses.front()->inputs[1] != weight ||
          uses.front()->u64(ir::AttrKey::Implementation, 1U) != 3U ||
          program.tensor(weight)->has_role(ir::TensorRole::Output)) {
        eligible = false;
        break;
      }
      linears[component] = uses.front();
    }
    if (!eligible)
      continue;
    const auto head_dim = operation.u64(ir::AttrKey::HeadDim, 0U);
    for (std::size_t component = 0; component < linears.size(); ++component)
      fusions.emplace(linears[component]->id,
                      LowbitLinearFusion{dequant_found->second,
                                         linears[component], component,
                                         head_dim});
    skipped.insert(dequant_found->second->id);
    skipped.insert(operation.id);
  }

  for (const auto &operation : program.operations) {
    if (operation.opcode != ir::Opcode::Linear ||
        operation.inputs.size() < 2U || fusions.contains(operation.id))
      continue;
    if (operation.u64(ir::AttrKey::Implementation, 1U) != 3U)
      continue;
    const auto weight = operation.inputs[1];
    const auto dequant_found = producer.find(weight);
    if (dequant_found == producer.end() ||
        dequant_found->second->opcode != ir::Opcode::DequantizeInt5 ||
        consumers[weight].size() != 1U ||
        program.tensor(weight)->has_role(ir::TensorRole::Output))
      continue;
    fusions.emplace(operation.id,
                    LowbitLinearFusion{dequant_found->second, &operation});
    skipped.insert(dequant_found->second->id);
  }
  return fusions;
}

void emit_lowbit_linear(std::ostringstream &out, const ir::Program &program,
                        const LowbitLinearFusion &fusion) {
  const auto &linear = *fusion.linear;
  const auto &dequant = *fusion.dequant;
  const auto *input = program.tensor(linear.inputs[0]);
  const auto *weight = program.tensor(linear.inputs[1]);
  const auto *output = program.tensor(linear.outputs[0]);
  const auto *source_weight = program.tensor(dequant.outputs[0]);
  const auto k = weight->dims[1];
  const auto n = weight->dims[0];
  const auto m = input->element_count() / k;
  const auto group = dequant.u64(ir::AttrKey::GroupSize, 64U);
  const auto groups = k / group;
  const auto row_bytes = source_weight->dims[1] * 5U / 8U;
  const bool column_scaled = dequant.inputs.size() == 3U;
  const bool biased = linear.inputs.size() == 3U;
  if (m == 0U || m > 32U)
    fail("direct packed INT5 Linear currently admits flattened M in [1,32]");
  out << "extern \"C\" __global__ void " << function_name(linear)
      << "(const dif_scalar* x,const unsigned char* packed,const dif_scalar* scales,";
  if (column_scaled)
    out << "const dif_scalar* column_scales,";
  if (biased)
    out << "const dif_scalar* bias,";
  out << "dif_scalar* y){extern __shared__ float partials[];"
         "unsigned long long row=blockIdx.x;unsigned tid=threadIdx.x,"
         "lane=tid&31U,warp=tid>>5U;if(row>="
      << n << "ULL)return;unsigned long long source_row=";
  if (fusion.qkv_component < 3U) {
    out << "(row/" << fusion.qkv_head_dim << "ULL)*"
        << fusion.qkv_head_dim * 3U << "ULL+"
        << fusion.qkv_component * fusion.qkv_head_dim << "ULL+(row%"
        << fusion.qkv_head_dim << "ULL);";
  } else {
    out << "row;";
  }
  for (std::uint64_t batch = 0; batch < m; ++batch)
    out << "float acc" << batch << "=0.0f;";
  out << "for(unsigned long long col=tid;col<" << k
      << "ULL;col+=256ULL){unsigned long long bit=col*5ULL,bi=source_row*"
      << row_bytes
      << "ULL+bit/8ULL;unsigned shift=(unsigned)(bit&7ULL);unsigned word="
         "packed[bi];if(shift+5U>8U)word|=((unsigned)packed[bi+1ULL])<<8U;"
         "unsigned encoded=(word>>shift)&31U;int q=encoded<16U?(int)encoded:"
         "(int)encoded-32;float w=(float)q*dif_load(scales,source_row*"
      << groups << "ULL+col/" << group << "ULL);"
      << (column_scaled ? "w*=dif_load(column_scales,col);" : "")
      ;
  for (std::uint64_t batch = 0; batch < m; ++batch)
    out << "acc" << batch << "=fmaf(dif_load(x," << batch * k
        << "ULL+col),w,acc" << batch << ");";
  out << "}for(unsigned offset=16U;offset>0U;offset>>=1U){";
  for (std::uint64_t batch = 0; batch < m; ++batch)
    out << "acc" << batch << "+=__shfl_down_sync(0xffffffffU,acc" << batch
        << ",offset);";
  out << "}if(lane==0U){";
  for (std::uint64_t batch = 0; batch < m; ++batch)
    out << "partials[warp*" << m << "ULL+" << batch << "ULL]=acc" << batch
        << ";";
  out << "}__syncthreads();if(warp==0U&&lane<" << m
      << "ULL){float total=0.0f;for(unsigned source_warp=0U;source_warp<8U;"
         "++source_warp)total+=partials[source_warp*"
      << m << "ULL+lane];";
  if (biased)
    out << "total+=dif_load(bias,row);";
  out << "dif_store(y,(unsigned long long)lane*" << n
      << "ULL+row,total);}";
  out << "}\n";
  (void)output;
}

void emit_header(std::ostringstream &out) {
  out << R"CUDA(
typedef float dif_f32;
typedef unsigned short dif_bf16;
typedef unsigned short dif_f16;
extern "C" __device__ float dif_load_f32(const dif_f32* value, unsigned long long index) {
  return value[index];
}
extern "C" __device__ void dif_store_f32(dif_f32* value, unsigned long long index, float input) {
  value[index] = input;
}
extern "C" __device__ float dif_round_f32(float input) { return input; }
extern "C" __device__ float dif_load_bf16(const dif_bf16* value, unsigned long long index) {
  return __uint_as_float((unsigned int)value[index] << 16U);
}
extern "C" __device__ void dif_store_bf16(dif_bf16* value, unsigned long long index, float input) {
  unsigned int bits = __float_as_uint(input);
  unsigned int rounding = 0x7fffU + ((bits >> 16U) & 1U);
  value[index] = (dif_bf16)((bits + rounding) >> 16U);
}
extern "C" __device__ float dif_round_bf16(float input) {
  unsigned int bits = __float_as_uint(input);
  unsigned int rounding = 0x7fffU + ((bits >> 16U) & 1U);
  return __uint_as_float(((bits + rounding) >> 16U) << 16U);
}
extern "C" __device__ float dif_load_f16(const dif_f16* value, unsigned long long index) {
  float result;
  asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(value[index]));
  return result;
}
extern "C" __device__ void dif_store_f16(dif_f16* value, unsigned long long index, float input) {
  dif_f16 result;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(input));
  value[index] = result;
}
extern "C" __device__ float dif_round_f16(float input) {
  dif_f16 rounded;
  float result;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(rounded) : "f"(input));
  asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(rounded));
  return result;
}
extern "C" __device__ float dif_silu(float x) {
  return x / (1.0f + expf(-x));
}
)CUDA";
}

ir::DType operation_float_dtype(const ir::Program &program,
                                const ir::Operation &op) {
  for (const auto &ids : {&op.outputs, &op.inputs}) {
    for (const auto id : *ids) {
      const auto dtype = program.tensor(id)->dtype;
      if (dtype == ir::DType::F32 || dtype == ir::DType::BF16 ||
          dtype == ir::DType::F16)
        return dtype;
    }
  }
  fail("CUDA operation has no supported float tensor");
}

void begin_float_operation(std::ostringstream &out, ir::DType dtype) {
  if (dtype == ir::DType::F32) {
    out << "#define dif_scalar dif_f32\n#define dif_load dif_load_f32\n"
           "#define dif_store dif_store_f32\n#define dif_round dif_round_f32\n";
    return;
  }
  if (dtype == ir::DType::BF16) {
    out << "#define dif_scalar dif_bf16\n#define dif_load dif_load_bf16\n"
           "#define dif_store dif_store_bf16\n#define dif_round dif_round_bf16\n";
    return;
  }
  if (dtype == ir::DType::F16) {
    out << "#define dif_scalar dif_f16\n#define dif_load dif_load_f16\n"
           "#define dif_store dif_store_f16\n#define dif_round dif_round_f16\n";
    return;
  }
  fail("CUDA source emitter admits f32, bf16, or f16 operations");
}

void end_float_operation(std::ostringstream &out) {
  out << "#undef dif_scalar\n#undef dif_load\n#undef dif_store\n"
         "#undef dif_round\n";
}

void emit_cast(std::ostringstream &out, const ir::Program &program,
               const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto count = output->element_count();
  const auto type = [](ir::DType dtype) -> const char * {
    if (dtype == ir::DType::F32)
      return "dif_f32";
    if (dtype == ir::DType::BF16)
      return "dif_bf16";
    if (dtype == ir::DType::F16)
      return "dif_f16";
    fail("CUDA Cast admits f32, bf16, and f16 storage");
  };
  const auto load = input->dtype == ir::DType::F32
                        ? "dif_load_f32"
                        : input->dtype == ir::DType::BF16 ? "dif_load_bf16"
                                                          : "dif_load_f16";
  const auto store = output->dtype == ir::DType::F32
                         ? "dif_store_f32"
                         : output->dtype == ir::DType::BF16 ? "dif_store_bf16"
                                                            : "dif_store_f16";
  out << "extern \"C\" __global__ void " << function_name(op) << "(const "
      << type(input->dtype) << "* x," << type(output->dtype)
      << "* y){unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+"
         "threadIdx.x;if(i<"
      << count << "ULL)" << store << "(y,i," << load << "(x,i));}\n";
}

void emit_elementwise(std::ostringstream &out, const ir::Program &program,
                      const ir::Operation &op, const char *expression) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* a, const dif_scalar* b, dif_scalar* y) {\n"
      << "  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
      << "  if (i < " << count << "ULL) dif_store(y,i," << expression << ");\n}\n";
}

void emit_affine_last_dim(std::ostringstream &out,
                          const ir::Program &program,
                          const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto width = program.tensor(op.inputs[1])->dims[0];
  out << "extern \"C\" __global__ void " << function_name(op)
      << (op.inputs.size() == 3U
              ? "(const dif_scalar* x,const dif_scalar* scale,const dif_scalar* bias,dif_scalar* y){"
              : "(const dif_scalar* x,const dif_scalar* scale,dif_scalar* y){")
      << "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+"
         "threadIdx.x;if(i<"
      << count << "ULL){unsigned long long col=i%" << width
      << "ULL;float value=dif_round(dif_load(x,i)*dif_load(scale,col));"
      << (op.inputs.size() == 3U ? "value=dif_round(value+dif_load(bias,col));"
                                 : "")
      << "dif_store(y,i,value);}}\n";
}

void emit_clamp(std::ostringstream &out, const ir::Program &program,
                const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto lower = static_cast<float>(op.f64(ir::AttrKey::Lower, 0.0));
  const auto upper = static_cast<float>(op.f64(ir::AttrKey::Upper, 1.0));
  out << std::scientific << std::setprecision(9)
      << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,dif_scalar* y){unsigned long long i="
         "(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL)dif_store(y,i,fminf(" << upper << "f,fmaxf(" << lower
      << "f,dif_load(x,i))));}\n" << std::defaultfloat;
}

void emit_silu(std::ostringstream &out, const ir::Program &program,
               const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,dif_scalar* y){unsigned long long i="
         "(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL)dif_store(y,i,dif_silu(dif_load(x,i)));}\n";
}

// The training ops below may legally mix storage dtypes across their
// arguments (e.g. BF16 prediction into an F32 loss, BF16 parameters with F32
// moments), so they bypass the per-operation dif_load/dif_store macro system
// and name the typed load/store helpers per argument.  All arithmetic stays
// in F32 registers; rounding happens only at the typed store.
const char *typed_scalar(ir::DType dtype) {
  if (dtype == ir::DType::F32)
    return "dif_f32";
  if (dtype == ir::DType::BF16)
    return "dif_bf16";
  if (dtype == ir::DType::F16)
    return "dif_f16";
  fail("CUDA training emitter admits f32, bf16, or f16 storage");
}

const char *typed_load(ir::DType dtype) {
  if (dtype == ir::DType::F32)
    return "dif_load_f32";
  if (dtype == ir::DType::BF16)
    return "dif_load_bf16";
  if (dtype == ir::DType::F16)
    return "dif_load_f16";
  fail("CUDA training emitter admits f32, bf16, or f16 storage");
}

const char *typed_store(ir::DType dtype) {
  if (dtype == ir::DType::F32)
    return "dif_store_f32";
  if (dtype == ir::DType::BF16)
    return "dif_store_bf16";
  if (dtype == ir::DType::F16)
    return "dif_store_f16";
  fail("CUDA training emitter admits f32, bf16, or f16 storage");
}

void emit_mse_loss(std::ostringstream &out, const ir::Program &program,
                   const ir::Operation &op) {
  const auto *prediction = program.tensor(op.inputs[0]);
  const auto *loss = program.tensor(op.outputs[0]);
  const auto count = prediction->element_count();
  const auto *scalar = typed_scalar(prediction->dtype);
  const auto *load = typed_load(prediction->dtype);
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const " << scalar << "* prediction,const " << scalar
      << "* target," << typed_scalar(loss->dtype) << "* loss){"
         "if(blockIdx.x==0U&&threadIdx.x==0U){float sum=0.0f;for(unsigned long long i=0ULL;i<"
      << count
      << "ULL;++i){float d=" << load << "(prediction,i)-" << load
      << "(target,i);sum=fmaf(d,d,sum);}" << typed_store(loss->dtype)
      << "(loss,0ULL,sum/" << count << ".0f);}}\n";
}

void emit_mse_loss_backward(std::ostringstream &out,
                            const ir::Program &program,
                            const ir::Operation &op) {
  const auto *prediction = program.tensor(op.inputs[0]);
  const auto *grad_loss = program.tensor(op.inputs[2]);
  const auto *grad_prediction = program.tensor(op.outputs[0]);
  const auto count = grad_prediction->element_count();
  const auto *scalar = typed_scalar(prediction->dtype);
  const auto *load = typed_load(prediction->dtype);
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const " << scalar << "* prediction,const " << scalar
      << "* target,const " << typed_scalar(grad_loss->dtype)
      << "* grad_loss," << typed_scalar(grad_prediction->dtype)
      << "* grad_prediction){"
         "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count
      << "ULL){float factor=2.0f*" << typed_load(grad_loss->dtype)
      << "(grad_loss,0ULL)/" << count << ".0f;"
      << typed_store(grad_prediction->dtype) << "(grad_prediction,i,("
      << load << "(prediction,i)-" << load
      << "(target,i))*factor);}}\n";
}

void emit_linear_backward_input(std::ostringstream &out,
                                const ir::Program &program,
                                const ir::Operation &op) {
  const auto *weight = program.tensor(op.inputs[1]);
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto inner = weight->dims[1];
  const auto outputs = weight->dims[0];
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* grad_output,const dif_scalar* weight,dif_scalar* grad_input){"
         "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){unsigned long long row=i/" << inner
      << "ULL,column=i%" << inner
      << "ULL;float value=0.0f;for(unsigned long long output=0ULL;output<"
      << outputs
      << "ULL;++output)value=fmaf(dif_load(grad_output,row*" << outputs
      << "ULL+output),dif_load(weight,output*" << inner
      << "ULL+column),value);dif_store(grad_input,i,value);}}\n";
}

void emit_linear_backward_weight(std::ostringstream &out,
                                 const ir::Program &program,
                                 const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[1]);
  const auto *grad_weight = program.tensor(op.outputs[0]);
  const auto count = grad_weight->element_count();
  // Geometry from the [N,K] weight gradient itself: rank-agnostic over both
  // admitted operand forms (same-rank broadcast and flatten).
  const auto outputs = grad_weight->dims[0];
  const auto inner = grad_weight->dims[1];
  const auto rows = input->element_count() / inner;
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* grad_output,const dif_scalar* input,dif_scalar* grad_weight){"
         "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){unsigned long long output=i/" << inner
      << "ULL,column=i%" << inner
      << "ULL;float value=0.0f;for(unsigned long long row=0ULL;row<" << rows
      << "ULL;++row)value=fmaf(dif_load(grad_output,row*" << outputs
      << "ULL+output),dif_load(input,row*" << inner
      << "ULL+column),value);dif_store(grad_weight,i,value);}}\n";
}

void emit_bias_backward(std::ostringstream &out, const ir::Program &program,
                        const ir::Operation &op) {
  const auto *grad_output = program.tensor(op.inputs[0]);
  const auto width = program.tensor(op.outputs[0])->dims[0];
  const auto rows = grad_output->element_count() / width;
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* grad_output,dif_scalar* grad_bias){unsigned long long column="
         "(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(column<"
      << width
      << "ULL){float value=0.0f;for(unsigned long long row=0ULL;row<" << rows
      << "ULL;++row)value+=dif_load(grad_output,row*" << width
      << "ULL+column);dif_store(grad_bias,column,value);}}\n";
}

void emit_silu_backward(std::ostringstream &out, const ir::Program &program,
                        const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,const dif_scalar* grad_output,dif_scalar* grad_input){"
         "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count
      << "ULL){float value=dif_load(x,i),sigmoid=1.0f/(1.0f+expf(-value));"
         "float derivative=sigmoid*(1.0f+value*(1.0f-sigmoid));"
         "dif_store(grad_input,i,dif_load(grad_output,i)*derivative);}}\n";
}

void emit_adamw_update(std::ostringstream &out, const ir::Program &program,
                       const ir::Operation &op) {
  const auto *parameter = program.tensor(op.inputs[0]);
  const auto *gradient = program.tensor(op.inputs[1]);
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto learning_rate =
      static_cast<float>(op.f64(ir::AttrKey::LearningRate, 1.0e-3));
  const auto beta1 = static_cast<float>(op.f64(ir::AttrKey::Beta1, 0.9));
  const auto beta2 = static_cast<float>(op.f64(ir::AttrKey::Beta2, 0.999));
  const auto epsilon = static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-8));
  const auto weight_decay =
      static_cast<float>(op.f64(ir::AttrKey::WeightDecay, 0.0));
  // Parameter and gradient storage are typed independently (verifier admits
  // F32/BF16 for each); moments are F32 always.  Every intermediate is an
  // F32 register; the decoupled decay multiplies the parameter BEFORE the
  // moment-based update is subtracted, and weight decay is never folded
  // into the gradient ahead of the moment updates (flame's measured LoRA-A
  // "unlearning" runaway).
  const auto *parameter_scalar = typed_scalar(parameter->dtype);
  out << std::scientific << std::setprecision(9)
      << "extern \"C\" __global__ void " << function_name(op)
      << "(const " << parameter_scalar << "* parameter,const "
      << typed_scalar(gradient->dtype)
      << "* gradient,const dif_f32* first,const dif_f32* second,const int* completed_steps,"
      << parameter_scalar
      << "* updated,dif_f32* updated_first,dif_f32* updated_second){"
         "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){float step=(float)(completed_steps[0]+1),beta1="
      << beta1 << "f,beta2=" << beta2
      << "f;float grad=" << typed_load(gradient->dtype)
      << "(gradient,i);float m=beta1*dif_load_f32(first,i)+(1.0f-beta1)*grad;"
         "float v=beta2*dif_load_f32(second,i)+(1.0f-beta2)*grad*grad;float bias1=1.0f-powf(beta1,step);"
         "float bias2_sqrt=sqrtf(1.0f-powf(beta2,step));float decayed="
      << typed_load(parameter->dtype) << "(parameter,i)*(1.0f-"
      << learning_rate << "f*" << weight_decay
      << "f);float denominator=sqrtf(v)/bias2_sqrt+" << epsilon
      << "f;float value=decayed-(" << learning_rate
      << "f/bias1)*m/denominator;" << typed_store(parameter->dtype)
      << "(updated,i,value);dif_store_f32(updated_first,i,m);"
         "dif_store_f32(updated_second,i,v);}}\n"
      << std::defaultfloat;
}

void emit_fill(std::ostringstream &out, const ir::Program &program,
               const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto value = static_cast<float>(op.f64(ir::AttrKey::Value, 0.0));
  out << std::setprecision(17) << "extern \"C\" __global__ void "
      << function_name(op)
      << "(dif_scalar* y){unsigned long long i=(unsigned long long)blockIdx.x*"
         "blockDim.x+threadIdx.x;if(i<"
      << count << "ULL)dif_store(y,i," << std::scientific
      << std::setprecision(9) << value << "f);}\n" << std::defaultfloat;
}

void emit_gather_rows(std::ostringstream &out, const ir::Program &program,
                      const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto input_rows = input->dims[0];
  const auto row_width = input->element_count() / input_rows;
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,const int* indices,dif_scalar* y){unsigned long "
         "long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){unsigned long long row=i/" << row_width
      << "ULL,col=i%" << row_width
      << "ULL;int source=indices[row];if(source>=0&&source<"
      << input_rows
      << ")dif_store(y,i,dif_load(x,(unsigned long long)source*"
      << row_width
      << "ULL+col));else dif_store(y,i,__int_as_float(0x7fffffff));}}\n";
}

void emit_indexed_update_rows(std::ostringstream &out,
                              const ir::Program &program,
                              const ir::Operation &op) {
  const auto *base = program.tensor(op.inputs[0]);
  const auto *updates = program.tensor(op.inputs[1]);
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto rows = base->dims[0];
  const auto update_rows = updates->dims[0];
  const auto row_width = base->element_count() / rows;
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* base,const dif_scalar* updates,const int* map,"
         "dif_scalar* y){unsigned long long i=(unsigned long long)blockIdx.x*"
         "blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){unsigned long long row=i/" << row_width
      << "ULL,col=i%" << row_width
      << "ULL;int source=map[row];if(source==-1)dif_store(y,i,dif_load(base,i));"
         "else if(source>=0&&source<"
      << update_rows
      << ")dif_store(y,i,dif_load(updates,(unsigned long long)source*"
      << row_width
      << "ULL+col));else dif_store(y,i,__int_as_float(0x7fffffff));}}\n";
}

void emit_select_row_chunks(std::ostringstream &out,
                            const ir::Program &program,
                            const ir::Operation &op) {
  const auto *values = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto count = output->element_count();
  const auto rows = values->dims[0];
  const auto width = output->dims[1];
  const auto source_width = values->dims[1];
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* values,const int* indices";
  for (std::size_t chunk = 0; chunk < op.outputs.size(); ++chunk)
    out << ",dif_scalar* o" << chunk;
  out << "){unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+"
         "threadIdx.x;if(i<"
      << count << "ULL){unsigned long long row=i/" << width
      << "ULL,col=i%" << width << "ULL;int source=indices[row];";
  for (std::size_t chunk = 0; chunk < op.outputs.size(); ++chunk) {
    out << "if(source>=0&&source<" << rows << ")dif_store(o" << chunk
        << ",i,dif_load(values,(unsigned long long)source*" << source_width
        << "ULL+" << chunk * width
        << "ULL+col));else dif_store(o" << chunk
        << ",i,__int_as_float(0x7fffffff));";
  }
  out << "}}\n";
}

void emit_sinusoidal_timestep(std::ostringstream &out,
                              const ir::Program &program,
                              const ir::Operation &op) {
  const auto *output = program.tensor(op.outputs[0]);
  const auto count = output->element_count();
  const auto width = output->dims[1];
  const auto half = width / 2U;
  const auto flip = op.boolean(ir::AttrKey::FlipSinToCos, false);
  const auto shift = static_cast<float>(
      op.f64(ir::AttrKey::DownscaleFreqShift, 1.0));
  const auto scale = static_cast<float>(op.f64(ir::AttrKey::Scale, 1.0));
  const auto max_period =
      static_cast<float>(op.f64(ir::AttrKey::MaxPeriod, 10000.0));
  const auto log_period = static_cast<float>(std::log(max_period));
  const auto denominator = static_cast<float>(half) - shift;
  out << std::scientific << std::setprecision(9)
      << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_f32* timesteps,dif_f32* y){unsigned long long i="
         "(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){unsigned long long row=i/" << width
      << "ULL,column=i%" << width << "ULL;if(column>=" << 2U * half
      << "ULL){dif_store_f32(y,i,0.0f);return;}unsigned long long component="
         "column%"
      << half << "ULL;float exponent=(-" << log_period
      << "f*(float)component)/" << denominator
      << "f;float frequency=expf(exponent);float angle=" << scale
      << "f*(dif_load_f32(timesteps,row)*frequency);float s=sinf(angle),"
         "c=cosf(angle);dif_store_f32(y,i,"
      << (flip ? "(column<" + std::to_string(half) + "ULL?c:s)"
               : "(column<" + std::to_string(half) + "ULL?s:c)")
      << ");}}\n" << std::defaultfloat;
}

void emit_rotary_position(std::ostringstream &out,
                          const ir::Program &program,
                          const ir::Operation &op) {
  const auto *positions = program.tensor(op.inputs[0]);
  const auto *inv_freq = program.tensor(op.inputs[1]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto count = output->element_count();
  const auto axes = positions->dims[1];
  const auto frequencies = inv_freq->dims[0];
  const auto unrepeated_width = axes * frequencies;
  const auto width = 2U * unrepeated_width;
  const auto output_type = output->dtype == ir::DType::F32
                               ? "dif_f32"
                               : output->dtype == ir::DType::BF16 ? "dif_bf16"
                                                                  : "dif_f16";
  const auto store = output->dtype == ir::DType::F32
                         ? "dif_store_f32"
                         : output->dtype == ir::DType::BF16 ? "dif_store_bf16"
                                                            : "dif_store_f16";
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_f32* positions,const dif_f32* inv_freq," << output_type
      << "* cosine," << output_type
      << "* sine){unsigned long long i=(unsigned long long)blockIdx.x*"
         "blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){unsigned long long row=i/" << width
      << "ULL,column=i%" << width << "ULL,component=column%"
      << unrepeated_width << "ULL,axis=component/" << frequencies
      << "ULL,frequency=component%" << frequencies
      << "ULL;float angle=dif_load_f32(positions,row*" << axes
      << "ULL+axis)*dif_load_f32(inv_freq,frequency);" << store
      << "(cosine,i,cosf(angle));" << store << "(sine,i,sinf(angle));}}\n";
}

void emit_linear_blend(std::ostringstream &out, const ir::Program &program,
                       const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* left,const dif_scalar* right,const dif_f32* factor,"
         "dif_scalar* output){unsigned long long i=(unsigned long long)blockIdx.x*"
         "blockDim.x+threadIdx.x;if(i<"
      << count
      << "ULL){float f=dif_load_f32(factor,0ULL),left_value=dif_load(left,i),"
         "right_value=dif_load(right,i);float complement=__fsub_rn(1.0f,f);"
         "float weighted_left=__fmul_rn(f,left_value);float weighted_right="
         "__fmul_rn(complement,right_value);dif_store(output,i,__fadd_rn("
         "weighted_left,weighted_right));}}\n";
}

void emit_flow_euler_step(std::ostringstream &out,
                          const ir::Program &program,
                          const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto step = op.u64(ir::AttrKey::StepIndex, 0U);
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* sample,const dif_scalar* velocity,const dif_f32* "
         "timesteps,const dif_f32* sigmas,dif_scalar* output){unsigned long "
         "long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){float timestep=dif_load_f32(timesteps," << step
      << "ULL),sigma=dif_load_f32(sigmas," << step
      << "ULL),sigma_next=dif_load_f32(sigmas," << step + 1U
      << "ULL),sample_value=dif_load(sample,i),velocity_value=dif_load(velocity,"
         "i),sigma_from_timestep=__fsub_rn(1.0f,timestep),ratio=sigma_next/sigma;"
         "float velocity_delta=__fmul_rn(sigma_from_timestep,velocity_value);"
         "float denoised=__fadd_rn(sample_value,velocity_delta);float complement="
         "__fsub_rn(1.0f,ratio);float weighted_sample=__fmul_rn(ratio,sample_value);"
         "float weighted_denoised=__fmul_rn(complement,denoised);dif_store(output,"
         "i,__fadd_rn(weighted_sample,weighted_denoised));}}\n";
}

void emit_patchify_3d(std::ostringstream &out, const ir::Program &program,
                      const ir::Operation &op, bool inverse) {
  const auto *volume =
      program.tensor(inverse ? op.outputs[0] : op.inputs[0]);
  const auto *rows = program.tensor(inverse ? op.inputs[0] : op.outputs[0]);
  const auto patch_t = op.u64(ir::AttrKey::PatchT, 0U);
  const auto patch_h = op.u64(ir::AttrKey::PatchH, 0U);
  const auto patch_w = op.u64(ir::AttrKey::PatchW, 0U);
  const auto channels = volume->dims[1];
  const auto frames = volume->dims[2];
  const auto height = volume->dims[3];
  const auto width = volume->dims[4];
  const auto output_frames = frames / patch_t;
  const auto output_height = height / patch_h;
  const auto output_width = width / patch_w;
  const auto count = rows->element_count();
  const auto row_width = rows->dims[1];
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* input,dif_scalar* output){unsigned long long i="
         "(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){unsigned long long row=i/" << row_width
      << "ULL,column=i%" << row_width
      << "ULL,outer=row,patch_x=outer%" << output_width
      << "ULL;outer/=" << output_width << "ULL;unsigned long long patch_y=outer%"
      << output_height << "ULL;outer/=" << output_height
      << "ULL;unsigned long long patch_frame=outer%" << output_frames
      << "ULL,batch=outer/" << output_frames
      << "ULL,inner=column,offset_x=inner%" << patch_w << "ULL;inner/="
      << patch_w << "ULL;unsigned long long offset_y=inner%" << patch_h
      << "ULL;inner/=" << patch_h
      << "ULL;unsigned long long offset_t=inner%" << patch_t
      << "ULL;inner/=" << patch_t
      << "ULL;unsigned long long channel=inner,frame=patch_frame*" << patch_t
      << "ULL+offset_t,y=patch_y*" << patch_h
      << "ULL+offset_y,x=patch_x*" << patch_w
      << "ULL+offset_x,volume_index=((((batch*" << channels
      << "ULL+channel)*" << frames << "ULL+frame)*" << height
      << "ULL+y)*" << width << "ULL+x);"
      << (inverse
              ? "dif_store(output,volume_index,dif_load(input,i));"
              : "dif_store(output,i,dif_load(input,volume_index));")
      << "}}\n";
}

void emit_rms_norm(std::ostringstream &out, const ir::Program &program,
                   const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto columns = input->dims.back();
  const auto rows = input->element_count() / columns;
  const auto epsilon = op.f64(ir::AttrKey::Epsilon, 1.0e-5);
  const auto block = op.u64(ir::AttrKey::BlockSize, 256U);
  out << std::setprecision(17) << "extern \"C\" __global__ void "
      << function_name(op)
      << "(const dif_scalar* x,const dif_scalar* weight,dif_scalar* y){\n"
         "  extern __shared__ float reduction[];unsigned long long row=blockIdx.x;"
         "float local=0.0f;if(row>="
      << rows << "ULL)return;\n";
  if (columns % 4U == 0U && block >= 128U) {
    out << "  if(threadIdx.x<128U){for(unsigned long long pack=threadIdx.x;pack<"
        << columns / 4U
        << "ULL;pack+=128ULL){unsigned long long col=pack*4ULL;unsigned long "
           "long base=row*"
        << columns
        << "ULL+col;float v0=dif_load(x,base);local+=v0*v0;float v1=dif_load(x,"
           "base+1ULL);local+=v1*v1;float v2=dif_load(x,base+2ULL);local+=v2*v2;"
           "float v3=dif_load(x,base+3ULL);local+=v3*v3;}}\n"
           "  reduction[threadIdx.x]=local;__syncthreads();for(unsigned stride="
           "16U;stride>0U;stride>>=1U){unsigned lane=threadIdx.x&31U;if(threadIdx.x<"
           "128U&&lane<stride)reduction[threadIdx.x]+=reduction[threadIdx.x+stride];"
           "__syncthreads();}if(threadIdx.x==0U)reduction[0]+=reduction[64];else "
           "if(threadIdx.x==32U)reduction[32]+=reduction[96];__syncthreads();if("
           "threadIdx.x==0U)reduction[0]+=reduction[32];__syncthreads();\n";
  } else {
    out << "  for(unsigned long long col=threadIdx.x;col<" << columns
        << "ULL;col+=blockDim.x){float v=dif_load(x,row*" << columns
        << "ULL+col);local+=v*v;}reduction[threadIdx.x]=local;__syncthreads();"
           "for(unsigned stride=blockDim.x/2;stride>0;stride>>=1){if(threadIdx.x<"
           "stride)reduction[threadIdx.x]+=reduction[threadIdx.x+stride];"
           "__syncthreads();}\n";
  }
  out << "  float inv=rsqrtf(reduction[0]/" << columns << ".0f+"
      << static_cast<float>(epsilon)
      << "f);for(unsigned long long col=threadIdx.x;col<" << columns
      << "ULL;col+=blockDim.x){unsigned long long i=row*" << columns
      << "ULL+col;dif_store(y,i,dif_load(x,i)*inv*dif_load(weight,col));}}\n";
}

void emit_layer_norm(std::ostringstream &out, const ir::Program &program,
                     const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto columns = input->dims.back();
  const auto rows = input->element_count() / columns;
  const auto epsilon = op.f64(ir::AttrKey::Epsilon, 1.0e-5);
  out << std::setprecision(17) << "extern \"C\" __global__ void "
      << function_name(op)
      << "(const dif_scalar* x,const dif_scalar* weight,const dif_scalar* bias,"
         "dif_scalar* y){extern __shared__ float reduction[];unsigned long long "
         "row=blockIdx.x;if(row>="
      << rows
      << "ULL)return;float local=0.0f;for(unsigned long long col=threadIdx.x;col<"
      << columns << "ULL;col+=blockDim.x)local+=dif_load(x,row*" << columns
      << "ULL+col);reduction[threadIdx.x]=local;__syncthreads();for(unsigned "
         "stride=blockDim.x/2;stride>0;stride>>=1){if(threadIdx.x<stride)"
         "reduction[threadIdx.x]+=reduction[threadIdx.x+stride];__syncthreads();}"
         "float mean=reduction[0]/"
      << columns
      << ".0f;local=0.0f;for(unsigned long long col=threadIdx.x;col<"
      << columns << "ULL;col+=blockDim.x){float centered=dif_load(x,row*"
      << columns
      << "ULL+col)-mean;local+=centered*centered;}reduction[threadIdx.x]=local;"
         "__syncthreads();for(unsigned stride=blockDim.x/2;stride>0;stride>>=1)"
         "{if(threadIdx.x<stride)reduction[threadIdx.x]+=reduction[threadIdx.x+"
         "stride];__syncthreads();}float inv=rsqrtf(reduction[0]/"
      << columns << ".0f+" << static_cast<float>(epsilon)
      << "f);for(unsigned long long col=threadIdx.x;col<" << columns
      << "ULL;col+=blockDim.x){unsigned long long i=row*" << columns
      << "ULL+col;dif_store(y,i,(dif_load(x,i)-mean)*inv*dif_load(weight,col)+"
         "dif_load(bias,col));}}\n";
}

void emit_rms_norm_modulate(std::ostringstream &out, const ir::Program &program,
                            const ir::Operation &op) {
  const auto &shape = program.tensor(op.inputs[0])->dims;
  const auto rows = shape[0];
  const auto cols = shape[1];
  const auto epsilon = op.f64(ir::AttrKey::Epsilon, 1.0e-5);
  const bool weighted = op.inputs.size() == 4;
  const auto block = op.u64(ir::AttrKey::BlockSize, 256U);
  const auto *scale = program.tensor(op.inputs[weighted ? 2U : 1U]);
  if (weighted && program.tensor(op.inputs[0])->dtype == ir::DType::BF16 &&
      block == 256U && scale->element_count() % cols == 0U &&
      rows % (scale->element_count() / cols) == 0U) {
    const auto rows_per_vector = rows / (scale->element_count() / cols);
    // Port of Serenity's accepted fused BF16 RMSNorm + AdaLN modulation.
    // Preserve the BF16 norm output boundary, then perform modulation in F32
    // with only the final BF16 store.
    out << std::setprecision(17)
        << "extern \"C\" __global__ void " << function_name(op)
        << "(const dif_scalar* x,const dif_scalar* weight,const dif_scalar* scale,const dif_scalar* shift,dif_scalar* y){"
           "extern __shared__ float reduction[];unsigned long long row=blockIdx.x;"
           "unsigned tid=threadIdx.x;if(row>="
        << rows
        << "ULL)return;float local=0.0f;for(unsigned long long col=tid;col<"
        << cols
        << "ULL;col+=256ULL){float value=dif_load(x,row*" << cols
        << "ULL+col);local=__fadd_rn(local,__fmul_rn(value,value));}"
           "reduction[tid]=local;__syncthreads();for(unsigned active=128U;active>0U;active>>=1U){"
           "if(tid<active)reduction[tid]=__fadd_rn(reduction[tid],reduction[tid+active]);"
           "__syncthreads();}float inv=rsqrtf(__fadd_rn(__fdiv_rn(reduction[0],"
        << cols << ".0f)," << static_cast<float>(epsilon)
        << "f));unsigned long long vector=(row/" << rows_per_vector << "ULL)*"
        << cols << "ULL;for(unsigned long long col=tid;col<" << cols
        << "ULL;col+=256ULL){unsigned long long i=row*" << cols
        << "ULL+col;float normed=dif_round(dif_load(x,i)*inv*dif_load(weight,col));"
           "float result=(1.0f+dif_load(scale,vector+col))*normed+dif_load(shift,vector+col);"
           "dif_store(y,i,result);}}\n";
    return;
  }
  out << std::setprecision(17)
      << "extern \"C\" __global__ void " << function_name(op)
      << (weighted
              ? "(const dif_scalar* x, const dif_scalar* weight, const dif_scalar* scale, const dif_scalar* shift, dif_scalar* y) {\n"
              : "(const dif_scalar* x, const dif_scalar* scale, const dif_scalar* shift, dif_scalar* y) {\n")
      << "  extern __shared__ float reduction[];\n"
      << "  unsigned row = blockIdx.x;\n"
      << "  float local = 0.0f;\n"
      << "  if (row >= " << rows << "ULL) return;\n";
  if (cols % 4U == 0U && block >= 128U) {
    // Match PyTorch's vectorized CUDA RMSNorm: 128 logical threads, four
    // adjacent values per vector, per-warp 16..1 reduction, then the four
    // warp totals are combined as (warp0 + warp2) + (warp1 + warp3).
    out << "  if(threadIdx.x<128U){for(unsigned long long pack=threadIdx.x;pack<"
        << cols / 4U
        << "ULL;pack+=128ULL){unsigned long long col=pack*4ULL;"
           "unsigned long long base=(unsigned long long)row*"
        << cols
        << "ULL+col;float v0=dif_load(x,base);local+=v0*v0;"
           "float v1=dif_load(x,base+1ULL);local+=v1*v1;"
           "float v2=dif_load(x,base+2ULL);local+=v2*v2;"
           "float v3=dif_load(x,base+3ULL);local+=v3*v3;}}\n"
           "  reduction[threadIdx.x]=local;__syncthreads();\n"
           "  for(unsigned stride=16U;stride>0U;stride>>=1U){unsigned lane=threadIdx.x&31U;"
           "if(threadIdx.x<128U&&lane<stride)reduction[threadIdx.x]+="
           "reduction[threadIdx.x+stride];__syncthreads();}\n"
           "  if(threadIdx.x==0U)reduction[0]+=reduction[64];"
           "else if(threadIdx.x==32U)reduction[32]+=reduction[96];__syncthreads();\n"
           "  if(threadIdx.x==0U)reduction[0]+=reduction[32];__syncthreads();\n";
  } else {
    out << "  for (unsigned long long col = threadIdx.x; col < " << cols
        << "ULL; col += blockDim.x) { float v=dif_load(x,(unsigned long long)row*"
        << cols
        << "ULL+col); local += v*v; }\n"
           "  reduction[threadIdx.x] = local; __syncthreads();\n"
           "  for (unsigned stride=blockDim.x/2; stride>0; stride>>=1) {"
           " if (threadIdx.x<stride) reduction[threadIdx.x]+=reduction[threadIdx.x+stride];"
           " __syncthreads(); }\n";
  }
  out << "  float inv = rsqrtf(reduction[0] / " << cols << ".0f + "
      << static_cast<float>(epsilon) << "f);\n"
      << "  for (unsigned long long col=threadIdx.x; col<" << cols
      << "ULL; col+=blockDim.x) { unsigned long long i=(unsigned long long)row*"
      << cols << "ULL+col; float value=dif_load(x,i)*inv"
      << (weighted ? "*dif_load(weight,col)" : "")
      << ";value=dif_round(value);float modulation=dif_round(1.0f+dif_load(scale,i));"
         "value=dif_round(value*modulation);dif_store(y,i,value+dif_load(shift,i)); }\n}\n";
}

void emit_swiglu(std::ostringstream &out, const ir::Program &program,
                 const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto width = program.tensor(op.outputs[0])->dims.back();
  const bool gate_first = op.boolean(ir::AttrKey::GateFirst, false);
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x, dif_scalar* y) {\n"
      << "  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;\n"
      << "  if(i<" << count << "ULL){ unsigned long long row=i/" << width
      << "ULL, col=i%" << width << "ULL; float value=dif_load(x,row*" << width * 2U
      << "ULL+" << (gate_first ? width : 0U)
      << "ULL+col); float gate=dif_load(x,row*" << width * 2U << "ULL+"
      << (gate_first ? 0U : width)
      << "ULL+col); float activated=dif_round(dif_silu(gate));"
         "dif_store(y,i,value*activated);}\n}\n";
}

void emit_bias_add(std::ostringstream &out, const ir::Program &program,
                   const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto width = program.tensor(op.inputs[1])->dims[0];
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,const dif_scalar* bias,dif_scalar* y){"
         "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;"
         "if(i<"
      << count
      << "ULL)dif_store(y,i,dif_load(x,i)+dif_load(bias,i%" << width
      << "ULL));}\n";
}

void emit_h3_adaln_select(std::ostringstream &out, const ir::Program &program,
                          const ir::Operation &op) {
  const auto &shape = program.tensor(op.outputs[0])->dims;
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto hidden = shape[1];
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* projected,const int* indices,dif_scalar* o0,"
         "dif_scalar* o1,dif_scalar* o2,dif_scalar* o3,dif_scalar* o4,"
         "dif_scalar* o5){unsigned long long i=(unsigned long long)blockIdx.x*"
         "blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){unsigned long long row=i/" << hidden
      << "ULL,col=i%" << hidden
      << "ULL,table=(unsigned long long)indices[row];"
         "dif_store(o0,i,dif_load(projected,(table*6ULL+0ULL)*"
      << hidden
      << "ULL+col));dif_store(o1,i,dif_load(projected,(table*6ULL+1ULL)*"
      << hidden
      << "ULL+col));dif_store(o2,i,dif_load(projected,(table*6ULL+2ULL)*"
      << hidden
      << "ULL+col));dif_store(o3,i,dif_load(projected,(table*6ULL+3ULL)*"
      << hidden
      << "ULL+col));dif_store(o4,i,dif_load(projected,(table*6ULL+4ULL)*"
      << hidden
      << "ULL+col));dif_store(o5,i,dif_load(projected,(table*6ULL+5ULL)*"
      << hidden << "ULL+col));}}\n";
}

void emit_h3_deinterleave_qkv(std::ostringstream &out,
                              const ir::Program &program,
                              const ir::Operation &op) {
  const auto &shape = program.tensor(op.outputs[0])->dims;
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto heads = shape[1];
  const auto dim = shape[2];
  const auto packed = 3U * heads * dim;
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,dif_scalar* q,dif_scalar* k,dif_scalar* v){"
         "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;"
         "if(i<"
      << count << "ULL){unsigned long long row=i/" << heads * dim
      << "ULL,within=i%" << heads * dim << "ULL,head=within/" << dim
      << "ULL,d=within%" << dim << "ULL,base=row*" << packed
      << "ULL+head*" << 3U * dim
      << "ULL+d;dif_store(q,i,dif_load(x,base));dif_store(k,i,dif_load(x,base+"
      << dim
      << "ULL));dif_store(v,i,dif_load(x,base+" << 2U * dim << "ULL));}}\n";
}

void emit_h3_deinterleave_qkv_weight(std::ostringstream &out,
                                     const ir::Program &program,
                                     const ir::Operation &op) {
  const auto &shape = program.tensor(op.outputs[0])->dims;
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto dim = op.u64(ir::AttrKey::HeadDim, 0U);
  const auto hidden = shape[1];
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* packed,dif_scalar* q,dif_scalar* k,dif_scalar* v){"
         "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;"
         "if(i<"
      << count << "ULL){unsigned long long row=i/" << hidden
      << "ULL,col=i%" << hidden << "ULL,head=row/" << dim
      << "ULL,d=row%" << dim << "ULL,base=((head*3ULL)*" << dim
      << "ULL+d)*" << hidden
      << "ULL+col;dif_store(q,i,dif_load(packed,base));"
         "dif_store(k,i,dif_load(packed,base+"
      << dim * hidden
      << "ULL));dif_store(v,i,dif_load(packed,base+" << 2U * dim * hidden
      << "ULL));}}\n";
}

void emit_dequantize_int4(std::ostringstream &out,
                          const ir::Program &program,
                          const ir::Operation &op) {
  const auto &shape = program.tensor(op.outputs[0])->dims;
  const auto rows = shape[0];
  const auto columns = shape[1];
  const auto groups = columns / op.u64(ir::AttrKey::GroupSize, 64U);
  const auto group = op.u64(ir::AttrKey::GroupSize, 64U);
  const auto count = rows * columns;
  out << "extern \"C\" __global__ void " << function_name(op)
      << (op.inputs.size() == 4U
              ? "(const unsigned char* packed,const dif_scalar* scales,const unsigned char* outlier_indices,const dif_scalar* outlier_residuals,dif_scalar* y){"
              : "(const unsigned char* packed,const dif_scalar* scales,dif_scalar* y){")
      << "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;"
         "if(i<"
      << count << "ULL){unsigned long long row=i/" << columns
      << "ULL,col=i%" << columns
      << "ULL;unsigned char byte=packed[row*" << columns / 2U
      << "ULL+col/2ULL];unsigned int nibble=(col&1ULL)?(byte>>4U):(byte&15U);"
         "int q=nibble<8U?(int)nibble:(int)nibble-16;"
         "float scale=dif_load(scales,row*"
      << groups << "ULL+col/" << group
      << "ULL);float value=(float)q*scale;"
      << (op.inputs.size() == 4U
              ? "unsigned long long gi=row*" + std::to_string(groups) +
                    "ULL+col/" + std::to_string(group) +
                    "ULL;if(outlier_indices[gi]==col%" +
                    std::to_string(group) +
                    "ULL)value+=dif_load(outlier_residuals,gi);"
              : "")
      << "dif_store(y,i,value);}}\n";
}

void emit_dequantize_int5(std::ostringstream &out,
                          const ir::Program &program,
                          const ir::Operation &op) {
  const auto &shape = program.tensor(op.outputs[0])->dims;
  const auto rows = shape[0];
  const auto columns = shape[1];
  const auto row_bytes = columns * 5U / 8U;
  const auto group = op.u64(ir::AttrKey::GroupSize, 64U);
  const auto groups = columns / group;
  const auto count = rows * columns;
  out << "extern \"C\" __global__ void " << function_name(op)
      << (op.inputs.size() == 3U
              ? "(const unsigned char* packed,const dif_scalar* scales,const dif_scalar* column_scales,dif_scalar* y){"
              : "(const unsigned char* packed,const dif_scalar* scales,dif_scalar* y){")
      << "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;"
         "if(i<"
      << count << "ULL){unsigned long long row=i/" << columns
      << "ULL,col=i%" << columns
      << "ULL,bit=col*5ULL,bi=row*" << row_bytes
      << "ULL+bit/8ULL;unsigned int shift=(unsigned int)(bit&7ULL);"
         "unsigned int word=packed[bi];if(shift+5U>8U)word|=((unsigned int)"
         "packed[bi+1ULL])<<8U;unsigned int encoded=(word>>shift)&31U;"
         "int q=encoded<16U?(int)encoded:(int)encoded-32;float scale="
         "dif_load(scales,row*"
      << groups << "ULL+col/" << group
      << "ULL);float value=(float)q*scale;"
      << (op.inputs.size() == 3U ? "value*=dif_load(column_scales,col);" : "")
      << "dif_store(y,i,value);}}\n";
}

void emit_residual_gate(std::ostringstream &out, const ir::Program &program,
                        const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* residual, const dif_scalar* branch, const dif_scalar* gate, dif_scalar* y) {\n"
      << "  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;"
         " if(i<"
      << count
      << "ULL) dif_store(y,i,dif_load(residual,i)+"
         "dif_round(dif_load(gate,i)*dif_load(branch,i)));\n}\n";
}

void emit_qk_norm_rope(std::ostringstream &out, const ir::Program &program,
                       const ir::Operation &op) {
  const auto &shape = program.tensor(op.inputs[0])->dims;
  const auto sequence = shape[0];
  const auto heads = shape[1];
  const auto dim = shape[2];
  const auto rotary = op.u64(ir::AttrKey::RotaryDim, dim);
  const auto half = rotary / 2U;
  const auto table_width = program.tensor(op.inputs[2])->dims[1];
  const auto epsilon = op.f64(ir::AttrKey::Epsilon, 1.0e-5);
  if (program.tensor(op.inputs[0])->dtype == ir::DType::BF16 && dim == 128U &&
      table_width == rotary) {
    // Port of Serenity's accepted MiniMax-H3 fused Q/K RMSNorm + partial-RoPE
    // primitive. One lane owns one head value, preserving the 128-lane F32
    // reduction and the required BF16 normalization boundary before RoPE.
    out << std::setprecision(17)
        << "extern \"C\" __global__ void " << function_name(op)
        << "(const dif_scalar* x,const dif_scalar* weight,const dif_scalar* cosv,const dif_scalar* sinv,dif_scalar* y){\n"
           " extern __shared__ float reduction[];unsigned long long row=blockIdx.x;"
           "unsigned tid=threadIdx.x;if(row>="
        << sequence * heads
        << "ULL)return;unsigned long long base=row*128ULL;float local=0.0f;"
           "for(unsigned col=tid;col<128U;col+=128U){float value=dif_load(x,base+col);"
           "local=__fadd_rn(local,__fmul_rn(value,value));}"
           "reduction[tid]=local;__syncthreads();for(unsigned active=64U;active>0U;active>>=1U){"
           "if(tid<active)reduction[tid]=__fadd_rn(reduction[tid],reduction[tid+active]);__syncthreads();}"
           "float inv=rsqrtf(__fadd_rn(__fdiv_rn(reduction[0],128.0f),"
        << static_cast<float>(epsilon)
        << "f));unsigned long long token=row/" << heads << "ULL;if(tid<"
        << half
        << "U){unsigned lane=tid;float value0=dif_load(x,base+lane);"
           "float value1=dif_load(x,base+lane+"
        << half
        << "ULL);float norm0=dif_round(value0*inv*dif_load(weight,lane));"
           "float norm1=dif_round(value1*inv*dif_load(weight,lane+"
        << half
        << "ULL));unsigned long long table=token*" << table_width
        << "ULL;float result0=norm0*dif_load(cosv,table+lane)-norm1*dif_load(sinv,table+lane);"
           "float result1=norm1*dif_load(cosv,table+lane+"
        << half
        << "ULL)+norm0*dif_load(sinv,table+lane+" << half
        << "ULL);dif_store(y,base+lane,result0);dif_store(y,base+lane+"
        << half << "ULL,result1);}else if(tid>=" << rotary
        << "U&&tid<128U){float value=dif_load(x,base+tid);"
           "dif_store(y,base+tid,value*inv*dif_load(weight,tid));}}\n";
    return;
  }
  out << std::setprecision(17)
      << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,const dif_scalar* weight,const dif_scalar* cosv,const dif_scalar* sinv,dif_scalar* y){\n"
      << " extern __shared__ float reduction[]; unsigned long long bh=blockIdx.x;"
         " if(bh>="
      << sequence * heads << "ULL)return; unsigned long long s=bh/" << heads
      << "ULL,h=bh%" << heads << "ULL; unsigned long long base=(s*" << heads
      << "ULL+h)*" << dim << "ULL; float local=0.0f;\n";
  if (dim <= 128U && dim % 4U == 0U) {
    // Match the CUDA vectorized RMSNorm reduction used by the pinned PyTorch
    // oracle: one warp consumes four adjacent values per lane, followed by a
    // 16,8,4,2,1 shuffle-style reduction.  The exact grouping matters at BF16
    // boundaries even though it changes only a handful of values.
    out << " if(threadIdx.x<32U){unsigned long long d=(unsigned long long)threadIdx.x*4ULL;"
           "if(d<"
        << dim
        << "ULL){float v0=dif_load(x,base+d);local+=v0*v0;"
           "float v1=dif_load(x,base+d+1ULL);local+=v1*v1;"
           "float v2=dif_load(x,base+d+2ULL);local+=v2*v2;"
           "float v3=dif_load(x,base+d+3ULL);local+=v3*v3;}}"
           "reduction[threadIdx.x]=local;__syncthreads();\n"
           " for(unsigned stride=16U;stride>0U;stride>>=1U){if(threadIdx.x<stride)"
           "reduction[threadIdx.x]+=reduction[threadIdx.x+stride];__syncthreads();}\n";
  } else {
    out << " for(unsigned long long d=threadIdx.x;d<" << dim
        << "ULL;d+=blockDim.x){float v=dif_load(x,base+d);local+=v*v;}"
           "reduction[threadIdx.x]=local;__syncthreads();\n"
           " for(unsigned stride=blockDim.x/2;stride>0;stride>>=1){if(threadIdx.x<stride)"
           " reduction[threadIdx.x]+=reduction[threadIdx.x+stride];__syncthreads();}\n";
  }
  out << " float inv=rsqrtf(reduction[0]/" << dim << ".0f+"
      << static_cast<float>(epsilon) << "f);\n"
      << " for(unsigned long long d=threadIdx.x;d<" << dim
      << "ULL;d+=blockDim.x){float value=dif_round(dif_load(x,base+d)*inv*dif_load(weight,d));"
         "float result=value; if(d<"
      << half
      << "ULL){float other=dif_round(dif_load(x,base+d+" << half
      << "ULL)*inv*dif_load(weight,d+" << half
      << "ULL));float left=dif_round(value*dif_load(cosv,s*" << table_width
      << "ULL+d));float right=dif_round(other*dif_load(sinv,s*" << table_width
      << "ULL+d));result=dif_round(left-right);}else if(d<" << rotary
      << "ULL){unsigned long long r=d-" << half
      << "ULL;float other=dif_round(dif_load(x,base+r)*inv*dif_load(weight,r));"
         "unsigned long long ti="
      << (table_width == rotary ? "d" : "r")
      << ";float left=dif_round(value*dif_load(cosv,s*" << table_width
      << "ULL+ti));float right=dif_round(other*dif_load(sinv,s*" << table_width
      << "ULL+ti));result=dif_round(left+right);} dif_store(y,base+d,result);} }\n";
}

void emit_attention(std::ostringstream &out, const ir::Program &program,
                    const ir::Operation &op) {
  const auto &shape = program.tensor(op.inputs[0])->dims;
  const auto sequence = shape[0];
  const auto heads = shape[1];
  const auto dim = shape[2];
  const auto scale = op.f64(ir::AttrKey::AttentionScale,
                            1.0 / std::sqrt(static_cast<double>(dim)));
  const auto causal = op.boolean(ir::AttrKey::Causal, false);
  out << std::setprecision(17)
      << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* q,const dif_scalar* k,const dif_scalar* v,dif_scalar* y){\n"
      << " extern __shared__ float shared[]; float* reduction=shared;"
         "float* probabilities=shared+blockDim.x;"
         "unsigned long long item=blockIdx.x;if(item>="
      << sequence * heads << "ULL)return;unsigned long long qs=item/" << heads
      << "ULL,h=item%" << heads << "ULL;unsigned long long kend="
      << (causal ? "qs+1ULL" : std::to_string(sequence) + "ULL") << ";\n"
      << " for(unsigned long long ks=0;ks<kend;++ks){float partial=0.0f;"
         "for(unsigned long long d=threadIdx.x;d<"
      << dim << "ULL;d+=blockDim.x)partial=fmaf(dif_load(q,(qs*" << heads << "ULL+h)*"
      << dim << "ULL+d),dif_load(k,(ks*" << heads << "ULL+h)*" << dim
      << "ULL+d),partial);reduction[threadIdx.x]=partial;__syncthreads();"
         "for(unsigned int stride=blockDim.x/2;stride>0;stride>>=1){"
         "if(threadIdx.x<stride)reduction[threadIdx.x]+=reduction[threadIdx.x+stride];"
         "__syncthreads();}if(threadIdx.x==0)probabilities[ks]=reduction[0]*"
      << static_cast<float>(scale) << "f;__syncthreads();}\n"
      << " if(threadIdx.x==0){float maximum=-3.402823466e+38f;"
         "for(unsigned long long ks=0;ks<kend;++ks)maximum=fmaxf(maximum,probabilities[ks]);"
         "float denominator=0.0f;for(unsigned long long ks=0;ks<kend;++ks){"
         "probabilities[ks]=expf(probabilities[ks]-maximum);denominator+=probabilities[ks];}"
         "for(unsigned long long ks=0;ks<kend;++ks)probabilities[ks]/=denominator;}"
         "__syncthreads();\n"
      << " for(unsigned long long d=threadIdx.x;d<" << dim
      << "ULL;d+=blockDim.x){float acc=0.0f;for(unsigned long long ks=0;ks<kend;++ks)"
         "acc=fmaf(probabilities[ks],dif_load(v,(ks*"
      << heads << "ULL+h)*" << dim << "ULL+d),acc);dif_store(y,(qs*" << heads
      << "ULL+h)*" << dim << "ULL+d,acc);} }\n";
}

// DiT backward emitters.  Contract: generic launch (one thread per element
// of output[0], no shared memory), F32 register math with serial F32
// accumulators for every cross-element reduction, one typed rounding store
// per output element (flame dtype contract).
void emit_rms_norm_backward(std::ostringstream &out,
                            const ir::Program &program,
                            const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[1]);
  const auto columns = input->dims.back();
  const auto rows = input->element_count() / columns;
  const auto count = rows * columns;
  const auto epsilon = static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  const bool weight_grad = op.outputs.size() == 2U;
  out << std::scientific << std::setprecision(9)
      << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* grad_output,const dif_scalar* x,const dif_scalar* weight,dif_scalar* grad_input"
      << (weight_grad ? ",dif_scalar* grad_weight" : "")
      << "){unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){unsigned long long row=i/" << columns
      << "ULL,base=row*" << columns
      << "ULL;float ss=0.0f;for(unsigned long long k=0ULL;k<" << columns
      << "ULL;++k){float value=dif_load(x,base+k);ss=fmaf(value,value,ss);}"
         "float inv=rsqrtf(ss/"
      << columns << ".0f+" << epsilon
      << "f);float dot=0.0f;for(unsigned long long k=0ULL;k<" << columns
      << "ULL;++k)dot=fmaf(dif_load(grad_output,base+k)*dif_load(weight,k),"
         "dif_load(x,base+k),dot);float value=dif_load(x,i);"
         "float gradient=dif_load(grad_output,i)*dif_load(weight,i-base)*inv-"
         "value*inv*inv*inv*dot/"
      << columns << ".0f;dif_store(grad_input,i,gradient);";
  if (weight_grad)
    out << "if(i<" << columns
        << "ULL){float acc=0.0f;for(unsigned long long r=0ULL;r<" << rows
        << "ULL;++r){unsigned long long rb=r*" << columns
        << "ULL;float rss=0.0f;for(unsigned long long k=0ULL;k<" << columns
        << "ULL;++k){float rv=dif_load(x,rb+k);rss=fmaf(rv,rv,rss);}"
           "float rinv=rsqrtf(rss/"
        << columns << ".0f+" << epsilon
        << "f);acc=fmaf(dif_load(grad_output,rb+i)*dif_load(x,rb+i),rinv,acc);}"
           "dif_store(grad_weight,i,acc);}";
  out << "}}\n" << std::defaultfloat;
}

void emit_rms_norm_modulate_backward(std::ostringstream &out,
                                     const ir::Program &program,
                                     const ir::Operation &op) {
  const bool weighted = op.inputs.size() == 4U;
  const auto *x = program.tensor(op.inputs[1]);
  const auto rows = x->dims[0];
  const auto columns = x->dims[1];
  const auto count = rows * columns;
  const auto epsilon = static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  out << std::scientific << std::setprecision(9)
      << "extern \"C\" __global__ void " << function_name(op)
      << (weighted
              ? "(const dif_scalar* grad_output,const dif_scalar* x,const dif_scalar* weight,const dif_scalar* scale,dif_scalar* grad_input,dif_scalar* grad_scale,dif_scalar* grad_shift,dif_scalar* grad_weight){"
              : "(const dif_scalar* grad_output,const dif_scalar* x,const dif_scalar* scale,dif_scalar* grad_input,dif_scalar* grad_scale,dif_scalar* grad_shift){")
      << "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){unsigned long long row=i/" << columns
      << "ULL,col=i%" << columns << "ULL,base=row*" << columns
      << "ULL;float ss=0.0f;for(unsigned long long k=0ULL;k<" << columns
      << "ULL;++k){float value=dif_load(x,base+k);ss=fmaf(value,value,ss);}"
         "float inv=rsqrtf(ss/"
      << columns << ".0f+" << epsilon
      << "f);float dot=0.0f;for(unsigned long long k=0ULL;k<" << columns
      << "ULL;++k)dot=fmaf(dif_load(grad_output,base+k)*(1.0f+dif_load(scale,"
         "base+k))"
      << (weighted ? "*dif_load(weight,k)" : "")
      << ",dif_load(x,base+k),dot);float value=dif_load(x,i);"
         "float upstream=dif_load(grad_output,i);"
         "float normed_gradient=upstream*(1.0f+dif_load(scale,i));"
         "float weight_value="
      << (weighted ? "dif_load(weight,col);" : "1.0f;")
      << "dif_store(grad_input,i,normed_gradient*weight_value*inv-"
         "value*inv*inv*inv*dot/"
      << columns << ".0f);"
         "dif_store(grad_scale,i,upstream*value*inv*weight_value);"
         "dif_store(grad_shift,i,upstream);";
  if (weighted)
    out << "if(i<" << columns
        << "ULL){float acc=0.0f;for(unsigned long long r=0ULL;r<" << rows
        << "ULL;++r){unsigned long long rb=r*" << columns
        << "ULL;float rss=0.0f;for(unsigned long long k=0ULL;k<" << columns
        << "ULL;++k){float rv=dif_load(x,rb+k);rss=fmaf(rv,rv,rss);}"
           "float rinv=rsqrtf(rss/"
        << columns << ".0f+" << epsilon
        << "f);acc=fmaf(dif_load(grad_output,rb+i)*(1.0f+dif_load(scale,rb+i))"
           "*dif_load(x,rb+i),rinv,acc);}dif_store(grad_weight,i,acc);}";
  out << "}}\n" << std::defaultfloat;
}

void emit_swiglu_backward(std::ostringstream &out, const ir::Program &program,
                          const ir::Operation &op) {
  const auto *grad_output = program.tensor(op.inputs[0]);
  const auto width = grad_output->dims.back();
  const auto count = program.tensor(op.outputs[0])->element_count();
  const bool gate_first = op.boolean(ir::AttrKey::GateFirst, false);
  // Thread i owns one element of the packed [.., 2W] input gradient; the
  // value half receives silu(gate)*g, the gate half dsilu(gate)*value*g.
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* grad_output,const dif_scalar* x,dif_scalar* grad_input){"
         "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){unsigned long long row=i/" << width * 2U
      << "ULL,col=i%" << width * 2U << "ULL,cw=col<" << width
      << "ULL?col:col-" << width << "ULL,base=row*" << width * 2U
      << "ULL;float value=dif_load(x,base+" << (gate_first ? width : 0U)
      << "ULL+cw);float gate=dif_load(x,base+" << (gate_first ? 0U : width)
      << "ULL+cw);float sigmoid=1.0f/(1.0f+expf(-gate));"
         "float upstream=dif_load(grad_output,row*"
      << width << "ULL+cw);int is_value_slot=col"
      << (gate_first ? ">=" : "<") << width
      << "ULL;float gradient=is_value_slot?gate*sigmoid*upstream:"
         "sigmoid*(1.0f+gate*(1.0f-sigmoid))*value*upstream;"
         "dif_store(grad_input,i,gradient);}}\n";
}

void emit_qk_norm_rope_backward(std::ostringstream &out,
                                const ir::Program &program,
                                const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[1]);
  const auto sequence = input->dims[0];
  const auto heads = input->dims[1];
  const auto dim = input->dims[2];
  const auto count = sequence * heads * dim;
  const auto rotary = op.u64(ir::AttrKey::RotaryDim, dim);
  const auto half = rotary / 2U;
  const auto table_width = program.tensor(op.inputs[3])->dims[1];
  const auto epsilon = static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  const bool weight_grad = op.outputs.size() == 2U;
  // rot(k): the rotation-transpose of the upstream gradient at offset k of
  // one head row (rb = row base, tb = table base), F32 registers.
  const auto rotated = [&](const std::string &row_base,
                           const std::string &table_base,
                           const std::string &k) {
    std::ostringstream expression;
    expression << "(" << k << "<" << half << "ULL?"
               << "dif_load(grad_output," << row_base << "+" << k
               << ")*dif_load(cosv," << table_base << "+" << k
               << ")+dif_load(grad_output," << row_base << "+" << k << "+"
               << half << "ULL)*dif_load(sinv," << table_base << "+"
               << (table_width == rotary ? k + "+" + std::to_string(half) +
                                               "ULL"
                                         : k)
               << "):(" << k << "<" << rotary << "ULL?"
               << "-dif_load(grad_output," << row_base << "+" << k << "-"
               << half << "ULL)*dif_load(sinv," << table_base << "+" << k
               << "-" << half << "ULL)+dif_load(grad_output," << row_base
               << "+" << k << ")*dif_load(cosv," << table_base << "+"
               << (table_width == rotary ? k
                                         : k + "-" + std::to_string(half) +
                                               "ULL")
               << "):dif_load(grad_output," << row_base << "+" << k << ")))";
    return expression.str();
  };
  out << std::scientific << std::setprecision(9)
      << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* grad_output,const dif_scalar* x,const dif_scalar* weight,const dif_scalar* cosv,const dif_scalar* sinv,dif_scalar* grad_input"
      << (weight_grad ? ",dif_scalar* grad_weight" : "")
      << "){unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){unsigned long long row=i/" << dim
      << "ULL,d=i%" << dim << "ULL,rb=row*" << dim
      << "ULL,tb=(row/" << heads << "ULL)*" << table_width
      << "ULL;float ss=0.0f;for(unsigned long long k=0ULL;k<" << dim
      << "ULL;++k){float value=dif_load(x,rb+k);ss=fmaf(value,value,ss);}"
         "float inv=rsqrtf(ss/"
      << dim << ".0f+" << epsilon
      << "f);float dot=0.0f;for(unsigned long long k=0ULL;k<" << dim
      << "ULL;++k){float rotated_gradient=" << rotated("rb", "tb", "k")
      << ";dot=fmaf(rotated_gradient*dif_load(weight,k),dif_load(x,rb+k),"
         "dot);}float own_rotated="
      << rotated("rb", "tb", "d")
      << ";float value=dif_load(x,i);"
         "dif_store(grad_input,i,own_rotated*dif_load(weight,d)*inv-"
         "value*inv*inv*inv*dot/"
      << dim << ".0f);";
  if (weight_grad)
    out << "if(i<" << dim
        << "ULL){float acc=0.0f;for(unsigned long long r=0ULL;r<"
        << sequence * heads << "ULL;++r){unsigned long long rrb=r*" << dim
        << "ULL,rtb=(r/" << heads << "ULL)*" << table_width
        << "ULL;float rss=0.0f;for(unsigned long long k=0ULL;k<" << dim
        << "ULL;++k){float rv=dif_load(x,rrb+k);rss=fmaf(rv,rv,rss);}"
           "float rinv=rsqrtf(rss/"
        << dim << ".0f+" << epsilon << "f);float rotated_gradient="
        << rotated("rrb", "rtb", "i")
        << ";acc=fmaf(rotated_gradient*dif_load(x,rrb+i),rinv,acc);}"
           "dif_store(grad_weight,i,acc);}";
  out << "}}\n" << std::defaultfloat;
}

void emit_layer_norm_backward(std::ostringstream &out,
                              const ir::Program &program,
                              const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[1]);
  const auto columns = input->dims.back();
  const auto rows = input->element_count() / columns;
  const auto count = rows * columns;
  const auto epsilon = static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  // dx = inv*(gw - mean(gw) - xhat*mean(gw*xhat)), mean/inv recomputed from
  // the original input in F32 (flame's non-affine-LN cancellation lesson).
  out << std::scientific << std::setprecision(9)
      << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* grad_output,const dif_scalar* x,const dif_scalar* weight,dif_scalar* grad_input,dif_scalar* grad_weight,dif_scalar* grad_bias){"
         "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){unsigned long long row=i/" << columns
      << "ULL,base=row*" << columns
      << "ULL;float mean=0.0f;for(unsigned long long k=0ULL;k<" << columns
      << "ULL;++k)mean+=dif_load(x,base+k);mean/=" << columns
      << ".0f;float variance=0.0f;for(unsigned long long k=0ULL;k<" << columns
      << "ULL;++k){float centered=dif_load(x,base+k)-mean;"
         "variance=fmaf(centered,centered,variance);}float inv=rsqrtf(variance/"
      << columns << ".0f+" << epsilon
      << "f);float gradient_mean=0.0f,projected_mean=0.0f;"
         "for(unsigned long long k=0ULL;k<"
      << columns
      << "ULL;++k){float weighted=dif_load(grad_output,base+k)*dif_load(weight,k);"
         "float normalized=(dif_load(x,base+k)-mean)*inv;"
         "gradient_mean+=weighted;projected_mean=fmaf(weighted,normalized,"
         "projected_mean);}gradient_mean/="
      << columns << ".0f;projected_mean/=" << columns
      << ".0f;float upstream=dif_load(grad_output,i);"
         "float weighted=upstream*dif_load(weight,i-base);"
         "float normalized=(dif_load(x,i)-mean)*inv;"
         "dif_store(grad_input,i,inv*(weighted-gradient_mean-normalized*"
         "projected_mean));if(i<"
      << columns
      << "ULL){float weight_accumulator=0.0f,bias_accumulator=0.0f;"
         "for(unsigned long long r=0ULL;r<"
      << rows << "ULL;++r){unsigned long long rb=r*" << columns
      << "ULL;float rmean=0.0f;for(unsigned long long k=0ULL;k<" << columns
      << "ULL;++k)rmean+=dif_load(x,rb+k);rmean/=" << columns
      << ".0f;float rvariance=0.0f;for(unsigned long long k=0ULL;k<" << columns
      << "ULL;++k){float centered=dif_load(x,rb+k)-rmean;"
         "rvariance=fmaf(centered,centered,rvariance);}float rinv=rsqrtf("
         "rvariance/"
      << columns << ".0f+" << epsilon
      << "f);float g=dif_load(grad_output,rb+i);"
         "weight_accumulator=fmaf(g,(dif_load(x,rb+i)-rmean)*rinv,"
         "weight_accumulator);bias_accumulator+=g;}"
         "dif_store(grad_weight,i,weight_accumulator);"
         "dif_store(grad_bias,i,bias_accumulator);}}}\n"
      << std::defaultfloat;
}

void emit_residual_gate_backward(std::ostringstream &out,
                                 const ir::Program &program,
                                 const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* grad_output,const dif_scalar* branch,const dif_scalar* gate,dif_scalar* grad_branch,dif_scalar* grad_gate){"
         "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count
      << "ULL){float upstream=dif_load(grad_output,i);"
         "dif_store(grad_branch,i,upstream*dif_load(gate,i));"
         "dif_store(grad_gate,i,upstream*dif_load(branch,i));}}\n";
}

void emit_attention_lse(std::ostringstream &out, const ir::Program &program,
                        const ir::Operation &op) {
  const auto *q = program.tensor(op.inputs[0]);
  const auto sequence = q->dims[0];
  const auto heads = q->dims[1];
  const auto dim = q->dims[2];
  const auto count = sequence * heads;
  const auto scale = static_cast<float>(op.f64(
      ir::AttrKey::AttentionScale, 1.0 / std::sqrt(static_cast<double>(dim))));
  const bool causal = op.boolean(ir::AttrKey::Causal, false);
  const auto *scalar = typed_scalar(q->dtype);
  const auto *load = typed_load(q->dtype);
  out << std::scientific << std::setprecision(9)
      << "extern \"C\" __global__ void " << function_name(op) << "(const "
      << scalar << "* q,const " << scalar
      << "* k,dif_f32* lse){unsigned long long i=(unsigned long long)"
         "blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){unsigned long long qs=i/" << heads << "ULL,h=i%"
      << heads << "ULL,qb=(qs*" << heads << "ULL+h)*" << dim
      << "ULL,kend=" << (causal ? "qs+1ULL" : std::to_string(sequence) + "ULL")
      << ";float maximum=-3.402823466e+38f;"
         "for(unsigned long long ks=0ULL;ks<kend;++ks){float score=0.0f;"
         "unsigned long long kb=(ks*"
      << heads << "ULL+h)*" << dim
      << "ULL;for(unsigned long long d=0ULL;d<" << dim
      << "ULL;++d)score=fmaf(" << load << "(q,qb+d)," << load
      << "(k,kb+d),score);score*=" << scale
      << "f;maximum=fmaxf(maximum,score);}float denominator=0.0f;"
         "for(unsigned long long ks=0ULL;ks<kend;++ks){float score=0.0f;"
         "unsigned long long kb=(ks*"
      << heads << "ULL+h)*" << dim
      << "ULL;for(unsigned long long d=0ULL;d<" << dim
      << "ULL;++d)score=fmaf(" << load << "(q,qb+d)," << load
      << "(k,kb+d),score);denominator+=expf(score*" << scale
      << "f-maximum);}dif_store_f32(lse,i,maximum+logf(denominator));}}\n"
      << std::defaultfloat;
}

void emit_attention_backward(std::ostringstream &out,
                             const ir::Program &program,
                             const ir::Operation &op) {
  const auto *q = program.tensor(op.inputs[1]);
  const auto sequence = q->dims[0];
  const auto heads = q->dims[1];
  const auto dim = q->dims[2];
  const auto count = sequence * heads * dim;
  const auto scale = static_cast<float>(op.f64(
      ir::AttrKey::AttentionScale, 1.0 / std::sqrt(static_cast<double>(dim))));
  const bool causal = op.boolean(ir::AttrKey::Causal, false);
  const auto *scalar = typed_scalar(q->dtype);
  const auto *load = typed_load(q->dtype);
  const auto *store = typed_store(q->dtype);
  // Thread i owns element (s,h,d) of ALL THREE gradients: dq for row s as a
  // query, dk/dv for row s as a key.  P is recomputed from Q,K and the saved
  // F32 logsumexp; delta = rowsum(dO*O) uses the forward output.  O(S) score
  // recomputations per thread — the O(S^2) recompute path; acceptable at
  // gate scale, cuDNN SDPA backward is the Wave-3 replacement.
  out << std::scientific << std::setprecision(9)
      << "extern \"C\" __global__ void " << function_name(op) << "(const "
      << scalar << "* grad_output,const " << scalar << "* q,const " << scalar
      << "* k,const " << scalar << "* v,const " << scalar
      << "* forward_output,const dif_f32* lse," << scalar << "* grad_q,"
      << scalar << "* grad_k," << scalar
      << "* grad_v){unsigned long long i=(unsigned long long)blockIdx.x*"
         "blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){unsigned long long row=i/" << dim << "ULL,d=i%"
      << dim << "ULL,s=row/" << heads << "ULL,h=row%" << heads
      << "ULL,base=row*" << dim
      << "ULL;"
      // dq[s,h,d]: iterate keys ks<kend(s).
         "float dq=0.0f;{unsigned long long qb=base,kend="
      << (causal ? "s+1ULL" : std::to_string(sequence) + "ULL")
      << ";float row_lse=dif_load_f32(lse,row);float delta=0.0f;"
         "for(unsigned long long e=0ULL;e<"
      << dim << "ULL;++e)delta=fmaf(" << load << "(grad_output,qb+e)," << load
      << "(forward_output,qb+e),delta);"
         "for(unsigned long long ks=0ULL;ks<kend;++ks){unsigned long long kb="
         "(ks*"
      << heads << "ULL+h)*" << dim
      << "ULL;float score=0.0f,projected=0.0f;"
         "for(unsigned long long e=0ULL;e<"
      << dim << "ULL;++e){score=fmaf(" << load << "(q,qb+e)," << load
      << "(k,kb+e),score);projected=fmaf(" << load << "(grad_output,qb+e),"
      << load << "(v,kb+e),projected);}float probability=expf(score*" << scale
      << "f-row_lse);dq=fmaf(probability*(projected-delta)*" << scale
      << "f," << load << "(k,kb+d),dq);}}" << store
      << "(grad_q,i,dq);"
      // dk[s,h,d], dv[s,h,d]: iterate queries qs (qs>=s when causal).
         "float dk=0.0f,dv=0.0f;{unsigned long long kb=base;"
         "for(unsigned long long qs="
      << (causal ? "s" : "0ULL") << ";qs<" << sequence
      << "ULL;++qs){unsigned long long qb=(qs*" << heads << "ULL+h)*" << dim
      << "ULL;float row_lse=dif_load_f32(lse,qs*" << heads
      << "ULL+h);float score=0.0f,projected=0.0f,delta=0.0f;"
         "for(unsigned long long e=0ULL;e<"
      << dim << "ULL;++e){score=fmaf(" << load << "(q,qb+e)," << load
      << "(k,kb+e),score);projected=fmaf(" << load << "(grad_output,qb+e),"
      << load << "(v,kb+e),projected);delta=fmaf(" << load
      << "(grad_output,qb+e)," << load
      << "(forward_output,qb+e),delta);}float probability=expf(score*"
      << scale << "f-row_lse);dk=fmaf(probability*(projected-delta)*" << scale
      << "f," << load << "(q,qb+d),dk);dv=fmaf(probability," << load
      << "(grad_output,qb+d),dv);}}" << store << "(grad_k,i,dk);" << store
      << "(grad_v,i,dv);}}\n"
      << std::defaultfloat;
}

// ---------------------------------------------------------------------------
// Elementwise region fusion (opt-in candidate property).
//
// A region is a single-consumer tree of pointwise operations {Add, Multiply,
// SiLU, Clamp, Cast, BiasAdd, ResidualGate, SwiGlu} collapsed into ONE kernel
// launched at the region's terminal ("anchor") operation; every interior
// operation is subsumed via skipped_operations and its output never touches
// global memory.  Fusion is a deliberate candidate property, not silent
// global behavior: an operation participates only when it carries
// Implementation=2 (stamped by `difc set-elementwise-fusion`, a fingerprinted
// transform).  Unstamped programs emit byte-identically to the pre-fusion
// compiler.
//
// Numerics contract: BYTE-IDENTICAL to unfused execution.  Each stage is
// computed in F32 registers and rounded to that intermediate's storage dtype
// in-register (dif_round_*) exactly where the unfused kernel rounds at its
// store.  Cross-boundary FMA contraction is prevented where it can occur --
// a mul-topped stage (Multiply, SwiGlu) feeding a consumer's add -- by
// emitting that top-level multiply as __fmul_rn, which is bit-identical to
// the lone rn-rounded FMUL of the unfused kernel and is never contracted
// (an identity mul.rn barrier was algebraically eliminated by ptxas and an
// empty asm barrier only stops front-end contraction; both measured by the
// multiply->add byte gate).  Intra-stage expression trees are structural
// copies of the unfused emitters, so intra-stage contraction stays
// identical on both sides.
//
// Memory-safety contract (the planner places tensors at slot base and
// best-fit-reuses non-dedicated slots by liveness): an external input still
// live at the anchor's position -- or holding a dedicated slot (Input,
// Output, resident Constant) -- can never alias the anchor's output.  An
// input that dies inside the region is admitted only when the program has no
// streamed constants, the region is contiguous in program order, and one of:
// (a) its aligned size is smaller than the output's (the planner cannot hand
// its slot to the output), (b) the anchor output is dedicated (nothing
// writes the dead slot before the anchor), or (c) every region read of it is
// at the thread's own output element index with equal element count and
// dtype size (per-thread read-before-write on identical bytes).  Anything
// else fails safe to per-op emission.
//
// Launch-geometry contract: the executor derives geometry from the anchor's
// opcode, so the anchor must be an element-per-thread pointwise operation.
// This is why the RmsNorm family cannot HEAD a fused region here: its
// one-block-per-row reduction structure is unreachable from a pointwise
// anchor without O(columns) redundant work per thread, so such candidates
// fail safe to per-op emission (recorded design boundary).
// ---------------------------------------------------------------------------

constexpr std::uint64_t kElementwiseFusionOptIn = 2U;

bool elementwise_fusable(ir::Opcode opcode) {
  switch (opcode) {
  case ir::Opcode::Add:
  case ir::Opcode::Multiply:
  case ir::Opcode::SiLU:
  case ir::Opcode::Clamp:
  case ir::Opcode::Cast:
  case ir::Opcode::BiasAdd:
  case ir::Opcode::ResidualGate:
  case ir::Opcode::SwiGlu:
    return true;
  default:
    return false;
  }
}

// Whether the operation reads this input slot at the same element index it
// writes its output (SwiGlu remaps rows/columns; BiasAdd's bias broadcasts).
bool elementwise_identity_slot(ir::Opcode opcode, std::size_t input_slot) {
  if (opcode == ir::Opcode::SwiGlu)
    return false;
  if (opcode == ir::Opcode::BiasAdd)
    return input_slot == 0U;
  return true;
}

struct ElementwiseRegion {
  const ir::Operation *anchor{};
  std::vector<const ir::Operation *> members; // program order, anchor last
  std::unordered_map<std::uint32_t, const ir::Operation *> produced_by;
};

std::unordered_map<std::uint32_t, ElementwiseRegion>
find_elementwise_regions(const ir::Program &program,
                         std::unordered_set<std::uint32_t> &skipped) {
  std::unordered_map<std::uint32_t, const ir::Operation *> producer;
  std::unordered_map<std::uint32_t, std::vector<const ir::Operation *>>
      consumers;
  std::unordered_map<std::uint32_t, std::uint64_t> position;
  bool streamed_program = false;
  for (const auto &tensor : program.tensors)
    if (tensor.has_role(ir::TensorRole::Streamed))
      streamed_program = true;
  for (std::uint64_t index = 0; index < program.operations.size(); ++index) {
    const auto &operation =
        program.operations[static_cast<std::size_t>(index)];
    position.emplace(operation.id, index);
    for (const auto output : operation.outputs)
      producer.emplace(output, &operation);
    for (const auto input : operation.inputs)
      consumers[input].push_back(&operation);
  }
  const auto eligible = [&](const ir::Operation &operation) {
    return elementwise_fusable(operation.opcode) &&
           operation.u64(ir::AttrKey::Implementation, 1U) ==
               kElementwiseFusionOptIn &&
           !skipped.contains(operation.id) && operation.outputs.size() == 1U;
  };
  // Merge edge P -> C: P's single output is a role-free internal value with
  // exactly one consumer, and both operations opted in.
  std::unordered_map<std::uint32_t, const ir::Operation *> merge_consumer;
  for (const auto &operation : program.operations) {
    if (!eligible(operation))
      continue;
    const auto output = operation.outputs[0];
    const auto &uses = consumers[output];
    if (program.tensor(output)->roles != 0U || uses.size() != 1U ||
        !eligible(*uses.front()))
      continue;
    merge_consumer.emplace(operation.id, uses.front());
  }
  std::unordered_map<std::uint32_t, std::vector<const ir::Operation *>> groups;
  for (const auto &operation : program.operations) {
    if (!eligible(operation))
      continue;
    const auto *terminal = &operation;
    while (true) {
      const auto next = merge_consumer.find(terminal->id);
      if (next == merge_consumer.end())
        break;
      terminal = next->second;
    }
    groups[terminal->id].push_back(&operation);
  }

  std::unordered_map<std::uint32_t, ElementwiseRegion> regions;
  // Deterministic acceptance order: iterate terminals in program order.
  std::vector<std::uint32_t> terminals;
  terminals.reserve(groups.size());
  for (const auto &[terminal_id, members] : groups)
    if (members.size() >= 2U)
      terminals.push_back(terminal_id);
  std::sort(terminals.begin(), terminals.end(),
            [&](std::uint32_t a, std::uint32_t b) {
              return position.at(a) < position.at(b);
            });
  for (const auto terminal_id : terminals) {
    auto members = groups.at(terminal_id);
    std::sort(members.begin(), members.end(),
              [&](const ir::Operation *a, const ir::Operation *b) {
                return position.at(a->id) < position.at(b->id);
              });
    const auto *anchor = members.back();
    if (anchor->id != terminal_id)
      continue; // defensive: the terminal must be last in program order
    const auto anchor_position = position.at(anchor->id);
    const auto *anchor_output = program.tensor(anchor->outputs[0]);
    std::unordered_set<std::uint32_t> in_region;
    for (const auto *member : members)
      in_region.insert(member->id);
    // Identity-index propagation from the anchor toward producers.
    std::unordered_map<std::uint32_t, bool> identity;
    identity.emplace(anchor->id, true);
    for (auto it = members.rbegin(); it != members.rend(); ++it) {
      const auto *member = *it;
      const auto member_identity = identity.at(member->id);
      for (std::size_t slot = 0; slot < member->inputs.size(); ++slot) {
        const auto found = producer.find(member->inputs[slot]);
        if (found == producer.end() ||
            !in_region.contains(found->second->id))
          continue;
        identity.emplace(found->second->id,
                         member_identity &&
                             elementwise_identity_slot(member->opcode, slot));
      }
    }
    // External inputs and how they are read.
    std::vector<std::uint32_t> external_order;
    std::unordered_map<std::uint32_t, bool> external_identity;
    bool eligible_region = true;
    for (const auto *member : members) {
      for (std::size_t slot = 0; slot < member->inputs.size(); ++slot) {
        const auto input = member->inputs[slot];
        const auto found = producer.find(input);
        if (found != producer.end() &&
            in_region.contains(found->second->id))
          continue;
        if (found != producer.end() && skipped.contains(found->second->id)) {
          eligible_region = false; // produced by another lowering's interior
          break;
        }
        const bool read_identity =
            identity.at(member->id) &&
            elementwise_identity_slot(member->opcode, slot);
        const auto emplaced = external_identity.emplace(input, read_identity);
        if (emplaced.second)
          external_order.push_back(input);
        else
          emplaced.first->second = emplaced.first->second && read_identity;
      }
      if (!eligible_region)
        break;
    }
    // The executor marshals launch arguments into a fixed 16-pointer array
    // (launch_inputs plus the anchor's one output).
    if (!eligible_region || external_order.size() > 15U)
      continue;
    const auto dedicated = [](const ir::TensorDesc &tensor) {
      return tensor.has_role(ir::TensorRole::Input) ||
             tensor.has_role(ir::TensorRole::Output) ||
             (tensor.has_role(ir::TensorRole::Constant) &&
              !tensor.has_role(ir::TensorRole::Streamed));
    };
    const auto align256 = [](std::uint64_t bytes) {
      return (bytes + 255U) & ~static_cast<std::uint64_t>(255U);
    };
    const bool contiguous = anchor_position -
                                position.at(members.front()->id) + 1U ==
                            members.size();
    for (const auto input : external_order) {
      const auto *description = program.tensor(input);
      if (dedicated(*description))
        continue;
      std::uint64_t last_use = 0U;
      for (const auto *use : consumers[input])
        last_use = std::max(last_use, position.at(use->id));
      if (last_use >= anchor_position)
        continue; // live at the anchor: the planner keeps its slot
      if (streamed_program || !contiguous) {
        eligible_region = false;
        break;
      }
      if (align256(description->byte_count()) <
          align256(anchor_output->byte_count()))
        continue; // slot too small for the planner to hand to the output
      if (dedicated(*anchor_output))
        continue; // nothing writes the dead slot before the anchor
      if (external_identity.at(input) &&
          description->element_count() == anchor_output->element_count() &&
          ir::dtype_size(description->dtype) ==
              ir::dtype_size(anchor_output->dtype))
        continue; // per-thread read-before-write on identical bytes
      eligible_region = false;
      break;
    }
    if (!eligible_region)
      continue;
    ElementwiseRegion region;
    region.anchor = anchor;
    region.members = members;
    for (const auto *member : members) {
      if (member == anchor)
        continue;
      region.produced_by.emplace(member->outputs[0], member);
      skipped.insert(member->id);
    }
    regions.emplace(anchor->id, std::move(region));
  }
  return regions;
}

const char *typed_round(ir::DType dtype) {
  if (dtype == ir::DType::F32)
    return "dif_round_f32";
  if (dtype == ir::DType::BF16)
    return "dif_round_bf16";
  if (dtype == ir::DType::F16)
    return "dif_round_f16";
  fail("CUDA elementwise fusion admits f32, bf16, or f16 storage");
}

std::string formatted_float(double value) {
  std::ostringstream text;
  text << std::scientific << std::setprecision(9)
       << static_cast<float>(value) << "f";
  return text.str();
}

struct ElementwiseFusionEmitter {
  ElementwiseFusionEmitter(const ir::Program &program_reference,
                           const ElementwiseRegion &region_reference)
      : program(program_reference), region(region_reference) {}

  const ir::Program &program;
  const ElementwiseRegion &region;
  std::ostringstream body;
  std::vector<std::uint32_t> arguments;
  std::unordered_map<std::uint32_t, std::size_t> argument_index;
  std::unordered_map<std::string, std::string> stage_variables;
  std::unordered_map<std::string, std::pair<std::string, std::string>>
      row_variables;
  std::size_t next_variable{};

  std::string argument(std::uint32_t tensor) {
    auto found = argument_index.find(tensor);
    if (found == argument_index.end()) {
      found = argument_index.emplace(tensor, arguments.size()).first;
      arguments.push_back(tensor);
    }
    return "a" + std::to_string(found->second);
  }

  std::string value(std::uint32_t tensor, const std::string &index) {
    const auto member = region.produced_by.find(tensor);
    if (member == region.produced_by.end())
      return std::string(typed_load(program.tensor(tensor)->dtype)) + "(" +
             argument(tensor) + "," + index + ")";
    return stage(*member->second, index);
  }

  // A rounded, barriered stage value: the in-register equivalent of the
  // unfused kernel's store followed by the consumer's load.  The barrier
  // pins the rounded value so FMA contraction cannot cross the boundary.
  std::string stage(const ir::Operation &op, const std::string &index) {
    const auto key = std::to_string(op.id) + "|" + index;
    const auto found = stage_variables.find(key);
    if (found != stage_variables.end())
      return found->second;
    const auto expression = stage_expression(op, index);
    const auto name = "s" + std::to_string(next_variable++);
    body << "float " << name << "="
         << typed_round(program.tensor(op.outputs[0])->dtype) << "("
         << expression << ");asm(\"\" : \"+f\"(" << name << "));";
    stage_variables.emplace(key, name);
    return name;
  }

  std::pair<std::string, std::string> row_column(const ir::Operation &op,
                                                 const std::string &index,
                                                 std::uint64_t width) {
    const auto key = std::to_string(op.id) + "|" + index;
    auto found = row_variables.find(key);
    if (found == row_variables.end()) {
      const auto suffix = std::to_string(next_variable++);
      const auto row = "r" + suffix;
      const auto column = "c" + suffix;
      body << "unsigned long long " << row << "=(" << index << ")/" << width
           << "ULL," << column << "=(" << index << ")%" << width << "ULL;";
      found = row_variables.emplace(key, std::make_pair(row, column)).first;
    }
    return found->second;
  }

  // The unrounded value the unfused kernel would pass to dif_store,
  // structurally mirroring each per-op emitter (same operand order, same
  // interior dif_round placement) so intra-stage codegen matches.  Operands
  // are evaluated into locals in input-slot order so argument registration
  // (and therefore launch_inputs and the generated source) is deterministic.
  //
  // Boundary-contraction rule: FMA contraction merges a multiply into the
  // add that consumes it.  A stage whose TOP-LEVEL operation is a bare
  // multiply (Multiply, SwiGlu) is a lone rn-rounded FMUL in the unfused
  // kernel, so it is emitted through __fmul_rn -- bit-identical to that
  // FMUL and guaranteed never contracted into the consumer's add (ptxas
  // eliminated a mul.rn-by-1.0 barrier and contracted anyway; measured by
  // the multiply->add byte gate).  Add-topped stages cannot fuse outward
  // (there is no add->mul or add->add fusion), and interior expression
  // trees are left verbatim so intra-stage contraction stays identical to
  // the unfused kernels (e.g. ResidualGate's gate*branch may fuse with its
  // own residual add on BOTH sides).
  std::string stage_expression(const ir::Operation &op,
                               const std::string &index) {
    const auto *output = program.tensor(op.outputs[0]);
    switch (op.opcode) {
    case ir::Opcode::Add: {
      const auto left = value(op.inputs[0], index);
      const auto right = value(op.inputs[1], index);
      return left + "+" + right;
    }
    case ir::Opcode::Multiply: {
      const auto left = value(op.inputs[0], index);
      const auto right = value(op.inputs[1], index);
      return "__fmul_rn(" + left + "," + right + ")";
    }
    case ir::Opcode::SiLU:
      return "dif_silu(" + value(op.inputs[0], index) + ")";
    case ir::Opcode::Cast:
      return value(op.inputs[0], index);
    case ir::Opcode::Clamp:
      return "fminf(" + formatted_float(op.f64(ir::AttrKey::Upper, 1.0)) +
             ",fmaxf(" + formatted_float(op.f64(ir::AttrKey::Lower, 0.0)) +
             "," + value(op.inputs[0], index) + "))";
    case ir::Opcode::BiasAdd: {
      const auto width = program.tensor(op.inputs[1])->dims[0];
      const auto left = value(op.inputs[0], index);
      const auto right =
          value(op.inputs[1], "(" + index + ")%" + std::to_string(width) +
                                  "ULL");
      return left + "+" + right;
    }
    case ir::Opcode::ResidualGate: {
      const auto residual = value(op.inputs[0], index);
      const auto branch = value(op.inputs[1], index);
      const auto gate = value(op.inputs[2], index);
      return residual + "+" + std::string(typed_round(output->dtype)) + "(" +
             gate + "*" + branch + ")";
    }
    case ir::Opcode::SwiGlu: {
      const auto width = output->dims.back();
      const bool gate_first = op.boolean(ir::AttrKey::GateFirst, false);
      const auto [row, column] = row_column(op, index, width);
      const auto lane = [&](std::uint64_t offset) {
        return row + "*" + std::to_string(width * 2U) + "ULL+" +
               std::to_string(offset) + "ULL+" + column;
      };
      const auto value_lane =
          value(op.inputs[0], lane(gate_first ? width : 0U));
      const auto gate_lane =
          value(op.inputs[0], lane(gate_first ? 0U : width));
      return "__fmul_rn(" + value_lane + "," +
             std::string(typed_round(output->dtype)) + "(dif_silu(" +
             gate_lane + ")))";
    }
    default:
      fail("elementwise fusion emitter received an unsupported opcode");
    }
  }
};

void emit_fused_elementwise(std::ostringstream &out,
                            const ir::Program &program,
                            const ElementwiseRegion &region,
                            std::vector<std::uint32_t> &launch_arguments) {
  ElementwiseFusionEmitter emitter(program, region);
  const auto *output = program.tensor(region.anchor->outputs[0]);
  const auto terminal = emitter.stage_expression(*region.anchor, "i");
  out << "extern \"C\" __global__ void " << function_name(*region.anchor)
      << "(";
  for (std::size_t argument = 0; argument < emitter.arguments.size();
       ++argument)
    out << "const "
        << typed_scalar(program.tensor(emitter.arguments[argument])->dtype)
        << "* a" << argument << ",";
  out << typed_scalar(output->dtype)
      << "* y){unsigned long long i=(unsigned long long)blockIdx.x*"
         "blockDim.x+threadIdx.x;if(i<"
      << output->element_count() << "ULL){" << emitter.body.str()
      << typed_store(output->dtype) << "(y,i," << terminal << ");}}\n";
  launch_arguments = emitter.arguments;
}

} // namespace

GeneratedCuda emit_cuda(const ir::Program &program) {
  ir::verify(program);
  GeneratedCuda generated;
  std::ostringstream source;
  for (const auto &tensor : program.tensors) {
    if (tensor.dtype != ir::DType::F32 && tensor.dtype != ir::DType::BF16 &&
        tensor.dtype != ir::DType::F16 &&
        tensor.dtype != ir::DType::I32 && tensor.dtype != ir::DType::I8)
      fail("CUDA source emitter admits mixed f32/bf16/f16 plus i32 indices "
           "and packed i8 constants");
  }
  emit_header(source);
  auto fusions = find_lowbit_linear_fusions(program, generated.skipped_operations);
  const auto elementwise_regions =
      find_elementwise_regions(program, generated.skipped_operations);
  for (const auto &op : program.operations) {
    if (generated.skipped_operations.contains(op.id))
      continue;
    if (op.opcode == ir::Opcode::Linear) {
      const auto found = fusions.find(op.id);
      if (found != fusions.end()) {
        generated.entrypoints.emplace(op.id, function_name(op));
        auto inputs = std::vector<std::uint32_t>{op.inputs[0]};
        inputs.insert(inputs.end(), found->second.dequant->inputs.begin(),
                      found->second.dequant->inputs.end());
        if (op.inputs.size() == 3U)
          inputs.push_back(op.inputs[2]);
        generated.launch_inputs.emplace(op.id, std::move(inputs));
        begin_float_operation(source, operation_float_dtype(program, op));
        emit_lowbit_linear(source, program, found->second);
        end_float_operation(source);
      } else if (op.u64(ir::AttrKey::Implementation, 1U) == 3U) {
        fail("direct packed INT5 Linear candidate is not an eligible exclusive "
             "dequantization chain");
      }
      continue;
    }
    if (op.opcode == ir::Opcode::Barrier ||
        (op.opcode == ir::Opcode::Attention &&
         op.u64(ir::AttrKey::Implementation, 1U) == 2U))
      continue;
    if (const auto region = elementwise_regions.find(op.id);
        region != elementwise_regions.end()) {
      generated.entrypoints.emplace(op.id, function_name(op));
      std::vector<std::uint32_t> arguments;
      emit_fused_elementwise(source, program, region->second, arguments);
      generated.launch_inputs.emplace(op.id, std::move(arguments));
      continue;
    }
    generated.entrypoints.emplace(op.id, function_name(op));
    if (op.opcode == ir::Opcode::Cast) {
      emit_cast(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::RotaryPosition) {
      emit_rotary_position(source, program, op);
      continue;
    }
    begin_float_operation(source, operation_float_dtype(program, op));
    switch (op.opcode) {
    case ir::Opcode::Add:
      emit_elementwise(source, program, op, "dif_load(a,i)+dif_load(b,i)");
      break;
    case ir::Opcode::Multiply:
      emit_elementwise(source, program, op, "dif_load(a,i)*dif_load(b,i)");
      break;
    case ir::Opcode::AffineLastDim:
      emit_affine_last_dim(source, program, op);
      break;
    case ir::Opcode::SiLU:
      emit_silu(source, program, op);
      break;
    case ir::Opcode::RmsNorm:
      emit_rms_norm(source, program, op);
      break;
    case ir::Opcode::LayerNorm:
      emit_layer_norm(source, program, op);
      break;
    case ir::Opcode::Clamp:
      emit_clamp(source, program, op);
      break;
    case ir::Opcode::MseLoss:
      emit_mse_loss(source, program, op);
      break;
    case ir::Opcode::MseLossBackward:
      emit_mse_loss_backward(source, program, op);
      break;
    case ir::Opcode::LinearBackwardInput:
      emit_linear_backward_input(source, program, op);
      break;
    case ir::Opcode::LinearBackwardWeight:
      emit_linear_backward_weight(source, program, op);
      break;
    case ir::Opcode::BiasBackward:
      emit_bias_backward(source, program, op);
      break;
    case ir::Opcode::SiLUBackward:
      emit_silu_backward(source, program, op);
      break;
    case ir::Opcode::AdamWUpdate:
      emit_adamw_update(source, program, op);
      break;
    case ir::Opcode::Fill:
      emit_fill(source, program, op);
      break;
    case ir::Opcode::GatherRows:
      emit_gather_rows(source, program, op);
      break;
    case ir::Opcode::IndexedUpdateRows:
      emit_indexed_update_rows(source, program, op);
      break;
    case ir::Opcode::Cast:
      break;
    case ir::Opcode::SelectRowChunks:
      emit_select_row_chunks(source, program, op);
      break;
    case ir::Opcode::SinusoidalTimestep:
      emit_sinusoidal_timestep(source, program, op);
      break;
    case ir::Opcode::RotaryPosition:
      break;
    case ir::Opcode::LinearBlend:
      emit_linear_blend(source, program, op);
      break;
    case ir::Opcode::FlowEulerStep:
      emit_flow_euler_step(source, program, op);
      break;
    case ir::Opcode::Patchify3D:
      emit_patchify_3d(source, program, op, false);
      break;
    case ir::Opcode::Unpatchify3D:
      emit_patchify_3d(source, program, op, true);
      break;
    case ir::Opcode::RmsNormModulate:
      emit_rms_norm_modulate(source, program, op);
      break;
    case ir::Opcode::SwiGlu:
      emit_swiglu(source, program, op);
      break;
    case ir::Opcode::ResidualGate:
      emit_residual_gate(source, program, op);
      break;
    case ir::Opcode::Linear:
      break;
    case ir::Opcode::QkNormPartialRope:
      emit_qk_norm_rope(source, program, op);
      break;
    case ir::Opcode::Attention:
      emit_attention(source, program, op);
      break;
    case ir::Opcode::Barrier:
      break;
    case ir::Opcode::BiasAdd:
      emit_bias_add(source, program, op);
      break;
    case ir::Opcode::H3AdaLNSelect:
      emit_h3_adaln_select(source, program, op);
      break;
    case ir::Opcode::H3DeinterleaveQkv:
      emit_h3_deinterleave_qkv(source, program, op);
      break;
    case ir::Opcode::H3DeinterleaveQkvWeight:
      emit_h3_deinterleave_qkv_weight(source, program, op);
      break;
    case ir::Opcode::DequantizeInt4:
      emit_dequantize_int4(source, program, op);
      break;
    case ir::Opcode::DequantizeInt5:
      emit_dequantize_int5(source, program, op);
      break;
    case ir::Opcode::RmsNormBackward:
      emit_rms_norm_backward(source, program, op);
      break;
    case ir::Opcode::RmsNormModulateBackward:
      emit_rms_norm_modulate_backward(source, program, op);
      break;
    case ir::Opcode::SwiGluBackward:
      emit_swiglu_backward(source, program, op);
      break;
    case ir::Opcode::ResidualGateBackward:
      emit_residual_gate_backward(source, program, op);
      break;
    case ir::Opcode::LayerNormBackward:
      emit_layer_norm_backward(source, program, op);
      break;
    case ir::Opcode::QkNormPartialRopeBackward:
      emit_qk_norm_rope_backward(source, program, op);
      break;
    case ir::Opcode::AttentionLse:
      emit_attention_lse(source, program, op);
      break;
    case ir::Opcode::AttentionBackward:
      emit_attention_backward(source, program, op);
      break;
    case ir::Opcode::Conv1d:
    case ir::Opcode::SnakeBeta:
      // Integrator-approved fail-closed stub (chunk 6 of the audio decode
      // plan lands the real emitters here).
      fail("audio opcodes have no CUDA emitter yet");
      break;
    }
    end_float_operation(source);
  }
  generated.source = source.str();
  return generated;
}

ElementwiseFusionCensus
census_elementwise_fusion(const ir::Program &program) {
  ir::verify(program);
  std::unordered_set<std::uint32_t> skipped;
  find_lowbit_linear_fusions(program, skipped);
  const auto regions = find_elementwise_regions(program, skipped);
  ElementwiseFusionCensus census;
  census.regions = regions.size();
  for (const auto &[anchor, region] : regions)
    census.fused_operations += region.members.size();
  census.eliminated_launches = census.fused_operations - census.regions;
  return census;
}

} // namespace dif::compiler
