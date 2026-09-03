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

// Generated kernels are functions of opcode, tensor geometry/dtypes, and
// attributes -- never tensor or operation identity. Repeated model blocks and
// unrolled schedules therefore share one compiled entrypoint. The executor
// still keeps an operation-id -> entrypoint map, so semantic provenance and
// launch arguments remain per operation while NVRTC sees only unique code.
std::string kernel_identity(const ir::Program &program,
                            const ir::Operation &operation) {
  std::ostringstream key;
  key << static_cast<std::uint32_t>(operation.opcode) << '|';
  const auto tensor = [&](std::uint32_t id) {
    const auto *description = program.tensor(id);
    if (!description)
      fail("CUDA kernel identity references a missing tensor");
    key << static_cast<std::uint32_t>(description->dtype) << ':';
    for (const auto extent : description->dims)
      key << extent << ',';
    key << ';';
  };
  key << 'i' << operation.inputs.size() << ':';
  for (const auto id : operation.inputs)
    tensor(id);
  key << 'o' << operation.outputs.size() << ':';
  for (const auto id : operation.outputs)
    tensor(id);
  key << 'a' << operation.attributes.size() << ':';
  for (const auto &attribute : operation.attributes)
    key << static_cast<std::uint32_t>(attribute.key) << ','
        << static_cast<std::uint32_t>(attribute.kind) << ','
        << attribute.bits << ';';
  return std::move(key).str();
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


// BigVGAN audio decode emitters.
// Contract: generic launch — one thread per output element, geometry baked
// as literals, F32 accumulation, dif_load/dif_store storage boundary. The
// loop order mirrors the CPU reference exactly (forward: in-channel outer,
// kernel-tap inner; transposed: gather with the padded input index
// ascending — the same per-output add order as the CPU scatter), so
// CPU-vs-CUDA differences stay at FP-contraction level.
void emit_conv1d(std::ostringstream &out, const ir::Program &program,
                 const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *weight = program.tensor(op.inputs[1]);
  const auto *output = program.tensor(op.outputs[0]);
  const bool biased = op.inputs.size() == 3U;
  const auto stride = op.u64(ir::AttrKey::Stride, 1U);
  const auto dilation = op.u64(ir::AttrKey::Dilation, 1U);
  const auto groups = op.u64(ir::AttrKey::Groups, 1U);
  const auto pad_left = op.u64(ir::AttrKey::PadLeft, 0U);
  const auto pad_right = op.u64(ir::AttrKey::PadRight, 0U);
  const auto replicate = op.u64(ir::AttrKey::PadMode, 0U) == 1U;
  const auto transposed = op.boolean(ir::AttrKey::Transposed, false);
  const auto trim_left = op.u64(ir::AttrKey::TrimLeft, 0U);
  const auto in_channels = input->dims[1];
  const auto length = input->dims[2];
  const auto kernel = weight->dims[2];
  const auto out_channels = output->dims[1];
  const auto out_length = output->dims[2];
  const auto count = output->element_count();
  const auto in_per_group = in_channels / groups;
  const auto out_per_group = out_channels / groups;
  const auto padded = length + pad_left + pad_right;

  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,const dif_scalar* w,"
      << (biased ? "const dif_scalar* bias," : "") << "dif_scalar* y){"
      << "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+"
         "threadIdx.x;if(i<" << count << "ULL){"
      << "unsigned long long o=i%" << out_length << "ULL;"
      << "unsigned long long oc=(i/" << out_length << "ULL)%" << out_channels
      << "ULL;"
      << "unsigned long long b=i/" << out_channels * out_length << "ULL;"
      << "unsigned long long group=oc/" << out_per_group << "ULL;"
      << "float acc=0.0f;";
  // Padded-coordinate sampler: position maps to input index pos-PadLeft,
  // replicate-clamped or zero outside; emitted as a literal expression.
  const auto clamped_sample = [&](const std::string &position) {
    const auto shifted = "(" + position + "-" +
                         std::to_string(pad_left) + "LL)";
    if (!replicate)
      return "((" + shifted + ">=0LL&&" + shifted + "<" +
             std::to_string(length) +
             "LL)?dif_load(xrow,(unsigned long long)" + shifted + "):0.0f)";
    const auto last = std::to_string(length - 1U) + "LL";
    return "dif_load(xrow,(unsigned long long)(" + shifted + "<0LL?0LL:(" +
           shifted + ">" + last + "?" + last + ":" + shifted + ")))";
  };
  const std::string forward_sample = clamped_sample("p");
  const std::string gather_sample = clamped_sample("pi");
  if (!transposed) {
    out << "for(unsigned long long ic=0;ic<" << in_per_group << "ULL;++ic){"
        << "const dif_scalar* xrow=x+((b*" << in_channels
        << "ULL)+(group*" << in_per_group << "ULL+ic))*" << length << "ULL;"
        << "const dif_scalar* wrow=w+((oc*" << in_per_group << "ULL)+ic)*"
        << kernel << "ULL;"
        << "long long start=(long long)(o*" << stride << "ULL);"
        << "for(unsigned long long k=0;k<" << kernel << "ULL;++k){"
        << "long long p=start+(long long)(k*" << dilation << "ULL);"
        << "acc+=" << forward_sample << "*dif_load(wrow,k);}}";
  } else {
    out << "unsigned long long ocg=oc%" << out_per_group << "ULL;"
        << "long long ofull=(long long)o+" << trim_left << "LL;"
        << "for(unsigned long long ic=0;ic<" << in_per_group << "ULL;++ic){"
        << "const dif_scalar* xrow=x+((b*" << in_channels
        << "ULL)+(group*" << in_per_group << "ULL+ic))*" << length << "ULL;"
        << "const dif_scalar* wrow=w+(((group*" << in_per_group
        << "ULL+ic)*" << out_per_group << "ULL)+ocg)*" << kernel << "ULL;"
        << "long long imin=(ofull-" << kernel - 1U << "LL+" << stride
        << "LL-1LL)/" << stride << "LL;if(imin<0LL)imin=0LL;"
        << "long long imax=ofull/" << stride << "LL;if(imax>"
        << padded - 1U << "LL)imax=" << padded - 1U << "LL;"
        << "for(long long pi=imin;pi<=imax;++pi){"
        << "long long k=ofull-pi*" << stride << "LL;"
        << "if(k<0LL||k>=" << kernel << "LL)continue;"
        << "acc+=" << gather_sample << "*dif_load(wrow,(unsigned long long)k);}}";
  }
  out << (biased ? "acc+=dif_load(bias,oc);" : "")
      << "dif_store(y,i,acc);}}\n";
}

void emit_channel_rms_norm(std::ostringstream &out,
                           const ir::Program &program,
                           const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto axis = op.u64(ir::AttrKey::Axis, 1U);
  const auto channels = input->dims[axis];
  std::uint64_t inner = 1U;
  for (std::size_t index = static_cast<std::size_t>(axis + 1U);
       index < input->dims.size(); ++index)
    inner *= input->dims[index];
  const auto vectors = input->element_count() / channels;
  const auto block = op.u64(ir::AttrKey::BlockSize, 256U);
  const auto epsilon = static_cast<float>(
      op.f64(ir::AttrKey::Epsilon, 1.0e-12));
  const auto scale = std::sqrt(static_cast<float>(channels));
  out << std::scientific << std::setprecision(9)
      << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,const dif_scalar* gamma,dif_scalar* y){"
         "extern __shared__ float reduction[];unsigned long long vector="
         "(unsigned long long)blockIdx.x;if(vector>="
      << vectors << "ULL)return;unsigned long long c=threadIdx.x;"
      << "unsigned long long leading=vector/" << inner
      << "ULL,trailing=vector%" << inner
      << "ULL;unsigned long long index=(leading*" << channels
      << "ULL+c)*" << inner
      << "ULL+trailing;float value=c<" << channels
      << "ULL?dif_load(x,index):0.0f;reduction[c]=value*value;"
         "__syncthreads();";
  for (auto stride = block / 2U; stride != 0U; stride /= 2U)
    out << "if(c<" << stride << "ULL)reduction[c]+=reduction[c+" << stride
        << "ULL];__syncthreads();";
  out << "if(c<" << channels
      << "ULL){float denominator=fmaxf(sqrtf(reduction[0])," << epsilon
      << "f);float normalized=dif_round(value/denominator);"
         "float scaled=dif_round(normalized*"
      << scale
      << "f);dif_store(y,index,scaled*dif_load(gamma,c));}}\n"
      << std::defaultfloat;
}

void emit_group_norm(std::ostringstream &out, const ir::Program &program,
                     const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto channels = input->dims[1];
  const auto groups = op.u64(ir::AttrKey::Groups, 1U);
  const auto channels_per_group = channels / groups;
  std::uint64_t inner = 1U;
  for (std::size_t axis = 2U; axis < input->dims.size(); ++axis)
    inner *= input->dims[axis];
  const auto elements = channels_per_group * inner;
  const auto block = op.u64(ir::AttrKey::BlockSize, 256U);
  const auto epsilon =
      static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  out << std::scientific << std::setprecision(9)
      << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,const dif_scalar* weight,const dif_scalar* bias,dif_scalar* y){"
         "extern __shared__ float reduction[];unsigned long long vector="
         "(unsigned long long)blockIdx.x;if(vector>="
      << input->dims[0] * groups
      << "ULL)return;unsigned long long lane=threadIdx.x,group=vector%"
      << groups << "ULL,batch=vector/" << groups
      << "ULL,base=(batch*" << channels << "ULL+group*"
      << channels_per_group << "ULL)*" << inner
      << "ULL;float sum=0.0f,squares=0.0f;for(unsigned long long k=lane;k<"
      << elements
      << "ULL;k+=blockDim.x){float v=dif_load(x,base+k);sum+=v;squares=fmaf(v,v,squares);}"
         "reduction[lane]=sum;reduction[blockDim.x+lane]=squares;__syncthreads();";
  for (auto stride = block / 2U; stride != 0U; stride /= 2U)
    out << "if(lane<" << stride
        << "ULL){reduction[lane]+=reduction[lane+" << stride
        << "ULL];reduction[blockDim.x+lane]+=reduction[blockDim.x+lane+"
        << stride << "ULL];}__syncthreads();";
  out << "float mean=reduction[0]/" << elements
      << ".0f;float variance=fmaxf(reduction[blockDim.x]/" << elements
      << ".0f-mean*mean,0.0f);float inv=rsqrtf(variance+" << epsilon
      << "f);for(unsigned long long k=lane;k<" << elements
      << "ULL;k+=blockDim.x){unsigned long long channel=group*"
      << channels_per_group << "ULL+k/" << inner
      << "ULL;float normalized=(dif_load(x,base+k)-mean)*inv;"
         "dif_store(y,base+k,fmaf(normalized,dif_load(weight,channel),"
         "dif_load(bias,channel)));}}\n"
      << std::defaultfloat;
}

void emit_upsample_nearest_2d(std::ostringstream &out,
                              const ir::Program &program,
                              const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto channels = input->dims[1];
  const auto input_h = input->dims[2];
  const auto input_w = input->dims[3];
  const auto output_h = output->dims[2];
  const auto output_w = output->dims[3];
  const auto scale_h = op.u64(ir::AttrKey::ScaleH, 1U);
  const auto scale_w = op.u64(ir::AttrKey::ScaleW, 1U);
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,dif_scalar* y){unsigned long long i="
         "(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << output->element_count() << "ULL){unsigned long long ow=i%"
      << output_w << "ULL,oh=(i/" << output_w << "ULL)%" << output_h
      << "ULL,c=(i/" << output_w * output_h << "ULL)%" << channels
      << "ULL,b=i/" << channels * output_h * output_w
      << "ULL;unsigned long long source=((b*" << channels
      << "ULL+c)*" << input_h << "ULL+oh/" << scale_h << "ULL)*"
      << input_w << "ULL+ow/" << scale_w
      << "ULL;dif_store(y,i,dif_load(x,source));}}\n";
}

void emit_pad_constant(std::ostringstream &out, const ir::Program &program,
                       const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto top = op.u64(ir::AttrKey::PadTop, 0U);
  const auto west = op.u64(ir::AttrKey::PadWest, 0U);
  const auto value = static_cast<float>(op.f64(ir::AttrKey::Value, 0.0));
  out << std::scientific << std::setprecision(9)
      << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,dif_scalar* y){unsigned long long i="
         "(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << output->element_count() << "ULL){";
  if (input->dims.size() == 4U) {
    out << "unsigned long long ow=i%" << output->dims[3]
        << "ULL,oh=(i/" << output->dims[3] << "ULL)%" << output->dims[2]
        << "ULL,c=(i/" << output->dims[2] * output->dims[3] << "ULL)%"
        << output->dims[1] << "ULL,b=i/"
        << output->dims[1] * output->dims[2] * output->dims[3]
        << "ULL;if(oh<" << top << "ULL||oh>=" << top + input->dims[2]
        << "ULL||ow<" << west << "ULL||ow>=" << west + input->dims[3]
        << "ULL){dif_store(y,i," << value
        << "f);return;}unsigned long long source=((b*" << input->dims[1]
        << "ULL+c)*" << input->dims[2] << "ULL+oh-" << top << "ULL)*"
        << input->dims[3] << "ULL+ow-" << west
        << "ULL;dif_store(y,i,dif_load(x,source));}}\n";
  } else {
    const auto front = op.u64(ir::AttrKey::PadFront, 0U);
    out << "unsigned long long ow=i%" << output->dims[4]
        << "ULL,oh=(i/" << output->dims[4] << "ULL)%" << output->dims[3]
        << "ULL,ot=(i/" << output->dims[3] * output->dims[4] << "ULL)%"
        << output->dims[2] << "ULL,c=(i/"
        << output->dims[2] * output->dims[3] * output->dims[4] << "ULL)%"
        << output->dims[1] << "ULL,b=i/"
        << output->dims[1] * output->dims[2] * output->dims[3] *
               output->dims[4]
        << "ULL;if(ot<" << front << "ULL||ot>="
        << front + input->dims[2] << "ULL||oh<" << top << "ULL||oh>="
        << top + input->dims[3] << "ULL||ow<" << west << "ULL||ow>="
        << west + input->dims[4] << "ULL){dif_store(y,i," << value
        << "f);return;}unsigned long long source=(((b*" << input->dims[1]
        << "ULL+c)*" << input->dims[2] << "ULL+ot-" << front << "ULL)*"
        << input->dims[3] << "ULL+oh-" << top << "ULL)*" << input->dims[4]
        << "ULL+ow-" << west
        << "ULL;dif_store(y,i,dif_load(x,source));}}\n";
  }
  out << std::defaultfloat;
}

void emit_pad_reflect(std::ostringstream &out, const ir::Program &program,
                      const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto front = op.u64(ir::AttrKey::PadFront, 0U);
  const auto top = op.u64(ir::AttrKey::PadTop, 0U);
  const auto west = op.u64(ir::AttrKey::PadWest, 0U);
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,dif_scalar* y){unsigned long long i="
         "(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << output->element_count() << "ULL){";
  if (input->dims.size() == 4U) {
    out << "unsigned long long ow=i%" << output->dims[3]
        << "ULL,oh=(i/" << output->dims[3] << "ULL)%" << output->dims[2]
        << "ULL,c=(i/" << output->dims[2] * output->dims[3] << "ULL)%"
        << output->dims[1] << "ULL,b=i/"
        << output->dims[1] * output->dims[2] * output->dims[3]
        << "ULL;unsigned long long sy=oh<" << top << "ULL?" << top
        << "ULL-oh:(oh-" << top << "ULL<" << input->dims[2] << "ULL?oh-"
        << top << "ULL:2ULL*" << input->dims[2] << "ULL-2ULL-(oh-" << top
        << "ULL));unsigned long long sx=ow<" << west << "ULL?" << west
        << "ULL-ow:(ow-" << west << "ULL<" << input->dims[3] << "ULL?ow-"
        << west << "ULL:2ULL*" << input->dims[3] << "ULL-2ULL-(ow-" << west
        << "ULL));unsigned long long source=((b*" << input->dims[1]
        << "ULL+c)*" << input->dims[2] << "ULL+sy)*" << input->dims[3]
        << "ULL+sx;dif_store(y,i,dif_load(x,source));}}\n";
  } else {
    out << "unsigned long long ow=i%" << output->dims[4]
        << "ULL,oh=(i/" << output->dims[4] << "ULL)%" << output->dims[3]
        << "ULL,ot=(i/" << output->dims[3] * output->dims[4] << "ULL)%"
        << output->dims[2] << "ULL,c=(i/"
        << output->dims[2] * output->dims[3] * output->dims[4] << "ULL)%"
        << output->dims[1] << "ULL,b=i/"
        << output->dims[1] * output->dims[2] * output->dims[3] *
               output->dims[4]
        << "ULL;unsigned long long st=ot<" << front << "ULL?" << front
        << "ULL-ot:(ot-" << front << "ULL<" << input->dims[2] << "ULL?ot-"
        << front << "ULL:2ULL*" << input->dims[2] << "ULL-2ULL-(ot-" << front
        << "ULL));unsigned long long sy=oh<" << top << "ULL?" << top
        << "ULL-oh:(oh-" << top << "ULL<" << input->dims[3] << "ULL?oh-"
        << top << "ULL:2ULL*" << input->dims[3] << "ULL-2ULL-(oh-" << top
        << "ULL));unsigned long long sx=ow<" << west << "ULL?" << west
        << "ULL-ow:(ow-" << west << "ULL<" << input->dims[4] << "ULL?ow-"
        << west << "ULL:2ULL*" << input->dims[4] << "ULL-2ULL-(ow-" << west
        << "ULL));unsigned long long source=(((b*" << input->dims[1]
        << "ULL+c)*" << input->dims[2] << "ULL+st)*" << input->dims[3]
        << "ULL+sy)*" << input->dims[4]
        << "ULL+sx;dif_store(y,i,dif_load(x,source));}}\n";
  }
}

void emit_snake_beta(std::ostringstream &out, const ir::Program &program,
                     const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto channels = input->dims[1];
  const auto length = input->dims[2];
  const auto epsilon = static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-9));
  out << std::scientific << std::setprecision(9)
      << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,const dif_scalar* la,const dif_scalar* lb,"
         "dif_scalar* y){unsigned long long i=(unsigned long long)blockIdx.x*"
         "blockDim.x+threadIdx.x;if(i<" << count << "ULL){"
      << "unsigned long long c=(i/" << length << "ULL)%" << channels << "ULL;"
      << "float alpha=expf(dif_load(la,c));"
      << "float ib=1.0f/(expf(dif_load(lb,c))+" << epsilon << "f);"
      << "float xv=dif_load(x,i);float s=sinf(alpha*xv);"
      << "dif_store(y,i,xv+ib*s*s);}}\n" << std::defaultfloat;
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

void emit_quantize_int8_rows(std::ostringstream &out,
                             const ir::Program &program,
                             const ir::Operation &op) {
  const auto *quantized = program.tensor(op.outputs[0]);
  const auto columns = quantized->dims.back();
  const auto rows = quantized->element_count() / columns;
  const auto dynamic_clip =
      program.tensor(op.inputs.back())->dtype == ir::DType::F32;
  const auto data_input_count =
      op.inputs.size() - static_cast<std::size_t>(dynamic_clip);
  const auto input_parameters = [&] {
    std::string result;
    for (std::size_t index = 0U; index < data_input_count; ++index)
      result += "const dif_bf16* x" + std::to_string(index) + ",";
    if (dynamic_clip)
      result += "const dif_f32* clip_ratio,";
    return result;
  }();
  const auto load_input = [&] {
    std::string result;
    std::uint64_t prefix = 0U;
    for (std::size_t index = 0U; index < data_input_count; ++index) {
      const auto width = program.tensor(op.inputs[index])->dims.back();
      const auto load = "dif_load_bf16(x" + std::to_string(index) +
                        ",row*" + std::to_string(width) + "ULL+column-" +
                        std::to_string(prefix) + "ULL)";
      if (index + 1U == data_input_count) {
        result += load;
      } else {
        result += "(column<" + std::to_string(prefix + width) + "ULL?" +
                  load + ":";
      }
      prefix += width;
    }
    result.append(data_input_count - 1U, ')');
    return result;
  }();
  const auto residual2 = op.outputs.size() == 4U;
  const auto implementation = static_cast<ir::Int8RowQuantization>(
      op.u64(ir::AttrKey::Implementation,
             static_cast<std::uint64_t>(
                 ir::Int8RowQuantization::Direct)));
  const auto clip_ratio = op.f64(ir::AttrKey::Scale, 1.0);
  std::ostringstream clip_ratio_literal;
  clip_ratio_literal << std::scientific << std::setprecision(9)
                     << static_cast<float>(clip_ratio);
  const auto clip_ratio_source =
      dynamic_clip ? std::string{"dif_load_f32(clip_ratio,0ULL)"}
                   : clip_ratio_literal.str() + "f";
  if (implementation == ir::Int8RowQuantization::H256ConvRot ||
      implementation == ir::Int8RowQuantization::H256F32ConvRot ||
      implementation == ir::Int8RowQuantization::H256F32SignedConvRot ||
      implementation == ir::Int8RowQuantization::H256SignedConvRot ||
      implementation == ir::Int8RowQuantization::H4096SignedConvRot ||
      implementation == ir::Int8RowQuantization::H4096F32SignedConvRot) {
    const auto f32_convrot =
        implementation == ir::Int8RowQuantization::H256F32ConvRot ||
        implementation == ir::Int8RowQuantization::H256F32SignedConvRot ||
        implementation == ir::Int8RowQuantization::H4096F32SignedConvRot;
    const auto signed_rotation =
        implementation == ir::Int8RowQuantization::H256SignedConvRot ||
        implementation == ir::Int8RowQuantization::H256F32SignedConvRot ||
        implementation == ir::Int8RowQuantization::H4096SignedConvRot ||
        implementation == ir::Int8RowQuantization::H4096F32SignedConvRot;
    auto rotation_group =
        (implementation == ir::Int8RowQuantization::H4096SignedConvRot ||
         implementation == ir::Int8RowQuantization::H4096F32SignedConvRot)
            ? 4096U
            : 256U;
    const auto signed_rotation_source =
        signed_rotation
            ? "unsigned h=(unsigned)column+" +
                  std::to_string(0x9e3779b9U) +
                  "U;h=(h^(h>>16))*0x7feb352dU;h=(h^(h>>15))*"
                  "0x846ca68bU;h^=h>>16;value=(h&1U)?-value:value;"
            : std::string{};
    auto rotation_stages = 0U;
    for (auto width = rotation_group; width > 1U; width /= 4U)
      ++rotation_stages;
    out << "extern \"C\" __global__ void " << function_name(op)
        << "(" << input_parameters << "signed char* q,float* scales"
        << (residual2 ? ",signed char* q2,float* scales2" : "")
        << "){"
           "extern __shared__ float values[];__shared__ float maximums[256];"
           "unsigned long long row=blockIdx.x;unsigned tid=threadIdx.x;if(row>="
        << rows
        << "ULL)return;unsigned long long base=row*" << columns
        << "ULL;for(unsigned long long column=tid;column<" << columns
        << "ULL;column+=256ULL){float value=" << load_input << ";"
        << signed_rotation_source
        << "values[column]=value;}"
           "__syncthreads();for(unsigned stage=0U;stage<"
        << rotation_stages
        << "U;++stage){unsigned stride=1U<<(2U*stage);"
           "for(unsigned long long tuple=tid;tuple<"
        << columns
        << "ULL/4ULL;tuple+=256ULL){unsigned long long group=(tuple/"
        << rotation_group / 4U << "ULL)*" << rotation_group
        << "ULL;unsigned lane=(unsigned)(tuple%" << rotation_group / 4U
        << "ULL);unsigned offset=(lane%stride)+"
           "(lane/stride)*(4U*stride);unsigned long long i=group+offset;"
           "float x0=values[i],x1=values[i+stride],x2=values[i+2U*stride],"
           "x3=values[i+3U*stride];values[i]=0.5f*(x0+x1+x2-x3);"
           "values[i+stride]=0.5f*(x0+x1-x2+x3);"
           "values[i+2U*stride]=0.5f*(x0-x1+x2+x3);"
           "values[i+3U*stride]=0.5f*(-x0+x1+x2+x3);}__syncthreads();}"
           "float maximum=0.0f;for(unsigned long long column=tid;column<"
        << columns
        << "ULL;column+=256ULL)maximum=fmaxf(maximum,fabsf(values[column]));"
           "maximums[tid]=maximum;__syncthreads();for(unsigned active=128U;"
           "active>0U;active>>=1U){if(tid<active)maximums[tid]=fmaxf("
           "maximums[tid],maximums[tid+active]);__syncthreads();}float scale="
           "fmaxf(maximums[0]*"
        << clip_ratio_source
        << "/127.0f,1.0e-30f);float scale_bf16=dif_round_bf16(scale);"
           "if(tid==0U)scales[row]="
        << (signed_rotation && !f32_convrot ? "scale_bf16;" : "scale;")
        << "__syncthreads();"
           "for(unsigned long long column=tid;column<"
        << columns
        << "ULL;column+=256ULL){"
        << (f32_convrot
                ? "float value=values[column];float divided=value/scale;"
                  "int encoded=(int)nearbyintf(divided);encoded=encoded>127?"
                  "127:(encoded<-127?-127:encoded);"
                : "float value=dif_round_bf16(values[column]);float divided="
                  "dif_round_bf16(value/scale_bf16);int encoded=(int)"
                  "nearbyintf(divided);encoded=encoded>127?127:(encoded<-128?"
                  "-128:encoded);")
        << "q[base+column]=(signed char)encoded;}"
        << (residual2
                ? "__syncthreads();float stored_scale=scales[row];float "
                  "residual_maximum=0.0f;for(unsigned long long column=tid;"
                  "column<" + std::to_string(columns) +
                  "ULL;column+=256ULL){float value=" +
                  std::string(f32_convrot ? "values[column]"
                                          : "dif_round_bf16(values[column])") +
                  ";float residual=fmaf(-(float)q[base+column],"
                  "stored_scale,value);residual_maximum=fmaxf(residual_maximum,"
                  "fabsf(residual));}maximums[tid]=residual_maximum;"
                  "__syncthreads();for(unsigned active=128U;active>0U;"
                  "active>>=1U){if(tid<active)maximums[tid]=fmaxf(maximums["
                  "tid],maximums[tid+active]);__syncthreads();}float scale2="
                  "fmaxf(maximums[0]/127.0f,1.0e-30f);float scale2_bf16="
                  "dif_round_bf16(scale2);if(tid==0U)scales2[row]=" +
                  std::string(signed_rotation && !f32_convrot
                                  ? "scale2_bf16;"
                                  : "scale2;") +
                  "__syncthreads();float stored_scale2=scales2[row];for("
                  "unsigned long long column=tid;column<" +
                  std::to_string(columns) +
                  "ULL;column+=256ULL){float value=" +
                  std::string(f32_convrot ? "values[column]"
                                          : "dif_round_bf16(values[column])") +
                  ";float residual=" +
                  std::string(f32_convrot
                                  ? "fmaf(-(float)q[base+column],stored_scale,value)"
                                  : "dif_round_bf16(fmaf(-(float)q[base+column],stored_scale,value))") +
                  ";float divided=" +
                  std::string(f32_convrot
                                  ? "residual/scale2"
                                  : "dif_round_bf16(residual/scale2_bf16)") +
                  ";int encoded=(int)nearbyintf(divided);encoded=encoded>127?"
                  "127:(encoded<" +
                  std::string(f32_convrot ? "-127?-127:" : "-128?-128:") +
                  "encoded);q2[base+column]=(signed char)encoded;}"
                : "")
        << "}\n";
    return;
  }
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(" << input_parameters << "signed char* q,float* scales"
      << (residual2 ? ",signed char* q2,float* scales2" : "")
      << "){"
         "__shared__ float maximums[256];unsigned long long row=blockIdx.x;"
         "unsigned tid=threadIdx.x;if(row>="
      << rows
      << "ULL)return;unsigned long long base=row*" << columns
      << "ULL;float maximum=0.0f;for(unsigned long long column=tid;column<"
      << columns
      << "ULL;column+=256ULL){float value=fabsf(" << load_input << ");"
         "maximum=fmaxf(maximum,value);}maximums[tid]=maximum;__syncthreads();"
         "for(unsigned active=128U;active>0U;active>>=1U){if(tid<active)"
         "maximums[tid]=fmaxf(maximums[tid],maximums[tid+active]);"
         "__syncthreads();}float scale=fmaxf(maximums[0]*"
      << clip_ratio_source
      << "/127.0f,1.0e-30f);"
         "if(tid==0U)scales[row]=scale;for(unsigned long long column=tid;column<"
      << columns
      << "ULL;column+=256ULL){int value=(int)rintf(" << load_input
      << "/scale);value=value>127?127:(value<-127?-127:value);"
         "q[base+column]=(signed char)value;}"
      << (residual2
              ? "__syncthreads();float residual_maximum=0.0f;for(unsigned "
                "long long column=tid;column<" + std::to_string(columns) +
                "ULL;column+=256ULL){float residual=fmaf(-(float)q[base+"
                "column],scale," + load_input + ");residual_maximum=fmaxf("
                "residual_maximum,fabsf(residual));}maximums[tid]="
                "residual_maximum;__syncthreads();for(unsigned active=128U;"
                "active>0U;active>>=1U){if(tid<active)maximums[tid]=fmaxf("
                "maximums[tid],maximums[tid+active]);__syncthreads();}float "
                "scale2=fmaxf(maximums[0]/127.0f,1.0e-30f);if(tid==0U)"
                "scales2[row]=scale2;__syncthreads();for(unsigned long long "
                "column=tid;column<" + std::to_string(columns) +
                "ULL;column+=256ULL){float residual=fmaf(-(float)q[base+"
                "column],scale," + load_input + ");int value=(int)rintf("
                "residual/scale2);value=value>127?127:(value<-127?-127:"
                "value);q2[base+column]=(signed char)value;}"
              : "")
      << "}\n";
}

void emit_dequantize_int8_blocks(std::ostringstream &out,
                                 const ir::Program &program,
                                 const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *scales = program.tensor(op.inputs[1]);
  const auto count = input->element_count();
  const auto columns = input->dims[1];
  const auto scale_columns = scales->dims[1];
  const auto block = op.u64(ir::AttrKey::BlockSize, 0U);
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const signed char* x,const float* scales,dif_bf16* y){"
         "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+"
         "threadIdx.x;if(i<"
      << count << "ULL){unsigned long long row=i/" << columns
      << "ULL;unsigned long long column=i%" << columns
      << "ULL;dif_store_bf16(y,i,(float)x[i]*scales[row*"
      << scale_columns << "ULL+column/" << block << "ULL]);}}\n";
}

void emit_quantize_fp8_rows(std::ostringstream &out,
                            const ir::Program &program,
                            const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto columns = input->dims.back();
  const auto rows = input->element_count() / columns;
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_bf16* x,unsigned char* q,float* scales){"
         "__shared__ float maximums[256];unsigned long long row=blockIdx.x;"
         "unsigned tid=threadIdx.x;if(row>="
      << rows
      << "ULL)return;unsigned long long base=row*" << columns
      << "ULL;float maximum=0.0f;for(unsigned long long column=tid;column<"
      << columns
      << "ULL;column+=256ULL){maximum=fmaxf(maximum,fabsf(dif_load_bf16(x,"
         "base+column)));}maximums[tid]=maximum;__syncthreads();"
         "for(unsigned active=128U;active>0U;active>>=1U){if(tid<active)"
         "maximums[tid]=fmaxf(maximums[tid],maximums[tid+active]);"
         "__syncthreads();}float scale=fmaxf(maximums[0]/448.0f,1.0e-30f);"
         "if(tid==0U)scales[row]=scale;for(unsigned long long column=tid;"
         "column<"
      << columns
      << "ULL;column+=256ULL){float value=dif_load_bf16(x,base+column)/scale;"
         "unsigned short pair;asm(\"{cvt.rn.satfinite.e4m3x2.f32 %0, %2, "
         "%1;}\\n\":\"=h\"(pair):\"f\"(value),\"f\"(0.0f));"
         "q[base+column]=(unsigned char)pair;}}\n";
}

void emit_linear_fp8_output_scale(std::ostringstream &out,
                                  const ir::Program &program,
                                  const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *weight = program.tensor(op.inputs[1]);
  const auto rows = input->element_count() / input->dims.back();
  const auto columns = weight->dims.front();
  const auto count = rows * columns;
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const unsigned char* x,const unsigned char* w,const float* rs,"
         "const float* cs,dif_bf16* y){unsigned long long i="
         "(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){unsigned long long row=i/" << columns
      << "ULL;unsigned long long column=i%" << columns
      << "ULL;dif_store_bf16(y,i,dif_load_bf16(y,i)*rs[row]*cs[column]);}}\n";
}

void emit_quantize_fp8_blocks32(std::ostringstream &out,
                                const ir::Program &program,
                                const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *scales = program.tensor(op.outputs[1]);
  const auto columns = input->dims.back();
  const auto rows = input->element_count() / columns;
  const auto blocks = (columns + 31U) / 32U;
  const auto scale_inner_dimension = scales->dims.at(1);
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_bf16* x,unsigned char* q,unsigned char* scales){"
         "unsigned long long row=blockIdx.x;unsigned tid=threadIdx.x;"
         "if(row>="
      << rows
      << "ULL)return;unsigned lane=tid&31U;unsigned warp=tid>>5U;"
         "for(unsigned long long block=warp;block<"
      << blocks
      << "ULL;block+=8ULL){unsigned long long column=block*32ULL+lane;"
         "float value=column<"
      << columns
      << "ULL?dif_load_bf16(x,row*" << columns
      << "ULL+column):0.0f;float maximum=fabsf(value);"
         "for(unsigned offset=16U;offset>0U;offset>>=1U)"
         "maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffU,maximum,offset));"
         "maximum=__shfl_sync(0xffffffffU,maximum,0U);float target=maximum/448.0f;"
         "unsigned bits=__float_as_uint(target)&0x7fffffffU;"
         "unsigned exponent=bits>>23U;unsigned mantissa=bits&0x7fffffU;"
         "unsigned encoded=(bits==0U)?0U:((exponent==0U)?"
         "(mantissa>0x400000U?1U:0U):exponent+(mantissa!=0U));"
         "unsigned char encoded_scale=(unsigned char)(encoded>254U?254U:encoded);"
         "float scale=ldexpf(1.0f,(int)encoded_scale-127);"
         "if(lane==0U){unsigned long long tile_outer=row/128ULL;"
         "unsigned long long tile_inner=(block/4ULL)*4ULL;"
         "unsigned long long within=(row%32ULL)*16ULL+"
         "((row%128ULL)/32ULL)*4ULL+block%4ULL;"
         "unsigned long long scale_offset=(tile_inner+tile_outer*"
      << scale_inner_dimension
      << "ULL)*128ULL+within;scales[scale_offset]=encoded_scale;}"
         "if(column<"
      << columns
      << "ULL){float divided=value/scale;unsigned short pair;"
         "asm(\"{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}\\n\":"
         "\"=h\"(pair):\"f\"(divided),\"f\"(0.0f));"
         "q[row*"
      << columns << "ULL+column]=(unsigned char)pair;}}}\n";
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

void emit_gelu(std::ostringstream &out, const ir::Program &program,
               const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto approximation = static_cast<ir::GeluApproximation>(
      op.u64(ir::AttrKey::Approximation, 0U));
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,dif_scalar* y){unsigned long long i="
         "(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){float v=dif_load(x,i);";
  if (approximation == ir::GeluApproximation::ExactErf)
    out << "dif_store(y,i,5.0e-1f*v*(1.0f+erff(v*7.071067812e-1f)));";
  else
    out << "float c=v*v*v;float z="
           "7.978845608e-1f*(v+4.471500218e-2f*c);"
           "dif_store(y,i,5.0e-1f*v*(1.0f+tanhf(z)));";
  out << "}}\n";
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

void emit_sigmoid(std::ostringstream &out, const ir::Program &program,
                  const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,dif_scalar* y){unsigned long long i="
         "(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count
      << "ULL){float v=dif_load(x,i);dif_store(y,i,1.0f/(1.0f+expf(-v)));}}\n";
}

void emit_reshape(std::ostringstream &out, const ir::Program &program,
                  const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,dif_scalar* y){unsigned long long i="
         "(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL)dif_store(y,i,dif_load(x,i));}\n";
}

void emit_broadcast_to(std::ostringstream &out, const ir::Program &program,
                       const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto rank_pad = output->dims.size() - input->dims.size();
  std::vector<std::uint64_t> strides(input->dims.size(), 1U);
  for (std::size_t axis = input->dims.size(); axis-- > 1U;)
    strides[axis - 1U] = strides[axis] * input->dims[axis];
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,dif_scalar* y){unsigned long long i="
         "(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << output->element_count()
      << "ULL){unsigned long long coordinate=i,source=0ULL,at=0ULL;";
  for (std::size_t axis = output->dims.size(); axis-- > 0U;) {
    out << "at=coordinate%" << output->dims[axis]
        << "ULL;coordinate/=" << output->dims[axis] << "ULL;";
    if (axis >= rank_pad && input->dims[axis - rank_pad] != 1U)
      out << "source+=at*" << strides[axis - rank_pad] << "ULL;";
  }
  out << "dif_store(y,i,dif_load(x,source));}}\n";
}

void emit_slice(std::ostringstream &out, const ir::Program &program,
                const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto selected =
      static_cast<std::size_t>(op.u64(ir::AttrKey::Axis, 0U));
  const auto start = op.u64(ir::AttrKey::Start, 0U);
  std::vector<std::uint64_t> strides(input->dims.size(), 1U);
  for (std::size_t axis = input->dims.size(); axis-- > 1U;)
    strides[axis - 1U] = strides[axis] * input->dims[axis];
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,dif_scalar* y){unsigned long long i="
         "(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << output->element_count()
      << "ULL){unsigned long long coordinate=i,source=0ULL,at=0ULL;";
  for (std::size_t axis = output->dims.size(); axis-- > 0U;) {
    out << "at=coordinate%" << output->dims[axis]
        << "ULL;coordinate/=" << output->dims[axis] << "ULL;";
    if (axis == selected)
      out << "at+=" << start << "ULL;";
    out << "source+=at*" << strides[axis] << "ULL;";
  }
  out << "dif_store(y,i,dif_load(x,source));}}\n";
}

void emit_rotary_frequency(std::ostringstream &out,
                           const ir::Program &program,
                           const ir::Operation &op) {
  const auto *positions = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto sequence = positions->dims[1];
  const auto axes = positions->dims[2];
  const auto pairs = output->dims[2];
  out << std::setprecision(17)
      << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_f32* positions,const int* pair_axes,const int* "
         "pair_indices,const int* axis_dims,dif_f32* cosine,dif_f32* sine){"
         "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+"
         "threadIdx.x;if(i<"
      << output->element_count() << "ULL){unsigned long long pair=i%" << pairs
      << "ULL,token=(i/" << pairs << "ULL)%" << sequence
      << "ULL,batch=i/(" << pairs << "ULL*" << sequence
      << "ULL);int axis=pair_axes[pair],component=pair_indices[pair],"
         "axis_dim=axis_dims[axis];float scale=(2.0f*(float)component)/"
         "(float)axis_dim;float omega=1.0f/powf((float)("
      << op.f64(ir::AttrKey::Theta, 10000.0) << "*"
      << op.f64(ir::AttrKey::Ntk, 1.0)
      << "),scale);float angle=dif_load_f32(positions,(batch*"
      << sequence << "ULL+token)*" << axes
      << "ULL+(unsigned long long)axis)*omega;dif_store_f32(cosine,i,"
         "cosf(angle));dif_store_f32(sine,i,sinf(angle));}}\n"
      << std::defaultfloat;
}

void emit_rotary_apply(std::ostringstream &out, const ir::Program &program,
                       const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *cosine = program.tensor(op.inputs[1]);
  const auto sequence = input->dims[1];
  const auto heads = input->dims[2];
  const auto dim = input->dims[3];
  const auto pairs = cosine->dims[2];
  out << "extern \"C\" __global__ void " << function_name(op) << "(const "
      << typed_scalar(input->dtype)
      << "* x,const dif_f32* cosine,const dif_f32* sine,"
      << typed_scalar(input->dtype)
      << "* y){unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+"
         "threadIdx.x;if(i<"
      << input->element_count() << "ULL){unsigned long long d=i%" << dim
      << "ULL,outer=i/" << dim << "ULL,token=(outer/" << heads << "ULL)%"
      << sequence << "ULL,batch=outer/(" << heads << "ULL*" << sequence
      << "ULL);if(d<" << 2U * pairs
      << "ULL){unsigned long long pair=d/2ULL,base=i-d,table=(batch*"
      << sequence << "ULL+token)*" << pairs << "ULL+pair;float even="
      << typed_load(input->dtype) << "(x,base+2ULL*pair),odd="
      << typed_load(input->dtype)
      << "(x,base+2ULL*pair+1ULL),c=dif_load_f32(cosine,table),"
         "s=dif_load_f32(sine,table),first,second,result;"
         "if(d&1ULL){asm volatile(\"mul.rn.f32 %0,%1,%2;\":\"=f\"(first):"
         "\"f\"(even),\"f\"(s));"
         "asm volatile(\"mul.rn.f32 %0,%1,%2;\":\"=f\"(second):"
         "\"f\"(odd),\"f\"(c));"
         "asm volatile(\"add.rn.f32 %0,%1,%2;\":\"=f\"(result):"
         "\"f\"(first),\"f\"(second));}else{"
         "asm volatile(\"mul.rn.f32 %0,%1,%2;\":\"=f\"(first):"
         "\"f\"(even),\"f\"(c));"
         "asm volatile(\"mul.rn.f32 %0,%1,%2;\":\"=f\"(second):"
         "\"f\"(odd),\"f\"(s));"
         "asm volatile(\"sub.rn.f32 %0,%1,%2;\":\"=f\"(result):"
         "\"f\"(first),\"f\"(second));}"
      << typed_store(input->dtype)
      << "(y,i,result);}else "
      << typed_store(input->dtype) << "(y,i," << typed_load(input->dtype)
      << "(x,i));}}\n";
}

void emit_boolean_mask_to_bias(std::ostringstream &out,
                               const ir::Program &program,
                               const ir::Operation &op) {
  const auto *mask = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto sequence = output->dims[2];
  const bool vector_mask = mask->dims.size() == 2U;
  const bool mask_queries = op.boolean(ir::AttrKey::MaskQueries, true);
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const unsigned char* mask," << typed_scalar(output->dtype)
      << "* y){unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+"
         "threadIdx.x;if(i<"
      << output->element_count() << "ULL){unsigned long long key=i%"
      << sequence << "ULL,query=(i/" << sequence << "ULL)%" << sequence
      << "ULL,batch=i/(" << sequence << "ULL*" << sequence
      << "ULL);bool valid=";
  if (vector_mask) {
    if (mask_queries)
      out << "mask[batch*" << sequence << "ULL+query]&&";
    out << "mask[batch*" << sequence << "ULL+key]";
  }
  else
    out << "mask[(batch*" << sequence << "ULL+query)*" << sequence
        << "ULL+key]";
  out << ";" << typed_store(output->dtype)
      << "(y,i,valid?0.0f:-__int_as_float(0x7f800000));}}\n";
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
  const auto clip_scale =
      static_cast<float>(op.f64(ir::AttrKey::ClipScale, 1.0));
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
      << "(gradient,i)*" << clip_scale
      << "f;float m=beta1*dif_load_f32(first,i)+(1.0f-beta1)*grad;"
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
      << "f;float frequency=expf(exponent);float scaled_timestep="
      << "dif_load_f32(timesteps,row)*" << scale
      << "f;float angle=scaled_timestep*frequency;float s=sinf(angle),"
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

void emit_euler_velocity_step(std::ostringstream &out,
                              const ir::Program &program,
                              const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* sample,const dif_scalar* velocity,const dif_f32* "
         "current,const dif_f32* next,dif_scalar* output){unsigned long long "
         "i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count
      << "ULL){float dt=__fsub_rn(dif_load_f32(next,0ULL),"
         "dif_load_f32(current,0ULL));float scaled=dif_round(__fmul_rn(dt,"
         "dif_load(velocity,i)));dif_store(output,i,__fadd_rn("
         "dif_load(sample,i),scaled));}}\n";
}

void emit_permute(std::ostringstream &out, const ir::Program &program,
                  const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto rank = input->dims.size();
  std::vector<std::uint64_t> input_strides(rank, 1U);
  for (std::size_t axis = rank - 1U; axis > 0U; --axis)
    input_strides[axis - 1U] = input_strides[axis] * input->dims[axis];
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* input,dif_scalar* output){unsigned long long i="
         "(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << output->element_count()
      << "ULL){unsigned long long remaining=i,source=0ULL,coordinate;";
  for (std::size_t reverse = rank; reverse-- > 0U;) {
    const auto key = static_cast<ir::AttrKey>(
        static_cast<std::uint32_t>(ir::AttrKey::Permutation0) + reverse);
    const auto input_axis = op.u64(key, 0U);
    out << "coordinate=remaining%" << output->dims[reverse]
        << "ULL;remaining/=" << output->dims[reverse]
        << "ULL;source+=coordinate*" << input_strides[input_axis] << "ULL;";
  }
  out << "dif_store(output,i,dif_load(input,source));}}\n";
}

void emit_concat(std::ostringstream &out, const ir::Program &program,
                 const ir::Operation &op) {
  const auto *output = program.tensor(op.outputs[0]);
  const auto axis = static_cast<std::size_t>(op.u64(ir::AttrKey::Axis, 0U));
  std::uint64_t inner = 1U;
  for (std::size_t dimension = axis + 1U; dimension < output->dims.size();
       ++dimension)
    inner *= output->dims[dimension];
  out << "extern \"C\" __global__ void " << function_name(op) << "(";
  for (std::size_t input = 0U; input < op.inputs.size(); ++input) {
    if (input != 0U)
      out << ',';
    out << "const dif_scalar* input" << input;
  }
  out << ",dif_scalar* output){unsigned long long i=(unsigned long long)"
         "blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << output->element_count()
      << "ULL){unsigned long long inner_index=i%" << inner
      << "ULL,axis_coordinate=(i/" << inner << "ULL)%"
      << output->dims[axis] << "ULL,outer=i/(" << output->dims[axis]
      << "ULL*" << inner << "ULL),source=0ULL;";
  std::uint64_t offset = 0U;
  for (std::size_t input = 0U; input < op.inputs.size(); ++input) {
    const auto input_axis = program.tensor(op.inputs[input])->dims[axis];
    out << (input == 0U ? "if" : "else if") << "(axis_coordinate<"
        << offset + input_axis << "ULL){source=(outer*" << input_axis
        << "ULL+(axis_coordinate-" << offset << "ULL))*" << inner
        << "ULL+inner_index;dif_store(output,i,dif_load(input" << input
        << ",source));}";
    offset += input_axis;
  }
  out << "}}\n";
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
  const auto weight_offset = op.f64(ir::AttrKey::WeightOffset, 0.0);
  std::ostringstream weight_offset_literal;
  weight_offset_literal << std::scientific << std::setprecision(9)
                        << static_cast<float>(weight_offset);
  const auto block = op.u64(ir::AttrKey::BlockSize, 256U);
  const auto reduction_tile = op.u64(ir::AttrKey::ReductionTileSize, 0U);
  if (op.u64(ir::AttrKey::Implementation, 1U) == 2U) {
    out << std::setprecision(17) << "extern \"C\" __global__ void "
        << function_name(op)
        << "(const dif_scalar* x,const dif_scalar* weight,dif_scalar* y){"
           "extern __shared__ float reduction[];unsigned long long row="
           "blockIdx.x;if(row>="
        << rows
        << "ULL)return;unsigned tid=threadIdx.x;float sigma2=0.0f;if(tid<32U){"
           "unsigned long long base=row*128ULL+(unsigned long long)tid;float "
           "v0=dif_load(x,base),v1=dif_load(x,base+32ULL),v2=dif_load(x,base+"
           "64ULL),v3=dif_load(x,base+96ULL),s0,s1,s2,s3;asm volatile("
           "\"mul.rn.f32 %0,%1,%1;\":\"=f\"(s0):\"f\"(v0));asm volatile("
           "\"mul.rn.f32 %0,%1,%1;\":\"=f\"(s1):\"f\"(v1));asm volatile("
           "\"mul.rn.f32 %0,%1,%1;\":\"=f\"(s2):\"f\"(v2));asm volatile("
           "\"mul.rn.f32 %0,%1,%1;\":\"=f\"(s3):\"f\"(v3));sigma2=((s0+"
           "s1)+s2)+s3;for(unsigned offset=1U;offset<32U;offset<<=1U)sigma2="
           "sigma2+__shfl_down_sync(0xffffffffU,sigma2,offset);}if(tid==0U)"
           "reduction[0]=sigma2*0.0078125f;__syncthreads();float inverse=rsqrtf("
           "reduction[0]+"
        << static_cast<float>(epsilon)
        << "f);if(tid<128U){unsigned long long index=row*128ULL+tid;dif_store("
           "y,index,dif_load(weight,tid)*(inverse*dif_load(x,index)));}}\n";
    return;
  }
  const auto triton_blocked_reduction =
      block == 512U && columns == block * 12U && reduction_tile == 8192U;
  const auto triton_chunked_reduction =
      block == 512U && columns == block * 12U && reduction_tile == 2048U;
  const auto triton_per_row_reduction = block == 128U && columns == 128U;
  out << std::setprecision(17) << "extern \"C\" __global__ void "
      << function_name(op)
      << "(const dif_scalar* x,const dif_scalar* weight,dif_scalar* y){\n"
         "  extern __shared__ float reduction[];unsigned long long row=blockIdx.x;"
         "float local=0.0f;if(row>="
      << rows << "ULL)return;\n";
  if (triton_blocked_reduction) {
    out << "  unsigned long long base=row*" << columns
        << "ULL+(unsigned long long)threadIdx.x*8ULL;"
           "float v0=dif_load(x,base),v1=dif_load(x,base+1ULL),"
           "v2=dif_load(x,base+2ULL),v3=dif_load(x,base+3ULL),"
           "v4=dif_load(x,base+4ULL),v5=dif_load(x,base+5ULL),"
           "v6=dif_load(x,base+6ULL),v7=dif_load(x,base+7ULL);"
           "local=v1*v1;local=fmaf(v0,v0,local);"
           "local=fmaf(v2,v2,local);local=fmaf(v3,v3,local);"
           "local=fmaf(v4,v4,local);local=fmaf(v5,v5,local);"
           "local=fmaf(v6,v6,local);local=fmaf(v7,v7,local);"
           "if(threadIdx.x<256U){float square,value;"
           "value=dif_load(x,base+4096ULL);"
           "asm volatile(\"mul.rn.f32 %0,%1,%1;\":\"=f\"(square):\"f\"(value));"
           "local=square+local;value=dif_load(x,base+4097ULL);"
           "asm volatile(\"mul.rn.f32 %0,%1,%1;\":\"=f\"(square):\"f\"(value));"
           "local=square+local;value=dif_load(x,base+4098ULL);"
           "asm volatile(\"mul.rn.f32 %0,%1,%1;\":\"=f\"(square):\"f\"(value));"
           "local=square+local;value=dif_load(x,base+4099ULL);"
           "asm volatile(\"mul.rn.f32 %0,%1,%1;\":\"=f\"(square):\"f\"(value));"
           "local=square+local;value=dif_load(x,base+4100ULL);"
           "asm volatile(\"mul.rn.f32 %0,%1,%1;\":\"=f\"(square):\"f\"(value));"
           "local=square+local;value=dif_load(x,base+4101ULL);"
           "asm volatile(\"mul.rn.f32 %0,%1,%1;\":\"=f\"(square):\"f\"(value));"
           "local=square+local;value=dif_load(x,base+4102ULL);"
           "asm volatile(\"mul.rn.f32 %0,%1,%1;\":\"=f\"(square):\"f\"(value));"
           "local=square+local;value=dif_load(x,base+4103ULL);"
           "asm volatile(\"mul.rn.f32 %0,%1,%1;\":\"=f\"(square):\"f\"(value));"
           "local=square+local;}"
           "for(unsigned delta=16U;delta>0U;delta>>=1U)"
           "local+=__shfl_xor_sync(0xffffffffU,local,delta);"
           "unsigned lane=threadIdx.x&31U,warp=threadIdx.x>>5U;"
           "if(lane==0U)reduction[warp]=local;__syncthreads();"
           "if(warp==0U){local=lane<16U?reduction[lane]:0.0f;"
           "for(unsigned delta=8U;delta>0U;delta>>=1U)"
           "local+=__shfl_xor_sync(0xffffffffU,local,delta);"
           "if(lane==0U)reduction[0]=local;}__syncthreads();\n";
  } else if (triton_chunked_reduction) {
    out << "  unsigned long long base=row*" << columns
        << "ULL+(unsigned long long)threadIdx.x*4ULL;float a0,a1,a2,a3;"
           "float m0=dif_load(x,base+2048ULL),m1=dif_load(x,base+2049ULL),"
           "m2=dif_load(x,base+2050ULL),m3=dif_load(x,base+2051ULL);"
           "float l0=dif_load(x,base),l1=dif_load(x,base+1ULL),"
           "l2=dif_load(x,base+2ULL),l3=dif_load(x,base+3ULL);"
           "a0=fmaf(l0,l0,m0*m0);a1=fmaf(l1,l1,m1*m1);"
           "a2=fmaf(l2,l2,m2*m2);a3=fmaf(l3,l3,m3*m3);"
           "float h0=dif_load(x,base+4096ULL),h1=dif_load(x,base+4097ULL),"
           "h2=dif_load(x,base+4098ULL),h3=dif_load(x,base+4099ULL);"
           "a0=fmaf(h0,h0,a0);a1=fmaf(h1,h1,a1);"
           "a2=fmaf(h2,h2,a2);a3=fmaf(h3,h3,a3);"
           "local=((a0+a1)+a2)+a3;"
           "for(unsigned delta=16U;delta>0U;delta>>=1U)"
           "local+=__shfl_xor_sync(0xffffffffU,local,delta);"
           "unsigned lane=threadIdx.x&31U,warp=threadIdx.x>>5U;"
           "if(lane==0U)reduction[warp]=local;__syncthreads();"
           "if(warp==0U){local=lane<16U?reduction[lane]:0.0f;"
           "for(unsigned delta=8U;delta>0U;delta>>=1U)"
           "local+=__shfl_xor_sync(0xffffffffU,local,delta);"
           "if(lane==0U)reduction[0]=local;}__syncthreads();\n";
  } else if (triton_per_row_reduction) {
    out << "  if(threadIdx.x<16U){unsigned long long base=row*128ULL+"
           "(unsigned long long)threadIdx.x*8ULL;"
           "float v0=dif_load(x,base),v1=dif_load(x,base+1ULL),"
           "v2=dif_load(x,base+2ULL),v3=dif_load(x,base+3ULL),"
           "v4=dif_load(x,base+4ULL),v5=dif_load(x,base+5ULL),"
           "v6=dif_load(x,base+6ULL),v7=dif_load(x,base+7ULL);"
           "local=v1*v1;local=fmaf(v0,v0,local);"
           "local=fmaf(v2,v2,local);local=fmaf(v3,v3,local);"
           "local=fmaf(v4,v4,local);local=fmaf(v5,v5,local);"
           "local=fmaf(v6,v6,local);local=fmaf(v7,v7,local);}"
           "else local=0.0f;"
           "if(threadIdx.x<32U){for(unsigned delta=8U;delta>0U;delta>>=1U)"
           "local+=__shfl_xor_sync(0xffffffffU,local,delta);"
           "if(threadIdx.x==0U)reduction[0]=local;}__syncthreads();\n";
  } else if (columns % 4U == 0U && block >= 128U && block < 512U) {
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
  if (triton_blocked_reduction || triton_chunked_reduction ||
      triton_per_row_reduction) {
    out << "  float mean,mean_eps,inv;asm volatile(\"div.full.f32 %0,%1,%2;\""
           ":\"=f\"(mean):\"f\"(reduction[0]),\"f\"("
        << columns
        << ".0f));mean_eps=mean+" << static_cast<float>(epsilon)
        << "f;asm volatile(\"rsqrt.approx.ftz.f32 %0,%1;\""
           ":\"=f\"(inv):\"f\"(mean_eps));";
  } else {
    out << "  float inv=rsqrtf(reduction[0]/" << columns << ".0f+"
        << static_cast<float>(epsilon) << "f);";
  }
  out << "for(unsigned long long col=threadIdx.x;col<" << columns
      << "ULL;col+=blockDim.x){unsigned long long i=row*" << columns
      << "ULL+col;dif_store(y,i,dif_load(x,i)*inv*(dif_load(weight,col)+"
      << weight_offset_literal.str() << "f));}}\n";
}

void emit_layer_norm(std::ostringstream &out, const ir::Program &program,
                     const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto columns = input->dims.back();
  const auto rows = input->element_count() / columns;
  const auto epsilon = op.f64(ir::AttrKey::Epsilon, 1.0e-5);
  const auto block = op.u64(ir::AttrKey::BlockSize, 256U);
  if (input->dtype == ir::DType::BF16 && columns % 4U == 0U &&
      block == 128U) {
    const auto suffix = std::to_string(op.id);
    out << std::setprecision(17)
        << "struct dif_welford_" << suffix
        << "{float mean;float sigma2;float count;};\n"
        << "extern \"C\" __device__ __forceinline__ dif_welford_" << suffix
        << " dif_welford_online_" << suffix
        << "(float value,dif_welford_" << suffix
        << " current){float delta=value-current.mean;float count=current.count+"
           "1.0f;float mean=current.mean+delta*(1.0f/count);return {mean,"
           "current.sigma2+delta*(value-mean),count};}\n"
        << "extern \"C\" __device__ __forceinline__ dif_welford_" << suffix
        << " dif_welford_combine_" << suffix << "(dif_welford_" << suffix
        << " data_b,dif_welford_" << suffix
        << " data_a){float delta=data_b.mean-data_a.mean;float count="
           "data_a.count+data_b.count;if(count>0.0f){float coefficient="
           "1.0f/count;float n_a=data_a.count*coefficient;float n_b="
           "data_b.count*coefficient;return {n_a*data_a.mean+n_b*data_b.mean,"
           "data_a.sigma2+data_b.sigma2+delta*delta*data_a.count*n_b,count};}"
           "return {0.0f,0.0f,0.0f};}\n"
        << "extern \"C\" __global__ void " << function_name(op)
        << "(const dif_scalar* x,const dif_scalar* weight,const dif_scalar* "
           "bias,dif_scalar* y){extern __shared__ float reduction[];unsigned "
           "long long row=blockIdx.x;if(row>="
        << rows
        << "ULL)return;unsigned tid=threadIdx.x,lane=tid&31U,warp=tid>>5U;"
           "dif_welford_"
        << suffix
        << " state={0.0f,0.0f,0.0f};for(unsigned long long pack=tid;pack<"
        << columns / 4U
        << "ULL;pack+=128ULL){unsigned long long base=row*" << columns
        << "ULL+pack*4ULL;state=dif_welford_online_" << suffix
        << "(dif_load(x,base),state);state=dif_welford_online_" << suffix
        << "(dif_load(x,base+1ULL),state);state=dif_welford_online_" << suffix
        << "(dif_load(x,base+2ULL),state);state=dif_welford_online_" << suffix
        << "(dif_load(x,base+3ULL),state);}for(int offset=16;offset>0;offset>>=1)"
           "{dif_welford_"
        << suffix
        << " other={__shfl_down_sync(0xffffffffU,state.mean,offset),"
           "__shfl_down_sync(0xffffffffU,state.sigma2,offset),"
           "__shfl_down_sync(0xffffffffU,state.count,offset)};state="
           "dif_welford_combine_"
        << suffix
        << "(state,other);}float* meansigma=reduction;float* counts=reduction+"
           "4U;for(unsigned offset=2U;offset>0U;offset>>=1U){if(lane==0U&&"
           "warp>=offset&&warp<2U*offset){unsigned target=warp-offset;"
           "meansigma[2U*target]=state.mean;meansigma[2U*target+1U]="
           "state.sigma2;counts[target]=state.count;}__syncthreads();if(lane=="
           "0U&&warp<offset){dif_welford_"
        << suffix
        << " other={meansigma[2U*warp],meansigma[2U*warp+1U],counts[warp]};"
           "state=dif_welford_combine_"
        << suffix
        << "(state,other);}__syncthreads();}if(tid==0U){meansigma[0]=state.mean;"
           "meansigma[1]=state.sigma2/"
        << columns
        << ".0f;}__syncthreads();float mean=meansigma[0];float inverse=rsqrtf("
           "meansigma[1]+"
        << static_cast<float>(epsilon)
        << "f);for(unsigned long long pack=tid;pack<" << columns / 4U
        << "ULL;pack+=128ULL){unsigned long long base=row*" << columns
        << "ULL+pack*4ULL;for(unsigned inner=0U;inner<4U;++inner){"
           "unsigned long long index=base+inner;float normalized=inverse*("
           "dif_load(x,index)-mean);dif_store(y,index,dif_load(weight,pack*"
           "4ULL+inner)*normalized+dif_load(bias,pack*4ULL+inner));}}}\n";
    return;
  }
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
      << ".0f;__syncthreads();local=0.0f;for(unsigned long long col=threadIdx.x;col<"
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

void emit_layer_norm_modulate(std::ostringstream &out,
                              const ir::Program &program,
                              const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *scale = program.tensor(op.inputs[3]);
  const auto columns = input->dims.back();
  const auto rows = input->element_count() / columns;
  const auto modulation_rows = scale->element_count() / columns;
  const auto rows_per_modulation = rows / modulation_rows;
  const auto epsilon = op.f64(ir::AttrKey::Epsilon, 1.0e-5);
  const auto block = op.u64(ir::AttrKey::BlockSize, 256U);
  if (input->dtype == ir::DType::BF16 && columns % 4U == 0U &&
      block == 128U) {
    const auto suffix = std::to_string(op.id);
    out << std::setprecision(17)
        << "struct dif_welford_mod_" << suffix
        << "{float mean;float sigma2;float count;};\n"
        << "extern \"C\" __device__ __forceinline__ dif_welford_mod_"
        << suffix << " dif_welford_mod_online_" << suffix
        << "(float value,dif_welford_mod_" << suffix
        << " current){float delta=value-current.mean;float count=current.count+"
           "1.0f;float mean=current.mean+delta*(1.0f/count);return {mean,"
           "current.sigma2+delta*(value-mean),count};}\n"
        << "extern \"C\" __device__ __forceinline__ dif_welford_mod_"
        << suffix << " dif_welford_mod_combine_" << suffix
        << "(dif_welford_mod_" << suffix << " data_b,dif_welford_mod_"
        << suffix
        << " data_a){float delta=data_b.mean-data_a.mean;float count="
           "data_a.count+data_b.count;if(count>0.0f){float coefficient="
           "1.0f/count;float n_a=data_a.count*coefficient;float n_b="
           "data_b.count*coefficient;return {n_a*data_a.mean+n_b*data_b.mean,"
           "data_a.sigma2+data_b.sigma2+delta*delta*data_a.count*n_b,count};}"
           "return {0.0f,0.0f,0.0f};}\n"
        << "extern \"C\" __global__ void " << function_name(op)
        << "(const dif_scalar* x,const dif_scalar* weight,const dif_scalar* "
           "bias,const dif_scalar* scale,const dif_scalar* shift,dif_scalar* "
           "y){extern __shared__ float reduction[];unsigned long long row="
           "blockIdx.x;if(row>="
        << rows
        << "ULL)return;unsigned tid=threadIdx.x,lane=tid&31U,warp=tid>>5U;"
           "dif_welford_mod_"
        << suffix
        << " state={0.0f,0.0f,0.0f};for(unsigned long long pack=tid;pack<"
        << columns / 4U
        << "ULL;pack+=128ULL){unsigned long long base=row*" << columns
        << "ULL+pack*4ULL;state=dif_welford_mod_online_" << suffix
        << "(dif_load(x,base),state);state=dif_welford_mod_online_" << suffix
        << "(dif_load(x,base+1ULL),state);state=dif_welford_mod_online_"
        << suffix
        << "(dif_load(x,base+2ULL),state);state=dif_welford_mod_online_"
        << suffix
        << "(dif_load(x,base+3ULL),state);}for(int offset=16;offset>0;"
           "offset>>=1){dif_welford_mod_"
        << suffix
        << " other={__shfl_down_sync(0xffffffffU,state.mean,offset),"
           "__shfl_down_sync(0xffffffffU,state.sigma2,offset),"
           "__shfl_down_sync(0xffffffffU,state.count,offset)};state="
           "dif_welford_mod_combine_"
        << suffix
        << "(state,other);}float* meansigma=reduction;float* counts=reduction+"
           "4U;for(unsigned offset=2U;offset>0U;offset>>=1U){if(lane==0U&&"
           "warp>=offset&&warp<2U*offset){unsigned target=warp-offset;"
           "meansigma[2U*target]=state.mean;meansigma[2U*target+1U]="
           "state.sigma2;counts[target]=state.count;}__syncthreads();if(lane=="
           "0U&&warp<offset){dif_welford_mod_"
        << suffix
        << " other={meansigma[2U*warp],meansigma[2U*warp+1U],counts[warp]};"
           "state=dif_welford_mod_combine_"
        << suffix
        << "(state,other);}__syncthreads();}if(tid==0U){meansigma[0]=state.mean;"
           "meansigma[1]=state.sigma2/"
        << columns
        << ".0f;}__syncthreads();float mean=meansigma[0];float inverse=rsqrtf("
           "meansigma[1]+"
        << static_cast<float>(epsilon)
        << "f);unsigned long long modulation_row=row/" << rows_per_modulation
        << "ULL;for(unsigned long long pack=tid;pack<" << columns / 4U
        << "ULL;pack+=128ULL){unsigned long long base=row*" << columns
        << "ULL+pack*4ULL;for(unsigned inner=0U;inner<4U;++inner){unsigned long "
           "long column=pack*4ULL+inner,index=base+inner,modulation_index="
           "modulation_row*"
        << columns
        << "ULL+column;float normalized=inverse*(dif_load(x,index)-mean);"
           "normalized=dif_round(dif_load(weight,column)*normalized+"
           "dif_load(bias,column));float one_plus_scale=dif_round(1.0f+"
           "dif_load(scale,modulation_index));float scaled=dif_round("
           "normalized*one_plus_scale);dif_store(y,index,scaled+dif_load("
           "shift,modulation_index));}}}\n";
    return;
  }
  out << std::setprecision(17) << "extern \"C\" __global__ void "
      << function_name(op)
      << "(const dif_scalar* x,const dif_scalar* weight,const dif_scalar* "
         "bias,const dif_scalar* scale,const dif_scalar* shift,dif_scalar* "
         "y){extern __shared__ float reduction[];unsigned long long row="
         "blockIdx.x;if(row>="
      << rows
      << "ULL)return;float local=0.0f;for(unsigned long long col=threadIdx.x;"
         "col<"
      << columns
      << "ULL;col+=blockDim.x)local+=dif_load(x,row*" << columns
      << "ULL+col);reduction[threadIdx.x]=local;__syncthreads();for(unsigned "
         "stride=blockDim.x/2;stride>0;stride>>=1){if(threadIdx.x<stride)"
         "reduction[threadIdx.x]+=reduction[threadIdx.x+stride];"
         "__syncthreads();}float mean=reduction[0]/"
      << columns
      << ".0f;__syncthreads();local=0.0f;for(unsigned long long col=threadIdx.x;col<"
      << columns
      << "ULL;col+=blockDim.x){float centered=dif_load(x,row*" << columns
      << "ULL+col)-mean;local+=centered*centered;}reduction[threadIdx.x]=local;"
         "__syncthreads();for(unsigned stride=blockDim.x/2;stride>0;stride>>=1)"
         "{if(threadIdx.x<stride)reduction[threadIdx.x]+=reduction[threadIdx.x+"
         "stride];__syncthreads();}float inv=rsqrtf(reduction[0]/"
      << columns << ".0f+" << static_cast<float>(epsilon)
      << "f);unsigned long long modulation_row=row/" << rows_per_modulation
      << "ULL;for(unsigned long long col=threadIdx.x;col<" << columns
      << "ULL;col+=blockDim.x){unsigned long long i=row*" << columns
      << "ULL+col,mi=modulation_row*" << columns
      << "ULL+col;float normalized=dif_round((dif_load(x,i)-mean)*inv*"
         "dif_load(weight,col)+dif_load(bias,col));float one_plus_scale="
         "dif_round(1.0f+dif_load(scale,mi));float scaled=dif_round(normalized*"
         "one_plus_scale);dif_store(y,i,scaled+dif_load(shift,mi));}}\n";
}

void emit_rms_norm_modulate(std::ostringstream &out, const ir::Program &program,
                            const ir::Operation &op) {
  const auto &shape = program.tensor(op.inputs[0])->dims;
  const auto rows = shape[0];
  const auto cols = shape[1];
  const auto epsilon = op.f64(ir::AttrKey::Epsilon, 1.0e-5);
  const auto layout = static_cast<ir::ModulationLayout>(op.u64(
      ir::AttrKey::ModulationLayout,
      static_cast<std::uint64_t>(ir::ModulationLayout::ExplicitScaleShift)));
  if (layout == ir::ModulationLayout::SharedVectorDelta) {
    const auto vectors = program.tensor(op.inputs[2])->dims[0];
    const auto rows_per_vector = rows / vectors;
    const auto weight_offset = op.f64(ir::AttrKey::WeightOffset, 0.0);
    std::ostringstream weight_offset_literal;
    weight_offset_literal << std::scientific << std::setprecision(9)
                          << static_cast<float>(weight_offset);
    const auto reduction_tile =
        op.u64(ir::AttrKey::ReductionTileSize, 0U);
    if (program.tensor(op.inputs[0])->dtype != ir::DType::BF16 ||
        op.u64(ir::AttrKey::BlockSize, 256U) != 512U || cols != 6144U ||
        (reduction_tile != 2048U && reduction_tile != 8192U))
      fail("CUDA shared-vector rms_norm_modulate currently requires the "
           "source-faithful BF16 6144-wide reduction");
    if (reduction_tile == 2048U) {
      out << std::setprecision(17) << "extern \"C\" __global__ void "
          << function_name(op)
          << "(const dif_scalar* x,const dif_scalar* weight,const dif_scalar* "
             "vector,const dif_scalar* delta,dif_scalar* y){extern __shared__ "
             "float reduction[];unsigned long long row=blockIdx.x;unsigned tid="
             "threadIdx.x;if(row>="
          << rows << "ULL)return;unsigned long long base=row*6144ULL+(unsigned "
             "long long)tid*4ULL;float a0,a1,a2,a3;float m0=dif_load(x,base+"
             "2048ULL),m1=dif_load(x,base+2049ULL),m2=dif_load(x,base+2050ULL),"
             "m3=dif_load(x,base+2051ULL);float l0=dif_load(x,base),l1="
             "dif_load(x,base+1ULL),l2=dif_load(x,base+2ULL),l3=dif_load(x,"
             "base+3ULL);a0=fmaf(l0,l0,m0*m0);a1=fmaf(l1,l1,m1*m1);a2=fmaf("
             "l2,l2,m2*m2);a3=fmaf(l3,l3,m3*m3);float h0=dif_load(x,base+"
             "4096ULL),h1=dif_load(x,base+4097ULL),h2=dif_load(x,base+4098ULL),"
             "h3=dif_load(x,base+4099ULL);a0=fmaf(h0,h0,a0);a1=fmaf(h1,h1,a1);"
             "a2=fmaf(h2,h2,a2);a3=fmaf(h3,h3,a3);float local=((a0+a1)+a2)+a3;"
             "for(unsigned delta_step=16U;delta_step>0U;delta_step>>=1U)local+="
             "__shfl_xor_sync(0xffffffffU,local,delta_step);unsigned lane=tid&"
             "31U,warp=tid>>5U;if(lane==0U)reduction[warp]=local;__syncthreads();"
             "if(warp==0U){local=lane<16U?reduction[lane]:0.0f;for(unsigned "
             "delta_step=8U;delta_step>0U;delta_step>>=1U)local+="
             "__shfl_xor_sync(0xffffffffU,local,delta_step);if(lane==0U)"
             "reduction[0]=local;}__syncthreads();float mean,mean_eps,inv;asm "
             "volatile(\"div.full.f32 %0,%1,%2;\":\"=f\"(mean):\"f\"(reduction[0]),"
             "\"f\"(6144.0f));mean_eps=mean+"
          << static_cast<float>(epsilon)
          << "f;asm volatile(\"rsqrt.approx.ftz.f32 %0,%1;\":\"=f\"(inv):"
             "\"f\"(mean_eps));unsigned long long shared_base=(row/"
          << rows_per_vector
          << "ULL)*6144ULL;for(unsigned long long col=tid;col<6144ULL;col+="
             "blockDim.x){unsigned long long i=row*6144ULL+col;float xv="
             "dif_load(x,i),wv=dif_load(weight,col),base_value=dif_load(vector,"
             "shared_base+col),scale_delta=dif_load(delta,col),shift_delta="
             "dif_load(delta,6144ULL+col);float normalized,weighted,scale,"
             "scale_one,shift,result;asm volatile(\"mul.rn.f32 %0,%1,%2;\""
             ":\"=f\"(normalized):\"f\"(xv),\"f\"(inv));float weight_value=wv+"
          << weight_offset_literal.str()
          << "f;asm volatile(\"mul.rn.f32 %0,%1,%2;\":\"=f\"(weighted):"
             "\"f\"(normalized),\"f\"(weight_value));asm volatile(\"add.rn.f32 "
             "%0,%1,%2;\":\"=f\"(scale):\"f\"(base_value),\"f\"(scale_delta));"
             "scale_one=scale+1.0f;asm volatile("
             "\"add.rn.f32 %0,%1,%2;\":\"=f\"(shift):\"f\"(base_value),"
             "\"f\"(shift_delta));result=fmaf(scale_one,weighted,shift);"
             "dif_store(y,i,result);}}\n";
      return;
    }
    out << std::setprecision(17) << "extern \"C\" __global__ void "
        << function_name(op)
        << "(const dif_scalar* x,const dif_scalar* weight,const dif_scalar* "
           "vector,const dif_scalar* delta,dif_scalar* y){extern __shared__ "
           "float reduction[];unsigned long long row=blockIdx.x;unsigned tid="
           "threadIdx.x;if(row>="
        << rows << "ULL)return;unsigned long long base=row*" << cols
        << "ULL+(unsigned long long)tid*8ULL;float v0=dif_load(x,base),"
           "v1=dif_load(x,base+1ULL),v2=dif_load(x,base+2ULL),"
           "v3=dif_load(x,base+3ULL),v4=dif_load(x,base+4ULL),"
           "v5=dif_load(x,base+5ULL),v6=dif_load(x,base+6ULL),"
           "v7=dif_load(x,base+7ULL);float local=v1*v1;"
           "local=fmaf(v0,v0,local);local=fmaf(v2,v2,local);"
           "local=fmaf(v3,v3,local);local=fmaf(v4,v4,local);"
           "local=fmaf(v5,v5,local);local=fmaf(v6,v6,local);"
           "local=fmaf(v7,v7,local);if(tid<256U){float square,value;"
           "value=dif_load(x,base+4096ULL);asm volatile(\"mul.rn.f32 %0,%1,%1;\""
           ":\"=f\"(square):\"f\"(value));local=square+local;"
           "value=dif_load(x,base+4097ULL);asm volatile(\"mul.rn.f32 %0,%1,%1;\""
           ":\"=f\"(square):\"f\"(value));local=square+local;"
           "value=dif_load(x,base+4098ULL);asm volatile(\"mul.rn.f32 %0,%1,%1;\""
           ":\"=f\"(square):\"f\"(value));local=square+local;"
           "value=dif_load(x,base+4099ULL);asm volatile(\"mul.rn.f32 %0,%1,%1;\""
           ":\"=f\"(square):\"f\"(value));local=square+local;"
           "value=dif_load(x,base+4100ULL);asm volatile(\"mul.rn.f32 %0,%1,%1;\""
           ":\"=f\"(square):\"f\"(value));local=square+local;"
           "value=dif_load(x,base+4101ULL);asm volatile(\"mul.rn.f32 %0,%1,%1;\""
           ":\"=f\"(square):\"f\"(value));local=square+local;"
           "value=dif_load(x,base+4102ULL);asm volatile(\"mul.rn.f32 %0,%1,%1;\""
           ":\"=f\"(square):\"f\"(value));local=square+local;"
           "value=dif_load(x,base+4103ULL);asm volatile(\"mul.rn.f32 %0,%1,%1;\""
           ":\"=f\"(square):\"f\"(value));local=square+local;}"
           "for(unsigned delta_step=16U;delta_step>0U;delta_step>>=1U)"
           "local+=__shfl_xor_sync(0xffffffffU,local,delta_step);"
           "unsigned lane=tid&31U,warp=tid>>5U;if(lane==0U)reduction[warp]="
           "local;__syncthreads();if(warp==0U){local=lane<16U?reduction[lane]:"
           "0.0f;for(unsigned delta_step=8U;delta_step>0U;delta_step>>=1U)"
           "local+=__shfl_xor_sync(0xffffffffU,local,delta_step);if(lane==0U)"
           "reduction[0]=local;}__syncthreads();float mean,mean_eps,inv;"
           "asm volatile(\"div.full.f32 %0,%1,%2;\":\"=f\"(mean):\"f\"(reduction[0]),"
           "\"f\"(6144.0f));mean_eps=mean+"
        << static_cast<float>(epsilon)
        << "f;asm volatile(\"rsqrt.approx.ftz.f32 %0,%1;\":\"=f\"(inv):"
           "\"f\"(mean_eps));unsigned long long shared_base=(row/"
        << rows_per_vector
        << "ULL)*6144ULL;for(unsigned long long col=tid;col<6144ULL;col+="
           "blockDim.x){unsigned long long i=row*6144ULL+col;float xv="
           "dif_load(x,i),wv=dif_load(weight,col),base_value=dif_load(vector,"
           "shared_base+col),scale_delta=dif_load(delta,col),shift_delta="
           "dif_load(delta,6144ULL+col);float normalized,weighted,scale,"
           "scale_one,shift,result;asm volatile(\"mul.rn.f32 %0,%1,%2;\""
           ":\"=f\"(normalized):\"f\"(xv),\"f\"(inv));float weight_value=wv+"
        << weight_offset_literal.str()
        << "f;asm volatile(\"mul.rn.f32 %0,%1,%2;\":\"=f\"(weighted):"
           "\"f\"(normalized),\"f\"(weight_value));asm volatile(\"add.rn.f32 "
           "%0,%1,%2;\":\"=f\"(scale):\"f\"(base_value),\"f\"(scale_delta));"
           "scale_one=scale+1.0f;asm volatile("
           "\"add.rn.f32 %0,%1,%2;\":\"=f\"(shift):\"f\"(base_value),"
           "\"f\"(shift_delta));result=fmaf(scale_one,weighted,shift);"
           "dif_store(y,i,result);}}\n";
    return;
  }
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
  const auto input_width = program.tensor(op.inputs[0])->dims.back();
  const auto start = op.u64(ir::AttrKey::Start, 0U);
  const bool gate_first = op.boolean(ir::AttrKey::GateFirst, false);
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x, dif_scalar* y) {\n"
      << "  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;\n"
      << "  if(i<" << count << "ULL){ unsigned long long row=i/" << width
      << "ULL, col=i%" << width << "ULL; float value=dif_load(x,row*" << input_width
      << "ULL+" << start + (gate_first ? width : 0U)
      << "ULL+col); float gate=dif_load(x,row*" << input_width << "ULL+"
      << start + (gate_first ? 0U : width)
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

void emit_linear_addmm_prefill(std::ostringstream &out,
                               const ir::Program &program,
                               const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto width = program.tensor(op.inputs[2])->element_count();
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* x,const dif_scalar* weight,const dif_scalar* bias,"
         "dif_scalar* y){(void)x;(void)weight;unsigned long long i="
         "(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL)dif_store(y,i,dif_load(bias,i%" << width
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
  const auto *output = program.tensor(op.outputs[0]);
  const auto *gate = program.tensor(op.inputs[2]);
  const auto count = output->element_count();
  const auto width = output->dims.back();
  const auto rows = count / width;
  const auto gate_rows = gate->element_count() / width;
  const auto rows_per_gate = rows / gate_rows;
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* residual, const dif_scalar* branch, const dif_scalar* gate, dif_scalar* y) {\n"
      << "  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;"
         " if(i<"
      << count
      << "ULL){unsigned long long row=i/" << width
      << "ULL,gate_index=(row/" << rows_per_gate << "ULL)*" << width
      << "ULL+i%" << width
      << "ULL;dif_store(y,i,dif_load(residual,i)+"
         "dif_round(dif_load(gate,gate_index)*dif_load(branch,i)));}\n}\n";
}

void emit_qk_norm_rope(std::ostringstream &out, const ir::Program &program,
                       const ir::Operation &op) {
  const auto &shape = program.tensor(op.inputs[0])->dims;
  const auto heads = shape[shape.size() - 2U];
  const auto dim = shape.back();
  const auto sequence =
      program.tensor(op.inputs[0])->element_count() / (heads * dim);
  const auto input_sequence =
      shape.size() == 4U ? shape[1] : shape[0];
  const auto &table_shape = program.tensor(op.inputs[2])->dims;
  const auto table_sequence =
      table_shape.size() == 3U ? table_shape[1] : table_shape[0];
  const auto table_start = op.u64(ir::AttrKey::Start, 0U);
  const auto rotary = op.u64(ir::AttrKey::RotaryDim, dim);
  const auto half = rotary / 2U;
  const auto table_width = program.tensor(op.inputs[2])->dims.back();
  const auto epsilon = op.f64(ir::AttrKey::Epsilon, 1.0e-5);
  const auto implementation = op.u64(ir::AttrKey::Implementation, 1U);
  const auto rotary_layout = static_cast<ir::RotaryLayout>(op.u64(
      ir::AttrKey::RotaryLayout,
      static_cast<std::uint64_t>(ir::RotaryLayout::HalfSplit)));
  if (implementation == 2U && rotary_layout == ir::RotaryLayout::Interleaved &&
      program.tensor(op.inputs[0])->dtype == ir::DType::BF16 && dim == 128U &&
      program.tensor(op.inputs[2])->dtype == ir::DType::F32 &&
      table_width * 2U == rotary) {
    out << std::setprecision(17)
        << "extern \"C\" __global__ void " << function_name(op)
        << "(const dif_bf16* x,const dif_bf16* weight,const dif_f32* cosv,"
           "const dif_f32* sinv,dif_bf16* y){extern __shared__ float "
           "reduction[];unsigned long long row=blockIdx.x;unsigned tid="
           "threadIdx.x;if(row>="
        << sequence * heads
        << "ULL)return;unsigned long long base=row*128ULL;float sigma2=0.0f;"
           "if(tid<32U){unsigned long long lane_base=base+(unsigned long long)tid;"
           "float v0=dif_load_bf16(x,lane_base),v1=dif_load_bf16(x,lane_base+32ULL),"
           "v2=dif_load_bf16(x,lane_base+64ULL),v3=dif_load_bf16(x,lane_base+96ULL),"
           "s0,s1,s2,s3;asm volatile(\"mul.rn.f32 %0,%1,%1;\":\"=f\"(s0):\"f\"(v0));"
           "asm volatile(\"mul.rn.f32 %0,%1,%1;\":\"=f\"(s1):\"f\"(v1));"
           "asm volatile(\"mul.rn.f32 %0,%1,%1;\":\"=f\"(s2):\"f\"(v2));"
           "asm volatile(\"mul.rn.f32 %0,%1,%1;\":\"=f\"(s3):\"f\"(v3));"
           "sigma2=((s0+s1)+s2)+s3;for(unsigned offset=1U;offset<32U;offset<<=1U)"
           "sigma2=sigma2+__shfl_down_sync(0xffffffffU,sigma2,offset);}if(tid==0U)"
           "reduction[0]=sigma2*0.0078125f;__syncthreads();float inv=rsqrtf("
           "reduction[0]+"
        << static_cast<float>(epsilon)
        << "f);if(tid<128U){unsigned pair=tid/2U,even=pair*2U;float e="
           "dif_round_bf16(dif_load_bf16(x,base+even)*inv);e=dif_round_bf16("
           "e*dif_load_bf16(weight,even));float o=dif_round_bf16(dif_load_bf16("
           "x,base+even+1U)*inv);o=dif_round_bf16(o*dif_load_bf16(weight,even+1U));"
           "unsigned long long token=row/"
        << heads << "ULL,table_token=(token/" << input_sequence << "ULL)*"
        << table_sequence << "ULL+" << table_start << "ULL+token%"
        << input_sequence << "ULL,table=table_token*" << table_width
        << "ULL+pair;float c=dif_load_f32(cosv,table),s=dif_load_f32(sinv,table),"
           "first,second,result;if(tid&1U){asm volatile(\"mul.rn.f32 %0,%1,%2;\":"
           "\"=f\"(first):\"f\"(e),\"f\"(s));asm volatile(\"mul.rn.f32 %0,%1,%2;\":"
           "\"=f\"(second):\"f\"(o),\"f\"(c));asm volatile(\"add.rn.f32 %0,%1,%2;\":"
           "\"=f\"(result):\"f\"(first),\"f\"(second));}else{asm volatile("
           "\"mul.rn.f32 %0,%1,%2;\":\"=f\"(first):\"f\"(e),\"f\"(c));"
           "asm volatile(\"mul.rn.f32 %0,%1,%2;\":\"=f\"(second):\"f\"(o),"
           "\"f\"(s));asm volatile(\"sub.rn.f32 %0,%1,%2;\":\"=f\"(result):"
           "\"f\"(first),\"f\"(second));}dif_store_bf16(y,base+tid,result);}}\n";
    return;
  }
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
        << "f));unsigned long long token=row/" << heads
        << "ULL,table_token=(token/" << input_sequence << "ULL)*"
        << table_sequence << "ULL+" << table_start << "ULL+token%"
        << input_sequence << "ULL;if(tid<"
        << half
        << "U){unsigned lane=tid;float value0=dif_load(x,base+lane);"
           "float value1=dif_load(x,base+lane+"
        << half
        << "ULL);float norm0=dif_round(value0*inv*dif_load(weight,lane));"
           "float norm1=dif_round(value1*inv*dif_load(weight,lane+"
        << half
        << "ULL));unsigned long long table=table_token*" << table_width
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
  // GQA (KvHeads attr): query head h reads kv head h/(H/KvHeads).  When
  // KvHeads == H the emitted source is BYTE-IDENTICAL to the pre-GQA kernel
  // (kv_head_expr collapses to "h"), so recorded programs keep their
  // generated-source identity.
  const auto kv_heads = op.u64(ir::AttrKey::KvHeads, heads);
  const auto group = heads / kv_heads;
  const std::string kv_head_expr =
      group == 1U ? std::string("h")
                  : "h/" + std::to_string(group) + "ULL";
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
      << dim << "ULL+d),dif_load(k,(ks*" << kv_heads << "ULL+"
      << kv_head_expr << ")*" << dim
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
      << kv_heads << "ULL+" << kv_head_expr << ")*" << dim
      << "ULL+d),acc);dif_store(y,(qs*" << heads
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
  const auto input_width = program.tensor(op.inputs[1])->dims.back();
  const auto start = op.u64(ir::AttrKey::Start, 0U);
  const bool gate_first = op.boolean(ir::AttrKey::GateFirst, false);
  // Thread i owns one element of the packed [.., 2W] input gradient; the
  // value half receives silu(gate)*g, the gate half dsilu(gate)*value*g.
  out << "extern \"C\" __global__ void " << function_name(op)
      << "(const dif_scalar* grad_output,const dif_scalar* x,dif_scalar* grad_input){"
         "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<"
      << count << "ULL){unsigned long long row=i/" << input_width
      << "ULL,col=i%" << input_width << "ULL;if(col<" << start
      << "ULL||col>=" << start + width * 2U
      << "ULL){dif_store(grad_input,i,0.0f);return;}unsigned long long lane=col-"
      << start << "ULL,cw=lane<" << width << "ULL?lane:lane-" << width
      << "ULL,base=row*" << input_width << "ULL+" << start
      << "ULL;float value=dif_load(x,base+" << (gate_first ? width : 0U)
      << "ULL+cw);float gate=dif_load(x,base+" << (gate_first ? 0U : width)
      << "ULL+cw);float sigmoid=1.0f/(1.0f+expf(-gate));"
         "float upstream=dif_load(grad_output,row*"
      << width << "ULL+cw);int is_value_slot=lane"
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
  const auto kv_heads = op.u64(ir::AttrKey::KvHeads, heads);
  const auto group = heads / kv_heads;
  // Collapses to "h" when KvHeads == H, keeping the emitted source
  // byte-identical to the pre-GQA kernel for every existing program.
  const std::string kv_head_expr =
      group == 1U ? std::string("h")
                  : "h/" + std::to_string(group) + "ULL";
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
      << kv_heads << "ULL+" << kv_head_expr << ")*" << dim
      << "ULL;for(unsigned long long d=0ULL;d<" << dim
      << "ULL;++d)score=fmaf(" << load << "(q,qb+d)," << load
      << "(k,kb+d),score);score*=" << scale
      << "f;maximum=fmaxf(maximum,score);}float denominator=0.0f;"
         "for(unsigned long long ks=0ULL;ks<kend;++ks){float score=0.0f;"
         "unsigned long long kb=(ks*"
      << kv_heads << "ULL+" << kv_head_expr << ")*" << dim
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
  const auto *k_tensor = program.tensor(op.inputs[2]);
  const auto sequence = q->dims[0];
  const auto heads = q->dims[1];
  const auto dim = q->dims[2];
  const auto kv_heads = k_tensor->dims[1];
  const auto group = heads / kv_heads;
  const auto count = sequence * heads * dim;
  const auto scale = static_cast<float>(op.f64(
      ir::AttrKey::AttentionScale, 1.0 / std::sqrt(static_cast<double>(dim))));
  const bool causal = op.boolean(ir::AttrKey::Causal, false);
  const auto *scalar = typed_scalar(q->dtype);
  const auto *load = typed_load(q->dtype);
  const auto *store = typed_store(q->dtype);
  // Thread i = (s,h,d) over the QUERY geometry computes dq[s,h,d]; the
  // threads with h < KvHeads additionally own dk/dv[s,h,d], accumulating in
  // F32 registers across every query AND every query head of their group
  // (GQA grouped-KV gradient accumulation).  P is recomputed from Q,K and
  // the saved F32 logsumexp; delta = rowsum(dO*O) uses the forward output.
  // O(S) score recomputations per thread — the O(S^2) recompute path;
  // acceptable at gate scale, cuDNN SDPA backward stays future work.
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
      // dq[s,h,d]: iterate keys ks<kend(s) against kv head h/group.
         "float dq=0.0f;{unsigned long long qb=base,kh=h/"
      << group << "ULL,kend="
      << (causal ? "s+1ULL" : std::to_string(sequence) + "ULL")
      << ";float row_lse=dif_load_f32(lse,row);float delta=0.0f;"
         "for(unsigned long long e=0ULL;e<"
      << dim << "ULL;++e)delta=fmaf(" << load << "(grad_output,qb+e)," << load
      << "(forward_output,qb+e),delta);"
         "for(unsigned long long ks=0ULL;ks<kend;++ks){unsigned long long kb="
         "(ks*"
      << kv_heads << "ULL+kh)*" << dim
      << "ULL;float score=0.0f,projected=0.0f;"
         "for(unsigned long long e=0ULL;e<"
      << dim << "ULL;++e){score=fmaf(" << load << "(q,qb+e)," << load
      << "(k,kb+e),score);projected=fmaf(" << load << "(grad_output,qb+e),"
      << load << "(v,kb+e),projected);}float probability=expf(score*" << scale
      << "f-row_lse);dq=fmaf(probability*(projected-delta)*" << scale
      << "f," << load << "(k,kb+d),dq);}}" << store
      << "(grad_q,i,dq);"
      // dk/dv[s,kvh,d] for threads with h < KvHeads: iterate the group's
      // query heads and every query (qs>=s when causal).
         "if(h<"
      << kv_heads << "ULL){unsigned long long kb=(s*" << kv_heads
      << "ULL+h)*" << dim
      << "ULL+d;float dk=0.0f,dv=0.0f;unsigned long long kvb=(s*" << kv_heads
      << "ULL+h)*" << dim
      << "ULL;for(unsigned long long g=0ULL;g<" << group
      << "ULL;++g){unsigned long long qh=h*" << group
      << "ULL+g;for(unsigned long long qs="
      << (causal ? "s" : "0ULL") << ";qs<" << sequence
      << "ULL;++qs){unsigned long long qb=(qs*" << heads << "ULL+qh)*" << dim
      << "ULL;float row_lse=dif_load_f32(lse,qs*" << heads
      << "ULL+qh);float score=0.0f,projected=0.0f,delta=0.0f;"
         "for(unsigned long long e=0ULL;e<"
      << dim << "ULL;++e){score=fmaf(" << load << "(q,qb+e)," << load
      << "(k,kvb+e),score);projected=fmaf(" << load << "(grad_output,qb+e),"
      << load << "(v,kvb+e),projected);delta=fmaf(" << load
      << "(grad_output,qb+e)," << load
      << "(forward_output,qb+e),delta);}float probability=expf(score*"
      << scale << "f-row_lse);dk=fmaf(probability*(projected-delta)*" << scale
      << "f," << load << "(q,qb+d),dk);dv=fmaf(probability," << load
      << "(grad_output,qb+d),dv);}}" << store << "(grad_k,kb,dk);" << store
      << "(grad_v,kb,dv);}}}\n"
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
  if (opcode == ir::Opcode::ResidualGate && input_slot == 2U)
    return false;
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
      const auto *gate_description = program.tensor(op.inputs[2]);
      auto gate_index = index;
      if (gate_description->element_count() != output->element_count()) {
        const auto width = output->dims.back();
        const auto rows = output->element_count() / width;
        const auto gate_rows = gate_description->element_count() / width;
        gate_index = "((" + index + ")/" + std::to_string(width) +
                     "ULL/" + std::to_string(rows / gate_rows) + "ULL)*" +
                     std::to_string(width) + "ULL+(" + index + ")%" +
                     std::to_string(width) + "ULL";
      }
      const auto gate = value(op.inputs[2], gate_index);
      return residual + "+" + std::string(typed_round(output->dtype)) + "(" +
             gate + "*" + branch + ")";
    }
    case ir::Opcode::SwiGlu: {
      const auto width = output->dims.back();
      const auto input_width = program.tensor(op.inputs[0])->dims.back();
      const auto start = op.u64(ir::AttrKey::Start, 0U);
      const bool gate_first = op.boolean(ir::AttrKey::GateFirst, false);
      const auto [row, column] = row_column(op, index, width);
      const auto lane = [&](std::uint64_t offset) {
        return row + "*" + std::to_string(input_width) + "ULL+" +
               std::to_string(start + offset) + "ULL+" + column;
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
        tensor.dtype != ir::DType::I32 && tensor.dtype != ir::DType::I8 &&
        tensor.dtype != ir::DType::FP8E4M3 &&
        tensor.dtype != ir::DType::FP8E8M0 &&
        tensor.dtype != ir::DType::Bool)
      fail("CUDA source emitter admits mixed f32/bf16/f16 plus i32 indices "
           "and packed i8/bool constants");
  }
  emit_header(source);
  std::unordered_map<std::string, std::string> pooled_entrypoints;
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
      } else if (op.inputs.size() == 3U &&
                 op.u64(ir::AttrKey::LinearBiasMode,
                        static_cast<std::uint64_t>(
                            ir::LinearBiasMode::Epilogue)) ==
                     static_cast<std::uint64_t>(
                         ir::LinearBiasMode::Addmm)) {
        generated.entrypoints.emplace(op.id, function_name(op));
        begin_float_operation(source, operation_float_dtype(program, op));
        emit_linear_addmm_prefill(source, program, op);
        end_float_operation(source);
      }
      continue;
    }
    if (op.opcode == ir::Opcode::Barrier ||
        op.opcode == ir::Opcode::LinearInt8Scaled ||
        op.opcode == ir::Opcode::LinearInt8WeightScaled ||
        op.opcode == ir::Opcode::LinearFp8BlockScaled ||
        op.opcode == ir::Opcode::Conv2d ||
        op.opcode == ir::Opcode::Conv3d ||
        (op.opcode == ir::Opcode::Attention &&
         op.u64(ir::AttrKey::Implementation, 1U) != 1U))
      continue;
    if (const auto region = elementwise_regions.find(op.id);
        region != elementwise_regions.end()) {
      generated.entrypoints.emplace(op.id, function_name(op));
      std::vector<std::uint32_t> arguments;
      emit_fused_elementwise(source, program, region->second, arguments);
      generated.launch_inputs.emplace(op.id, std::move(arguments));
      continue;
    }
    const auto identity = kernel_identity(program, op);
    if (const auto found = pooled_entrypoints.find(identity);
        found != pooled_entrypoints.end()) {
      generated.entrypoints.emplace(op.id, found->second);
      continue;
    }
    const auto entrypoint = function_name(op);
    pooled_entrypoints.emplace(identity, entrypoint);
    generated.entrypoints.emplace(op.id, entrypoint);
    if (op.opcode == ir::Opcode::Cast) {
      emit_cast(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::RotaryPosition) {
      emit_rotary_position(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::RotaryFrequency) {
      emit_rotary_frequency(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::RotaryApply) {
      emit_rotary_apply(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::BooleanMaskToBias) {
      emit_boolean_mask_to_bias(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::QuantizeInt8Rows) {
      emit_quantize_int8_rows(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::DequantizeInt8Blocks) {
      emit_dequantize_int8_blocks(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::QuantizeFp8Rows) {
      emit_quantize_fp8_rows(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::LinearFp8Scaled) {
      emit_linear_fp8_output_scale(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::QuantizeFp8Blocks32) {
      emit_quantize_fp8_blocks32(source, program, op);
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
    case ir::Opcode::Gelu:
      emit_gelu(source, program, op);
      break;
    case ir::Opcode::Sigmoid:
      emit_sigmoid(source, program, op);
      break;
    case ir::Opcode::Reshape:
      emit_reshape(source, program, op);
      break;
    case ir::Opcode::BroadcastTo:
      emit_broadcast_to(source, program, op);
      break;
    case ir::Opcode::Slice:
      emit_slice(source, program, op);
      break;
    case ir::Opcode::RotaryFrequency:
    case ir::Opcode::RotaryApply:
    case ir::Opcode::BooleanMaskToBias:
      break;
    case ir::Opcode::RmsNorm:
      emit_rms_norm(source, program, op);
      break;
    case ir::Opcode::LayerNorm:
      emit_layer_norm(source, program, op);
      break;
    case ir::Opcode::LayerNormModulate:
      emit_layer_norm_modulate(source, program, op);
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
    case ir::Opcode::EulerVelocityStep:
      emit_euler_velocity_step(source, program, op);
      break;
    case ir::Opcode::Permute:
      emit_permute(source, program, op);
      break;
    case ir::Opcode::Concat:
      emit_concat(source, program, op);
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
      emit_conv1d(source, program, op);
      break;
    case ir::Opcode::Conv2d:
      break;
    case ir::Opcode::ChannelRmsNorm:
      emit_channel_rms_norm(source, program, op);
      break;
    case ir::Opcode::GroupNorm:
      emit_group_norm(source, program, op);
      break;
    case ir::Opcode::UpsampleNearest2d:
      emit_upsample_nearest_2d(source, program, op);
      break;
    case ir::Opcode::PadConstant:
      emit_pad_constant(source, program, op);
      break;
    case ir::Opcode::PadReflect:
      emit_pad_reflect(source, program, op);
      break;
    case ir::Opcode::Conv3d:
      break;
    case ir::Opcode::SnakeBeta:
      emit_snake_beta(source, program, op);
      break;
    case ir::Opcode::QuantizeInt8Rows:
    case ir::Opcode::LinearInt8Scaled:
    case ir::Opcode::QuantizeFp8Rows:
    case ir::Opcode::LinearFp8Scaled:
    case ir::Opcode::QuantizeFp8Blocks32:
    case ir::Opcode::LinearFp8BlockScaled:
    case ir::Opcode::DequantizeInt8Blocks:
    case ir::Opcode::LinearInt8WeightScaled:
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
