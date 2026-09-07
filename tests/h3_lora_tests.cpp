#include "dif/frontend/h3_lora.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/weights/safetensors.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <unistd.h>

namespace {
void require(bool yes, const char *message) {
  if (!yes) throw std::runtime_error(message);
}
void refuses(const std::function<void()> &f) {
  bool rejected = false;
  try { f(); } catch (const std::exception &) { rejected = true; }
  require(rejected, "invalid adapter was accepted");
}
dif::runtime::Tensor values(std::vector<std::uint64_t> dims, float value) {
  dif::ir::TensorDesc desc{1U, dif::ir::DType::F32, dif::ir::TensorRole::Input, dims};
  auto result = dif::runtime::zeros(desc);
  for (auto &x : result.f32()) x = value;
  return result;
}
struct Fixture {
  dif::ir::Program program;
  dif::weights::WeightBundle bundle;
  dif::runtime::TensorMap inputs;
};
Fixture fixture(std::uint64_t width = 4U) {
  using namespace dif::ir;
  Fixture f;
  const auto in = std::uint32_t(TensorRole::Input);
  const auto con = std::uint32_t(TensorRole::Constant);
  const auto out = std::uint32_t(TensorRole::Output);
  const auto mid = std::uint32_t(TensorRole::Internal);
  f.program.tensors = {{1,DType::BF16,in,{2,width}},
    {2,DType::BF16,con,{3*width,width}},
    {3,DType::BF16,mid,{width,width}}, {4,DType::BF16,mid,{width,width}},
    {5,DType::BF16,mid,{width,width}},
    {6,DType::BF16,out,{2,2,width/2}}, {7,DType::BF16,out,{2,2,width/2}},
    {8,DType::BF16,out,{2,2,width/2}}, {9,DType::BF16,con,{2*width,width}},
    {10,DType::BF16,out,{2,2*width}}};
  f.program.operations = {{1,Opcode::H3DeinterleaveQkvWeight,{2},{3,4,5},
      {Attribute::u64(AttrKey::Heads,2),Attribute::u64(AttrKey::HeadDim,width/2)}},
    {2,Opcode::Linear,{1,3},{6},{}}, {3,Opcode::Linear,{1,4},{7},{}},
    {4,Opcode::Linear,{1,5},{8},{}}, {5,Opcode::Linear,{1,9},{10},{}}};
  for (const auto &[id,name] : std::array<std::pair<std::uint32_t,std::string>,2>{{
      {2,"blocks.0.attn.qkv_proj.weight"},{9,"blocks.0.mlp.fc1.weight"}}}) {
    const auto *t = f.program.tensor(id);
    f.bundle.bindings.push_back({id,0,name,t->dtype,t->dims,0,t->byte_count()});
  }
  for (const auto &t : f.program.tensors)
    if (t.has_role(TensorRole::Input) || t.has_role(TensorRole::Constant))
      f.inputs.emplace(t.id, dif::runtime::convert_float_tensor(
          values(t.dims,t.id == 1 ? 0.25F : 0.125F),DType::BF16));
  dif::ir::verify(f.program);
  return f;
}
std::filesystem::path write_adapter(const std::filesystem::path &dir,
    const std::string &file, bool legacy=false, bool zero=false,
    bool incomplete=false, bool unknown=false, std::uint64_t width=4U) {
  using namespace dif;
  std::vector<std::pair<std::string,runtime::Tensor>> tensors;
  for (const auto &module : {std::string("attn.qkv_proj"),std::string("mlp.fc1")}) {
    auto prefix = "diffusion_model.blocks.0." + module;
    if (legacy) {
      prefix = "lora_unet_blocks_0_" + module;
      std::replace(prefix.begin(),prefix.end(),'.','_');
    }
    if (unknown) prefix = "diffusion_model.blocks.50." + module;
    tensors.push_back({prefix + (legacy ? ".lora_down.weight" : ".lora_A.weight"),values({2,width},0.25F)});
    if (!incomplete) {
      auto b = values({module == "mlp.fc1" ? 2U*width : 3U*width,2},zero ? 0.0F : 0.125F);
      if (!zero)
        for (std::size_t i=0;i<b.f32().size();++i)
          b.f32()[i] *= float(i / 2 + 1);
      tensors.push_back({prefix + (legacy ? ".lora_up.weight" : ".lora_B.weight"),std::move(b)});
    }
  }
  std::vector<weights::SafeTensorWriteSpec> specs;
  for (const auto &[name,t] : tensors) specs.push_back({name,t.dtype,t.dims});
  const auto path=dir/file;
  weights::SafeTensorWriter writer(path,std::move(specs));
  for (const auto &[name,t] : tensors) writer.append(name,{t.data(),t.byte_size()});
  writer.finish(); return path;
}
void check_delta(const Fixture &f, const dif::frontend::H3LoraResult &adapted,
                 float multiplier) {
  auto inputs=f.inputs; inputs.insert(adapted.constants.begin(),adapted.constants.end());
  auto cpu=dif::runtime::make_cpu_executor();
  dif::runtime::RunOptions options; options.warmups=0; options.iterations=1;
  const auto before=cpu->run(f.program,f.inputs,options);
  const auto after=cpu->run(adapted.program,inputs,options);
  for (const auto &[id,offset] : std::array<std::pair<std::uint32_t,std::size_t>,4>{{{6,0},{7,4},{8,8},{10,0}}}) {
    const auto b=dif::runtime::convert_float_tensor(before.outputs.at(id),dif::ir::DType::F32);
    const auto a=dif::runtime::convert_float_tensor(after.outputs.at(id),dif::ir::DType::F32);
    const auto columns=a.element_count()/2;
    for (std::size_t i=0;i<a.element_count();++i) {
      const auto delta=0.25F * 2.0F * (float(i%columns+offset+1)*0.125F) * multiplier;
      const auto rounded_delta=dif::runtime::bf16_to_float(dif::runtime::float_to_bf16(delta));
      const auto expected=dif::runtime::bf16_to_float(dif::runtime::float_to_bf16(b.f32()[i]+rounded_delta));
      require(a.f32()[i]==expected,"activation delta/layout mismatch");
    }
  }
  for (const auto &[id,t] : f.inputs)
    require(inputs.at(id).byte_size()==t.byte_size() &&
      std::memcmp(inputs.at(id).data(),t.data(),t.byte_size())==0,"base mutated");
}
void cuda_gate(const std::filesystem::path &dir) {
  // Real H256 math, tiny token count. This is a kernel/mechanics fixture, not
  // a trained-adapter or decoded model-quality admission.
  using namespace dif;
  auto f=fixture(256);
  const auto path=write_adapter(dir,"cuda.safetensors",false,false,false,false,256);
  auto adapted=frontend::add_h3_lora(f.program,f.bundle,{{path,0.7F}});
  auto inputs=f.inputs; inputs.insert(adapted.constants.begin(),adapted.constants.end());
  runtime::RunOptions options; options.warmups=0; options.iterations=1;
  options.cache_directory=dir/"cuda-cache";
  auto cpu=runtime::make_cpu_executor();
  const auto reference=cpu->run(adapted.program,inputs,options);
  auto cuda=runtime::make_cuda_executor();
  auto compare=[&](const runtime::RunResult &result,const char *label) {
    for (const auto &[id,expected] : reference.outputs) {
      const auto actual=runtime::convert_float_tensor(result.outputs.at(id),ir::DType::F32);
      const auto want=runtime::convert_float_tensor(expected,ir::DType::F32);
      for (std::size_t i=0;i<want.element_count();++i)
        require(std::isfinite(actual.f32()[i]) && actual.f32()[i]==want.f32()[i],label);
    }
  };
  compare(cuda->run(adapted.program,inputs,options),"CUDA BF16 activation overlay disagrees with CPU");
  std::cout << "H3 LoRA CUDA BF16 PASS: fractional F32 strength0.7, CPU byte-exact values\n";
  std::vector<weights::SafeTensorWriteSpec> specs{{"__meta__.h3_convrot",ir::DType::I32,{14}}};
  for (const auto slot : {0U,2U}) {
    const auto n=slot==0?768U:512U;
    specs.push_back({"block.0.convrot_weight."+std::to_string(slot),ir::DType::I8,{n,256}});
    specs.push_back({"block.0.convrot_scale."+std::to_string(slot),ir::DType::F32,{n}});
  }
  const auto cache=dir/"convrot.safetensors";
  weights::SafeTensorWriter writer(cache,std::move(specs));
  std::array<std::uint32_t,14> metadata{0x44494643U,1U,256U,1U,1U,4U};
  writer.append("__meta__.h3_convrot",{reinterpret_cast<const std::uint8_t*>(metadata.data()),sizeof(metadata)});
  for (const auto slot : {0U,2U}) {
    const auto n=slot==0?768U:512U;
    // Native ConvRot uses the normalized 4-way reflection, not Walsh DC:
    // each output of a constant [c,c,c,c] is .5*(3*c-c)==c at every stage.
    // See tools/difh3convrot.cpp stage64 and quantize_h3_convrot.
    std::vector<std::uint8_t> quantized(n*256U,127U);
    const auto scale=values({n},0.125F/127.0F);
    writer.append("block.0.convrot_weight."+std::to_string(slot),quantized);
    writer.append("block.0.convrot_scale."+std::to_string(slot),{scale.data(),scale.byte_size()});
  }
  writer.finish();
  options.convrot_linear_bindings=frontend::h3_lora_convrot_bindings(f.program,f.bundle,cache,1,1,1);
  require(options.convrot_linear_bindings.size()==4,"ConvRot counted wrong base matrices");
  // Exercise both resident and streamed slots in the SAME prepared executor.
  options.convrot_linear_bindings.back().resident=false;
  auto prepared=cuda->prepare(adapted.program,inputs,options);
  for (unsigned run=0;run<2;++run) {
    const auto result=prepared->run(inputs,options);
    require(result.convrot_int8_linears.size()==4,"ConvRot skipped a base projection");
    compare(result,"CUDA ConvRot activation overlay disagrees with CPU");
  }
  std::cout << "H3 LoRA CUDA PASS: BF16 and mixed-residency ConvRot, 4 base projections, 2 repeated runs, CPU byte-exact values\n";
}
}

int main(int argc, char **argv) {
  try {
    char pattern[]="/tmp/dif-h3-lora-tests.XXXXXX";
    const auto *temporary=mkdtemp(pattern);
    require(temporary!=nullptr,"mkdtemp failed");
    const std::filesystem::path dir=temporary;
    auto f=fixture();
    const auto canonical=write_adapter(dir,"canonical.safetensors");
    const auto legacy=write_adapter(dir,"legacy.safetensors",true);
    const auto adapted=dif::frontend::add_h3_lora(f.program,f.bundle,{{canonical,1}});
    require(adapted.adapters==2 && adapted.projections==4,"incorrect adapter count");
    check_delta(f,adapted,1);
    check_delta(f,dif::frontend::add_h3_lora(f.program,f.bundle,{{legacy,1}}),1);
    check_delta(f,dif::frontend::add_h3_lora(f.program,f.bundle,{{canonical,-1}}),-1);
    check_delta(f,dif::frontend::add_h3_lora(f.program,f.bundle,{{canonical,0.7F}}),0.7F);
    check_delta(f,dif::frontend::add_h3_lora(f.program,f.bundle,{{canonical,1},{legacy,1}}),2);
    for (const auto &path : {write_adapter(dir,"zero.safetensors",false,true),
         write_adapter(dir,"incomplete.safetensors",false,false,true),
         write_adapter(dir,"unknown.safetensors",false,false,false,true)})
      refuses([&]{dif::frontend::add_h3_lora(f.program,f.bundle,{{path,1}});});
    refuses([&]{dif::frontend::add_h3_lora(f.program,f.bundle,{{canonical,0}});});
    if (argc==2 && std::string(argv[1])=="--cuda") cuda_gate(dir);
    else if (argc!=1) throw std::runtime_error("usage: dif_h3_lora_tests [--cuda]");
    std::cout << "H3 LoRA CPU PASS: canonical/legacy, QKV rank3, gate-first FC1 rank2, negative/multiple adapters, immutable base, zero/incomplete/unknown rejection; fixtures=" << dir << '\n';
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
