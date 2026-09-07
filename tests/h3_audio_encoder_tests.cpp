#include "dif/frontend/h3_audio_encoder.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/support/error.hpp"

#include <cmath>
#include <iostream>

namespace {
void require(bool result, const char *message) { if (!result) dif::fail(message); }
template <class F> void rejects(F run) {
  bool rejected = false;
  try { run(); } catch (const dif::Error &) { rejected = true; }
  require(rejected, "invalid audio encoder contract was accepted");
}
}

int main() {
  try {
    using namespace dif;
    const auto config = frontend::h3_audio_encoder_config(json::parse(R"({
      "encoder_dim":64,"encoder_rates":[2,4,4,5,5],"latent_dim":2048,
      "latent_channels":32,"num_attention_heads":8,"eps":0.00001})"));
    const auto build = frontend::build_h3_audio_encoder_program(2, 3201, config);
    ir::verify(ir::decode(ir::encode(build.program)));
    require(build.program.tensor(build.mean_output_id)->dims ==
                std::vector<std::uint64_t>{2,32,5}, "native pad/hop geometry changed");
    unsigned snakes=0, means=0, attention=0;
    for (const auto &op : build.program.operations) {
      snakes += op.opcode == ir::Opcode::SnakeAlpha;
      means += op.opcode == ir::Opcode::ReduceMean;
      if (op.opcode == ir::Opcode::Attention) {
        ++attention;
        require(op.boolean(ir::AttrKey::Causal, false) &&
                    op.u64(ir::AttrKey::Implementation, 0) == 2 &&
                    build.program.tensor(op.inputs[0])->dtype == ir::DType::BF16,
                "native BF16 causal attention changed");
      }
    }
    require(snakes == 36 && means == 2 && attention == 1, "native graph operation census changed");
    for (const auto &binding : build.bindings)
      require(!binding.name.starts_with("decoder.") && !binding.name.starts_with("logs_proj."),
              "encoder loaded unused decoder/posterior weights");
    rejects([&] { auto bad=config; bad.encoder_rates[0]=0;
                  frontend::build_h3_audio_encoder_program(2,3201,bad); });
    rejects([&] { auto bad=config; bad.latent_channels=31;
                  frontend::build_h3_audio_encoder_program(2,3201,bad); });

    ir::Program p;
    p.tensors = {{1,ir::DType::F32,ir::TensorRole::Input,{2,3,4}},
                 {2,ir::DType::F32,ir::TensorRole::Output,{2,4}}};
    p.operations = {{1,ir::Opcode::ReduceMean,{1},{2},
                     {ir::Attribute::u64(ir::AttrKey::Axis,1)}}};
    auto x=runtime::zeros(p.tensors[0]);
    for (std::size_t i=0; i<x.f32().size(); ++i) x.f32()[i]=static_cast<float>(i);
    runtime::RunOptions options; options.warmups=0; options.iterations=1;
    const auto result=runtime::make_cpu_executor()->run(p,{{1,x}},options);
    for (std::size_t i=0; i<8; ++i)
      require(result.outputs.at(2).f32()[i] == static_cast<float>((i/4)*12+i%4+4),
              "single-axis mean translation is incorrect");
    rejects([&] { auto bad=p; bad.operations[0].attributes.clear(); ir::verify(bad); });
    p.tensors[0].dims={2,3,4}; p.tensors[1].dims={2,3,4};
    p.tensors.push_back({3,ir::DType::F32,ir::TensorRole::Constant,{1,3,1}});
    p.operations={{1,ir::Opcode::SnakeAlpha,{1,3},{2},{ir::Attribute::f64(ir::AttrKey::Epsilon,1e-9)}}};
    auto alpha=runtime::zeros(p.tensors[2]);
    alpha.f32()[0]=0.3F; alpha.f32()[1]=1.1F; alpha.f32()[2]=2.0F;
    const auto snake=runtime::make_cpu_executor()->run(p,{{1,x},{3,alpha}},options);
    for (std::size_t i=0; i<24; ++i) {
      const float a=alpha.f32()[(i/4)%3], v=x.f32()[i], s=std::sin(a*v);
      require(std::abs(snake.outputs.at(2).f32()[i]-(v+(1.F/(a+1e-9F))*(s*s)))<1e-5F,
              "direct alpha activation changed");
    }
    rejects([&] { auto bad=p; bad.tensors[2].dims={3}; ir::verify(bad); });
    std::cout << "PASS h3 audio encoder integration contracts\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
