#include "dif/frontend/h3_control.hpp"
#include "dif/ir/codec.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"

#include <charconv>
#include <iostream>
#include <string_view>

namespace {
std::uint64_t number(std::string_view text) {
  std::uint64_t value{};
  const auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
  if (parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size() || value==0)
    dif::fail("H3 control geometry requires a positive unsigned integer");
  return value;
}
}

int main(int argc,char **argv) {
  try {
    if (argc>=4 && std::string_view(argv[1])=="rows") {
      std::uint64_t ph=2,pw=2,features=196;
      for (int i=4;i<argc;++i) {
        const std::string_view key=argv[i];
        if (i+1>=argc) dif::fail("H3 control option requires a value");
        const auto value=number(argv[++i]);
        if (key=="--patch-h") ph=value;
        else if (key=="--patch-w") pw=value;
        else if (key=="--features") features=value;
        else dif::fail("unknown H3 control row option");
      }
      if (std::filesystem::exists(argv[3])) dif::fail("H3 control refuses to overwrite output");
      const auto input=dif::runtime::read_tensor(argv[2]);
      const auto rows=dif::frontend::h3_control_rows(input,ph,pw,features);
      dif::runtime::write_tensor(rows,argv[3]);
      std::cout<<"CONTROL_ROWS rows="<<rows.dims[0]<<" features="<<rows.dims[1]<<'\n';
      return 0;
    }
    if (argc==3 && std::string_view(argv[1])=="inspect") {
      dif::frontend::H3DenoiserConfig d;
      d.video_tokens=1;d.audio_tokens=1;d.text_tokens=1;d.timestep_tables=2;
      const auto graph=dif::frontend::add_h3_control(dif::frontend::make_h3_denoiser(d),d);
      const auto checkpoint=dif::weights::read_safetensors(argv[2]);
      dif::frontend::validate_h3_control_checkpoint(checkpoint,graph);
      std::cout<<"CONTROL_CHECKPOINT dense=1 blocks=5 tensors="<<graph.weights.size()
               <<" bytes="<<graph.mapped_weight_bytes
               <<" hidden="<<d.hidden<<" time_embed="<<d.time_embed_dim
               <<" patch_features=196 injection_layers=0,10,20,30,40\n";
      return 0;
    }
    std::cerr<<"usage: difh3control rows LATENT.diftensor ROWS.diftensor [--patch-h N --patch-w N --features N]\n"
             <<"       difh3control inspect CHECKPOINT.safetensors\n";
    return 2;
  } catch (const std::exception &e) {
    std::cerr<<"difh3control: "<<e.what()<<'\n';return 1;
  }
}
