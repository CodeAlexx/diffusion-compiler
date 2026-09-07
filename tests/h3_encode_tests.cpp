// Exercise the actual media/temporal/posterior helpers used by the CLI without
// substituting an image-only path or requiring a neural/GPU fixture.
#define main difh3encode_tool_main
#include "../tools/difh3encode.cpp"
#undef main
#include "dif/frontend/h3_control.hpp"

namespace {
int failures{};
void expect(bool value,const char *message) {
  if (!value) {++failures;std::cerr<<"FAIL: "<<message<<'\n';}
}
template<class F> void rejects(F f,const char *message) {
  bool failed=false;try {f();} catch (const std::exception &) {failed=true;}
  expect(failed,message);
}
}

int main(int argc,char **argv) {
  try {
    PixelSource source;
    source.frames=20;source.height=32;source.width=32;
    source.pixels=dif::runtime::zeros({0,dif::ir::DType::F32,0,{1,3,20,32,32}});
    for (std::uint64_t c=0;c<3;++c)
      for (std::uint64_t t=0;t<20;++t)
        for (std::uint64_t y=0;y<32;++y)
          for (std::uint64_t x=0;x<32;++x)
            source.pixels.f32()[((c*20+t)*32+y)*32+x]=float(t)/20;
    const auto first=extract_pixels(source,0,0,32,32,0,17);
    const auto last=extract_pixels(source,0,0,32,32,17,17);
    expect(first.dims==std::vector<std::uint64_t>{1,3,17,32,32},"real 17-frame clip shape");
    expect(first.f32()[16*32*32]==(0.8F-0.485F)/0.229F,"interior clip uses frame16 not repeated image");
    expect(last.f32()[0]==(0.85F-0.485F)/0.229F &&
           last.f32()[2*32*32]==last.f32()[16*32*32],"final clip holds only actual final source frame");
    std::vector<dif::runtime::Tensor> clips;
    for (std::uint64_t i=0;i<8;++i) {
      auto clip=dif::runtime::zeros({0,dif::ir::DType::F32,0,{1,48,5,2,2}});
      for (std::uint64_t c=0;c<48;++c)
        for (std::uint64_t t=0;t<5;++t)
          for (std::uint64_t p=0;p<4;++p) clip.f32()[(c*5+t)*4+p]=float(c*100+i*5+t);
      clips.push_back(std::move(clip));
    }
    const auto joined=concatenate_moments(clips,3);
    expect(joined.dims==std::vector<std::uint64_t>{1,48,37,2,2},"124-frame source ->8x5-3=37 moments frames");
    expect(joined.f32()[36*4]==36 && joined.f32()[37*4]==100 &&
           joined.f32()[(47*37+36)*4]==4736,"temporal concatenate preserves channels then trims moments tail");
    rejects([&]{concatenate_moments(clips,40);},"all-token drop rejected");

    // Spatial raw-neighbor blending must apply independently to every frame.
    const TilePlan y{{0,32},{64,64},{32}},x{{0,32},{64,64},{32}};
    auto left=dif::runtime::zeros({0,dif::ir::DType::F32,0,{1,48,5,4,4}});
    auto right=left;
    auto bottom_left=left, bottom_right=left;
    for (std::uint64_t c=0;c<48;++c)
      for (std::uint64_t t=0;t<5;++t)
        for (std::uint64_t p=0;p<16;++p) {
          left.f32()[(c*5+t)*16+p]=float(t+10);
          right.f32()[(c*5+t)*16+p]=float(t+20);
          bottom_left.f32()[(c*5+t)*16+p]=float(t+30);
          bottom_right.f32()[(c*5+t)*16+p]=float(t+40);
        }
    const auto stitched=stitch_moments({left,right,bottom_left,bottom_right},y,x,6,6);
    expect(stitched.dims==std::vector<std::uint64_t>{1,48,5,6,6},"spatial stitching preserves complete clip time axis");
    expect(stitched.f32()[0]==10 && stitched.f32()[3]==15 && stitched.f32()[5]==20 &&
           stitched.f32()[4*36+3]==19 && stitched.f32()[3*6+3]==30,
           "temporal frames and seam corners use raw spatial neighbors in source order");

    auto moments=dif::runtime::zeros({0,dif::ir::DType::F32,0,{1,48,7,2,2}});
    for (std::uint64_t c=0;c<48;++c)
      for (std::uint64_t t=0;t<7;++t)
        for (std::uint64_t p=0;p<4;++p)
          moments.f32()[(c*7+t)*4+p]=c<24 ? 0.12345F+float(t)*0.01F : -1.25F;
    const auto [mean,mean_rows]=sample_and_patchify(moments,42,false);
    expect(mean.dims==std::vector<std::uint64_t>{1,24,7,2,2} && mean_rows.dims==std::vector<std::uint64_t>{7,96},
           "mean posterior supports complete temporal latent and rows");
    expect(mean.f32()[0]==0.12345F,"ControlNet posterior mean is not accidentally sampled or F16-rounded");
    const auto [sample,sample_rows]=sample_and_patchify(moments,42,true);
    const auto noise=dif::torch_cpu_normal(24*7*4,42);
    for (std::size_t i=0;i<noise.size();++i) {
      const auto expected=dif::runtime::f16_to_float(dif::runtime::float_to_f16(
          moments.f32()[i]+std::exp(-0.625F)*noise[i]));
      expect(sample.f32()[i]==expected,"reference posterior sampled once in full NCTHW order after concatenation");
    }
    const auto normalized=dif::frontend::unpack_h3_video_rows(mean_rows,0,7,2,2);
    expect(normalized.f32()[0]==(mean.f32()[0]-dif::frontend::kH3VideoLatentMean[0])/dif::frontend::kH3VideoLatentStd[0],
           "normalized NCTHW handoff uses the same row normalization as the denoiser");
    if (argc==2) {
      const std::filesystem::path directory=argv[1];
      if (std::filesystem::exists(directory)) dif::fail("oracle fixture output directory already exists");
      std::filesystem::create_directories(directory);
      auto save=[&](const char *name,const dif::runtime::Tensor &tensor) {
        dif::runtime::write_tensor(tensor,directory/name);
      };
      save("pixels.diftensor",source.pixels);save("first-clip.diftensor",first);save("last-clip.diftensor",last);
      for (std::size_t i=0;i<clips.size();++i)
        dif::runtime::write_tensor(clips[i],directory/("temporal-clip-"+std::to_string(i)+".diftensor"));
      save("temporal-joined.diftensor",joined);
      save("spatial-left.diftensor",left);save("spatial-right.diftensor",right);save("spatial-stitched.diftensor",stitched);
      save("spatial-bottom-left.diftensor",bottom_left);save("spatial-bottom-right.diftensor",bottom_right);
      save("posterior-moments.diftensor",moments);save("posterior-mean.diftensor",normalized);
      save("posterior-sample.diftensor",dif::frontend::unpack_h3_video_rows(sample_rows,0,7,2,2));
      save("control-24-rows.diftensor",dif::frontend::h3_control_rows(normalized));
      auto inpaint=dif::runtime::zeros({0,dif::ir::DType::F32,0,{1,49,7,2,2}});
      for (std::size_t c=0;c<49;++c)
        for (std::size_t p=0;p<28;++p)
          inpaint.f32()[c*28+p]=c<24 ? normalized.f32()[c*28+p] :
            c==24 ? float(p%3)*0.5F : -0.5F*normalized.f32()[(c-25)*28+p];
      save("control-49-latent.diftensor",inpaint);
      save("control-49-rows.diftensor",dif::frontend::h3_control_rows(inpaint));
      std::cout<<"ORACLE_FIXTURES directory="<<directory<<'\n';
    }
  } catch (const std::exception &e) {++failures;std::cerr<<e.what()<<'\n';}
  std::cout<<"H3 video encoder temporal/media CPU tests: "<<(failures ? "FAIL" : "PASS")<<'\n';
  return failures ? 1:0;
}
