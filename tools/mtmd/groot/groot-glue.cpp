#include <riscv_vector.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
using Half=_Float16;
extern "C" {
int groot_layer_norm(const uint16_t*input,const uint16_t*weight,const uint16_t*bias,uint16_t*output,size_t rows,size_t width,float eps){
 if(!input||!weight||!bias||!output||!width)return -1;
 auto*x=(const Half*)input;auto*w=(const Half*)weight;auto*b=(const Half*)bias;auto*y=(Half*)output;
 for(size_t r=0;r<rows;++r){
  float sum=0,ss=0;
  for(size_t c=0;c<width;){size_t vl=__riscv_vsetvl_e16m2(width-c);
   auto a=__riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(x+r*width+c,vl),vl);
   sum+=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(a,__riscv_vfmv_v_f_f32m1(0,1),vl));c+=vl;}
  float mean=sum/width;
  for(size_t c=0;c<width;){size_t vl=__riscv_vsetvl_e16m2(width-c);
   auto a=__riscv_vfsub_vf_f32m4(__riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(x+r*width+c,vl),vl),mean,vl);
   ss+=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(a,a,vl),__riscv_vfmv_v_f_f32m1(0,1),vl));c+=vl;}
  float inv=1/std::sqrt(ss/width+eps);
  for(size_t c=0;c<width;){size_t vl=__riscv_vsetvl_e16m2(width-c);
   auto a=__riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(x+r*width+c,vl),vl);
   auto ww=__riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(w+c,vl),vl);
   auto bb=__riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(b+c,vl),vl);
   a=__riscv_vfmul_vf_f32m4(__riscv_vfsub_vf_f32m4(a,mean,vl),inv,vl);
   a=__riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(a,ww,vl),bb,vl);
   __riscv_vse16_v_f16m2(y+r*width+c,__riscv_vfncvt_f_f_w_f16m2(a,vl),vl);c+=vl;}
 }
 return 0;
}
void groot_add_fp16(const uint16_t*a,const uint16_t*b,uint16_t*y,size_t n){
 for(size_t i=0;i<n;){size_t vl=__riscv_vsetvl_e16m4(n-i);
 auto u=__riscv_vle16_v_f16m4((const Half*)a+i,vl),v=__riscv_vle16_v_f16m4((const Half*)b+i,vl);
 __riscv_vse16_v_f16m4((Half*)y+i,__riscv_vfadd_vv_f16m4(u,v,vl),vl);i+=vl;}
}
void groot_euler_fp16(uint16_t*actions,const uint16_t*velocity,size_t n){
 for(size_t i=0;i<n;){size_t vl=__riscv_vsetvl_e16m4(n-i);
 auto a=__riscv_vle16_v_f16m4((Half*)actions+i,vl),v=__riscv_vle16_v_f16m4((const Half*)velocity+i,vl);
 v=__riscv_vfmul_vf_f16m4(v,Half(.25),vl);
 __riscv_vse16_v_f16m4((Half*)actions+i,__riscv_vfadd_vv_f16m4(a,v,vl),vl);i+=vl;}
}
}
extern "C" void groot_patch_pixels(const uint8_t*images,const uint16_t*table,uint16_t*output){
 alignas(128) uint16_t offsets[256];
 for(int i=0;i<256;++i)offsets[i]=(i/16)*256*3+(i%16)*3;
 const size_t vl=256;auto index=__riscv_vle16_v_u16m4(offsets,vl);
 for(int view=0;view<2;++view)for(int gh=0;gh<8;++gh)for(int gw=0;gw<8;++gw)
 for(int mh=0;mh<2;++mh)for(int mw=0;mw<2;++mw){
  int patch=(gh*8+gw)*4+mh*2+mw;
  const uint8_t*src=images+view*256*256*3+((gh*2+mh)*16*256+(gw*2+mw)*16)*3;
  for(int channel=0;channel<3;++channel){
   auto bytes=__riscv_vluxei16_v_u8m2(src+channel,index,vl);
   auto ti=__riscv_vsll_vx_u16m4(__riscv_vzext_vf2_u16m4(bytes,vl),1,vl);
   auto values=__riscv_vluxei16_v_f16m4((const Half*)table,ti,vl);
   Half*dst=(Half*)output+(view*256+patch)*1536+channel*512;
   __riscv_vse16_v_f16m4(dst,values,vl);__riscv_vse16_v_f16m4(dst+256,values,vl);
  }
 }
}
