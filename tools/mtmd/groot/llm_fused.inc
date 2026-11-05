#include "llm_fused_kernel.inc"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <filesystem>

namespace groot_fused {
using namespace nvfp4_block;
static void fp32_to_fp16(const float*x,Half*y,size_t n){
  for(size_t i=0;i<n;){size_t vl=__riscv_vsetvl_e16m2(n-i);
    __riscv_vse16_v_f16m2(y+i,__riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(x+i,vl),vl),vl);i+=vl;}
}
static void fp16_to_fp32(const Half*x,float*y,size_t n){
  for(size_t i=0;i<n;){size_t vl=__riscv_vsetvl_e16m2(n-i);
    __riscv_vse32_v_f32m4(y+i,__riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(x+i,vl),vl),vl);i+=vl;}
}
struct Mapping {
  uint8_t *data=nullptr;size_t bytes=0,offset=0;
  explicit Mapping(const std::string&path){
    int fd=open(path.c_str(),O_RDONLY);if(fd<0)throw std::runtime_error("open "+path);
    struct stat st{};if(fstat(fd,&st)){close(fd);throw std::runtime_error("stat "+path);}
    bytes=st.st_size;data=(uint8_t*)mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    if(data==MAP_FAILED)throw std::runtime_error("mmap "+path);
  }
  ~Mapping(){if(data&&data!=MAP_FAILED)munmap(data,bytes);}
  template<class T>const T*take(size_t count){
    if(offset%alignof(T)||offset>bytes||count>(bytes-offset)/sizeof(T))throw std::runtime_error("weight extent");
    auto*p=(const T*)(data+offset);offset+=count*sizeof(T);return p;
  }
};
struct Lease {
  std::vector<int> acquired;
  explicit Lease(int nt){
    for(int id=0;id<nt;++id){
      spine_tcm_block_info_t info{};
      // llama's CPU graph coordinator already leases its own TCM block.
      if(!spine_tcm_block_info(id,&info)&&info.is_acquired&&info.owner_tid==gettid())continue;
      if(!spine_tcm_mem_try_wait(id,1000000)){
        for(int i:acquired)spine_tcm_mem_release(i);
        throw std::runtime_error("LLM TCM occupied: "+std::to_string(id));
      }
      acquired.push_back(id);
    }
  }
  ~Lease(){for(int i:acquired)spine_tcm_mem_release(i);}
};
struct Decoder {
  std::vector<std::unique_ptr<Mapping>> mappings;
  std::vector<BlockParameters> layers;
  std::unique_ptr<PrefillBlock> block;
  const float*norm=nullptr;
  std::vector<Half>x,y;
  std::vector<int64_t>positions;
  std::vector<float>line,normalized;
  std::string error;
  int threads,capacity;
  Decoder(const std::string&root,int nt,int cap):layers(16),x(cap*2048),y(cap*2048),
    positions(cap*3),line(2048),normalized(2048),threads(nt),capacity(cap){
    for(int i=0;i<16;++i){
      char file[64];snprintf(file,sizeof(file),"/layer%02d.bin",i);
      auto map=std::make_unique<Mapping>(root+file);auto&p=layers[i];
      Matrix*w[]={&p.q,&p.k,&p.v,&p.o,&p.gate,&p.up,&p.down};
      const int ns[]={2048,1024,1024,2048,6144,6144,2048};
      const int ks[]={2048,2048,2048,2048,2048,2048,6144};
      for(int j=0;j<7;++j){*w[j]=Matrix{ns[j],ks[j],WeightType::PackedFullI8,
          map->take<uint8_t>(size_t(ns[j])*ks[j]),size_t(ns[j])*ks[j],256.f};
        w[j]->scales=map->take<float>(ns[j]);}
      p.input_smooth=map->take<float>(2048);p.o_smooth=map->take<float>(2048);
      p.ffn_smooth=map->take<float>(2048);p.down_smooth=map->take<float>(6144);
      p.attn_norm=map->take<float>(2048);p.q_norm=map->take<float>(128);
      p.k_norm=map->take<float>(128);p.ffn_norm=map->take<float>(2048);
      p.head_dim=128;p.epsilon=1e-6f;p.rope_base=5000000.f;
      if(map->offset!=map->bytes)throw std::runtime_error("unexpected LLM layer extent");
      mappings.push_back(std::move(map));
    }
    mappings.push_back(std::make_unique<Mapping>(root+"/norm.f32"));norm=mappings.back()->take<float>(2048);
    block=std::make_unique<PrefillBlock>(layers[0],cap,nt,true,true);block->enable_mrope(true);
  }
  void run(ggml_tensor*dst,const ggml_tensor*input,const ggml_tensor*pos){
    const int s=input->ne[1];
    if(s<1||s>capacity||input->type!=GGML_TYPE_F32||input->ne[0]!=2048||
       input->nb[1]<size_t(8192*4)||pos->type!=GGML_TYPE_I32||ggml_nelements(pos)<s*3)
      throw std::runtime_error("fused llama input contract");
    Lease lease(threads);block->set_tokens(s);
    for(int r=0;r<s;++r){
      const auto*src=(const float*)((const uint8_t*)input->data+r*input->nb[1]);
      fp32_to_fp16(src,x.data()+size_t(r)*2048,2048);
      for(int d=0;d<3;++d)positions[d*s+r]=((const int32_t*)pos->data)[d*s+r];
    }
    for(int i=0;i<16;++i){
      block->rebind(layers[i]);block->run(x.data(),y.data(),positions.data());x.swap(y);
      if(i<3)for(int r=0;r<s;++r){
        const auto*ds=(const float*)((const uint8_t*)input->data+r*input->nb[1])+(i+1)*2048;
        for(int c=0;c<2048;){size_t vl=__riscv_vsetvl_e16m2(2048-c);
          auto a=__riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(x.data()+size_t(r)*2048+c,vl),vl);
          a=__riscv_vfadd_vv_f32m4(a,__riscv_vle32_v_f32m4(ds+c,vl),vl);
          __riscv_vse16_v_f16m2(x.data()+size_t(r)*2048+c,__riscv_vfncvt_f_f_w_f16m2(a,vl),vl);c+=vl;}
      }
    }
    for(int r=0;r<s;++r){
      // GR00T consumes hidden_states[-1], before the language model's final
      // RMSNorm. Its action head applies a separate learned LayerNorm.
      fp16_to_fp32(x.data()+size_t(r)*2048,(float*)((uint8_t*)dst->data+r*dst->nb[1]),2048);
    }
  }
  static void callback(ggml_tensor*dst,const ggml_tensor*input,const ggml_tensor*pos,int ith,int nth,void*opaque){
    auto&self=*(Decoder*)opaque;
    try{if(ith||nth!=1)throw std::runtime_error("single coordinator required");self.run(dst,input,pos);}
    catch(const std::exception&e){self.error=e.what();std::fill_n((float*)dst->data,ggml_nelements(dst),NAN);}
  }
};
}
