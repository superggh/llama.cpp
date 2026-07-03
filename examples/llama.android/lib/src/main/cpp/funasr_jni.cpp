#include "shared_model.h"

#include <jni.h>
#include <android/log.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define TAG "FunasrJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static inline int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ======================= fbank + LFR =======================
static const int FS=16000, WINLEN=400, SHIFT=160, NFFT=512, NMEL=80, LFR_M=7, LFR_N=6;
static const float PREEMPH=0.97f, LOWF=20.0f, HIGHF=8000.0f;
static const float LN_EPS=1e-5f;

static inline float mel(float f){ return 1127.0f*logf(1.0f+f/700.0f); }
static void fft(std::vector<float>&re,std::vector<float>&im,int n){
    for(int i=1,j=0;i<n;i++){int b=n>>1;for(;j&b;b>>=1)j^=b;j^=b;if(i<j){std::swap(re[i],re[j]);std::swap(im[i],im[j]);}}
    for(int len=2;len<=n;len<<=1){double a=-2.0*M_PI/len;float wr=cosf(a),wi=sinf(a);
        for(int i=0;i<n;i+=len){float cr=1,ci=0;for(int k=0;k<len/2;k++){
            float ur=re[i+k],ui=im[i+k];float vr=re[i+k+len/2]*cr-im[i+k+len/2]*ci,vi=re[i+k+len/2]*ci+im[i+k+len/2]*cr;
            re[i+k]=ur+vr;im[i+k]=ui+vi;re[i+k+len/2]=ur-vr;im[i+k+len/2]=ui-vi;
            float n2=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=n2;}}}
}
static std::vector<float> compute_fbank(std::vector<float> wav, int & T_out) {
    for (auto & v : wav) v *= 32768.0f;
    std::vector<float> win(WINLEN);
    for (int i=0;i<WINLEN;i++) win[i]=0.54f-0.46f*cosf(2.0f*M_PI*i/(WINLEN-1));
    const int NBIN=NFFT/2+1; float bw=(float)FS/NFFT, ml=mel(LOWF), mh=mel(HIGHF), dm=(mh-ml)/(NMEL+1);
    std::vector<std::vector<float>> fb(NMEL, std::vector<float>(NBIN,0.0f));
    for(int m=0;m<NMEL;m++){float L=ml+m*dm,C=ml+(m+1)*dm,R=ml+(m+2)*dm;
        for(int k=0;k<NBIN;k++){float mf=mel(bw*k); if(mf>L&&mf<R) fb[m][k]=mf<=C?(mf-L)/(C-L):(R-mf)/(R-C);}}
    int N=wav.size(); int T=(N-WINLEN)/SHIFT+1;
    std::vector<std::vector<float>> feat(T, std::vector<float>(NMEL));
    std::vector<float> re(NFFT),im(NFFT),fr(WINLEN);
    const float fl=1.1920929e-07f;
    for(int t=0;t<T;t++){const float*s=wav.data()+t*SHIFT;
        double mn=0;for(int i=0;i<WINLEN;i++)mn+=s[i];mn/=WINLEN;
        for(int i=0;i<WINLEN;i++)fr[i]=s[i]-(float)mn;
        for(int i=WINLEN-1;i>0;i--)fr[i]-=PREEMPH*fr[i-1];fr[0]-=PREEMPH*fr[0];
        for(int i=0;i<NFFT;i++){re[i]=i<WINLEN?fr[i]*win[i]:0.0f;im[i]=0.0f;}
        fft(re,im,NFFT);
        for(int m=0;m<NMEL;m++){float e=0;for(int k=0;k<NBIN;k++)if(fb[m][k]>0)e+=fb[m][k]*(re[k]*re[k]+im[k]*im[k]);
            feat[t][m]=logf(e>fl?e:fl);}}
    const int pad=(LFR_M-1)/2; int T_lfr=(T+LFR_N-1)/LFR_N;
    std::vector<std::vector<float>> pd; pd.reserve(T+pad+LFR_M);
    for(int i=0;i<pad;i++)pd.push_back(feat[0]);
    for(int t=0;t<T;t++)pd.push_back(feat[t]);
    while((int)pd.size()<(T_lfr-1)*LFR_N+LFR_M)pd.push_back(feat[T-1]);
    int D=LFR_M*NMEL; std::vector<float> out((size_t)T_lfr*D);
    for(int i=0;i<T_lfr;i++)for(int j=0;j<LFR_M;j++)
        memcpy(&out[(size_t)i*D+j*NMEL],pd[i*LFR_N+j].data(),NMEL*sizeof(float));
    T_out=T_lfr; return out;
}

// ======================= SAN-M encoder + adaptor =======================
struct enc_cfg { int d_model=512,n_head=4,num_blocks=50,tp_blocks=20,kernel=11,adp_llm=1024,adp_layers=2,adp_head=8; };
struct enc_model { enc_cfg c; ggml_context*ctx_w=nullptr; std::map<std::string,ggml_tensor*> t;
    ggml_tensor* g(const std::string&n){auto it=t.find(n);if(it==t.end()){LOGE("missing tensor %s",n.c_str());exit(1);}return it->second;} };

static bool load_enc(const char*p, enc_model&m){
    gguf_init_params gp={false,&m.ctx_w}; gguf_context*g=gguf_init_from_file(p,gp); if(!g)return false;
    auto rd=[&](const char*k,int d){int i=gguf_find_key(g,k);return i<0?d:(int)gguf_get_val_u32(g,i);};
    m.c.d_model=rd("funasr.enc.output_size",512); m.c.n_head=rd("funasr.enc.attention_heads",4);
    m.c.num_blocks=rd("funasr.enc.num_blocks",50); m.c.tp_blocks=rd("funasr.enc.tp_blocks",20);
    m.c.kernel=rd("funasr.enc.kernel_size",11); m.c.adp_llm=rd("funasr.adp.llm_dim",1024);
    m.c.adp_layers=rd("funasr.adp.n_layer",2); m.c.adp_head=rd("funasr.adp.attention_heads",8);
    int n=gguf_get_n_tensors(g); for(int i=0;i<n;i++){const char*nm=gguf_get_tensor_name(g,i);m.t[nm]=ggml_get_tensor(m.ctx_w,nm);}
    gguf_free(g); return true;
}
static ggml_tensor* lin(ggml_context*c,ggml_tensor*w,ggml_tensor*b,ggml_tensor*x){auto y=ggml_mul_mat(c,w,x);return b?ggml_add(c,y,b):y;}
static ggml_tensor* lnorm(ggml_context*c,ggml_tensor*x,ggml_tensor*g,ggml_tensor*b){return ggml_add(c,ggml_mul(c,ggml_norm(c,x,LN_EPS),g),b);}
static ggml_tensor* sanm_attn(ggml_context*c,enc_model&m,const std::string&p,ggml_tensor*x,int T){
    const int D=m.c.d_model,H=m.c.n_head,dk=D/H,K=m.c.kernel;
    ggml_tensor*qkv=lin(c,m.g(p+"linear_q_k_v.weight"),m.g(p+"linear_q_k_v.bias"),x); size_t nb1=qkv->nb[1];
    ggml_tensor*q=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,0));
    ggml_tensor*k=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)D*sizeof(float)));
    ggml_tensor*v=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)2*D*sizeof(float)));
    const int pad=(K-1)/2; ggml_tensor*fk=m.g(p+"fsmn_block.weight");
    ggml_tensor*vp=ggml_pad_ext(c,v,0,0,pad,pad,0,0,0,0); ggml_tensor*fsmn=v;
    for(int j=0;j<K;j++){auto sl=ggml_view_2d(c,vp,D,T,vp->nb[1],(size_t)j*vp->nb[1]);
        auto wj=ggml_view_1d(c,fk,D,(size_t)j*fk->nb[1]); fsmn=ggml_add(c,fsmn,ggml_mul(c,ggml_cont(c,sl),wj));}
    q=ggml_permute(c,ggml_reshape_3d(c,q,dk,H,T),0,2,1,3); k=ggml_permute(c,ggml_reshape_3d(c,k,dk,H,T),0,2,1,3);
    ggml_tensor*vh=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,v,dk,H,T),1,2,0,3));
    ggml_tensor*kq=ggml_soft_max(c,ggml_scale(c,ggml_mul_mat(c,k,q),1.0f/sqrtf((float)dk)));
    ggml_tensor*o=ggml_cont_2d(c,ggml_permute(c,ggml_mul_mat(c,vh,kq),0,2,1,3),D,T);
    return ggml_add(c,lin(c,m.g(p+"linear_out.weight"),m.g(p+"linear_out.bias"),o),fsmn);
}
static ggml_tensor* sanm_layer(ggml_context*c,enc_model&m,const std::string&p,ggml_tensor*x,int T,bool res){
    auto r=x; auto h=lnorm(c,x,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));
    auto sa=sanm_attn(c,m,p+"self_attn.",h,T); x=res?ggml_add(c,r,sa):sa; r=x;
    h=lnorm(c,x,m.g(p+"norm2.weight"),m.g(p+"norm2.bias"));
    h=lin(c,m.g(p+"feed_forward.w_1.weight"),m.g(p+"feed_forward.w_1.bias"),h); h=ggml_relu(c,h);
    h=lin(c,m.g(p+"feed_forward.w_2.weight"),m.g(p+"feed_forward.w_2.bias"),h); return ggml_add(c,r,h);
}
static ggml_tensor* adp_layer(ggml_context*c,enc_model&m,const std::string&p,ggml_tensor*x,int T){
    const int D=m.c.adp_llm,H=m.c.adp_head,dk=D/H; auto r=x;
    auto h=lnorm(c,x,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));
    auto q=ggml_permute(c,ggml_reshape_3d(c,lin(c,m.g(p+"self_attn.linear_q.weight"),m.g(p+"self_attn.linear_q.bias"),h),dk,H,T),0,2,1,3);
    auto k=ggml_permute(c,ggml_reshape_3d(c,lin(c,m.g(p+"self_attn.linear_k.weight"),m.g(p+"self_attn.linear_k.bias"),h),dk,H,T),0,2,1,3);
    auto vh=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,lin(c,m.g(p+"self_attn.linear_v.weight"),m.g(p+"self_attn.linear_v.bias"),h),dk,H,T),1,2,0,3));
    auto kq=ggml_soft_max(c,ggml_scale(c,ggml_mul_mat(c,k,q),1.0f/sqrtf((float)dk)));
    auto o=ggml_cont_2d(c,ggml_permute(c,ggml_mul_mat(c,vh,kq),0,2,1,3),D,T);
    x=ggml_add(c,r,lin(c,m.g(p+"self_attn.linear_out.weight"),m.g(p+"self_attn.linear_out.bias"),o)); r=x;
    h=lnorm(c,x,m.g(p+"norm2.weight"),m.g(p+"norm2.bias"));
    h=lin(c,m.g(p+"feed_forward.w_1.weight"),m.g(p+"feed_forward.w_1.bias"),h); h=ggml_relu(c,h);
    h=lin(c,m.g(p+"feed_forward.w_2.weight"),m.g(p+"feed_forward.w_2.bias"),h); return ggml_add(c,r,h);
}
static void add_posenc(std::vector<float>&x,int T,int depth){
    double inc=log(10000.0)/(depth/2.0-1.0);
    for(int t=0;t<T;t++){double pos=t+1;for(int i=0;i<depth/2;i++){double its=exp(i*-inc),st=pos*its;
        x[(size_t)t*depth+i]+=(float)sin(st);x[(size_t)t*depth+depth/2+i]+=(float)cos(st);}}
}

struct EncGraphCache {
    int T = 0;
    int F = 0;
    int D = 0;
    ggml_context *ctx = nullptr;
    ggml_cgraph *gf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor *inp = nullptr;
    ggml_tensor *out = nullptr;
    ~EncGraphCache() {
        if (galloc) ggml_gallocr_free(galloc);
        if (ctx) ggml_free(ctx);
    }
};

// forward declaration
struct FunasrCtx;
static std::vector<float> run_encoder_cached(FunasrCtx *fc, std::vector<float>& fbank, int T, int F, int &Dout);

// ======================= LLM decode =======================
static int decode_batch(llama_context*ctx,int n,llama_token*tok,float*embd,int n_embd,int&n_past,bool last_logits){
    std::vector<llama_pos> pos(n); std::vector<int32_t> nsid(n,1);
    std::vector<llama_seq_id> s0(1,0); std::vector<llama_seq_id*> sid(n); std::vector<int8_t> lg(n,0);
    for(int i=0;i<n;i++){pos[i]=n_past+i;sid[i]=s0.data();}
    if(last_logits) lg[n-1]=1;
    llama_batch b={n,tok,embd,pos.data(),nsid.data(),sid.data(),lg.data()};
    int r=llama_decode(ctx,b); n_past+=n; return r;
}

// ======================= FunASR context (opaque ptr) =======================
struct FunasrCtx {
    enc_model em;
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    llama_sampler *smpl = nullptr;
    bool owns_model = true;
    std::vector<llama_token> pre;
    std::vector<llama_token> suf;

    ggml_backend_t enc_backend = nullptr;
    typedef void (*set_n_threads_fn_t)(ggml_backend_t, int);
    set_n_threads_fn_t set_n_threads_fn = nullptr;
    std::map<int, std::shared_ptr<EncGraphCache>> enc_cache;
    static const int MAX_ENC_CACHE = 3;

    llama_memory_t mem = nullptr;
    bool prefix_cached = false;
    int prefix_len = 0;

    ~FunasrCtx() {
        enc_cache.clear();
        if (enc_backend) ggml_backend_free(enc_backend);
        if (smpl) llama_sampler_free(smpl);
        if (ctx) llama_free(ctx);
        if (model && owns_model) llama_model_free(model);
        if (em.ctx_w) ggml_free(em.ctx_w);
    }
};

static std::shared_ptr<EncGraphCache> build_encoder_graph(enc_model &m, int T, int F, ggml_backend_t backend) {
    auto cache = std::make_shared<EncGraphCache>();
    cache->T = T;
    cache->F = F;
    ggml_init_params cp = {(size_t)128*1024*1024, nullptr, true};
    cache->ctx = ggml_init(cp);
    if (!cache->ctx) {
        LOGE("failed to init encoder graph context");
        return nullptr;
    }
    ggml_context *c = cache->ctx;
    cache->inp = ggml_new_tensor_2d(c, GGML_TYPE_F32, F, T);
    ggml_set_input(cache->inp);
    ggml_tensor *x = sanm_layer(c, m, "audio_encoder.encoders0.0.", cache->inp, T, false);
    for (int i = 0; i < m.c.num_blocks - 1; i++) {
        x = sanm_layer(c, m, "audio_encoder.encoders." + std::to_string(i) + ".", x, T, true);
    }
    x = lnorm(c, x, m.g("audio_encoder.after_norm.weight"), m.g("audio_encoder.after_norm.bias"));
    for (int i = 0; i < m.c.tp_blocks; i++) {
        x = sanm_layer(c, m, "audio_encoder.tp_encoders." + std::to_string(i) + ".", x, T, true);
    }
    x = lnorm(c, x, m.g("audio_encoder.tp_norm.weight"), m.g("audio_encoder.tp_norm.bias"));
    x = lin(c, m.g("audio_adaptor.linear1.weight"), m.g("audio_adaptor.linear1.bias"), x); x = ggml_relu(c, x);
    x = lin(c, m.g("audio_adaptor.linear2.weight"), m.g("audio_adaptor.linear2.bias"), x);
    for (int i = 0; i < m.c.adp_layers; i++) {
        x = adp_layer(c, m, "audio_adaptor.blocks." + std::to_string(i) + ".", x, T);
    }
    cache->out = x;
    ggml_set_output(cache->out);
    cache->gf = ggml_new_graph_custom(c, 32768, false);
    ggml_build_forward_expand(cache->gf, cache->out);
    cache->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(cache->galloc, cache->gf)) {
        LOGE("failed to allocate encoder graph");
        return nullptr;
    }
    cache->D = (int)cache->out->ne[0];
    return cache;
}

static int get_enc_bucket(int T) {
    if (T <= 64) return 64;
    if (T <= 96) return 96;
    if (T <= 128) return 128;
    if (T <= 160) return 160;
    if (T <= 192) return 192;
    if (T <= 256) return 256;
    return T;
}

static std::vector<float> run_encoder_cached(FunasrCtx *fc, std::vector<float>& fbank, int T, int F, int &Dout) {
    Dout = 0;
    if (!fc || !fc->enc_backend) return {};

    int T_bucket = get_enc_bucket(T);
    bool padded = T_bucket > T;
    if (padded) fbank.resize((size_t)T_bucket * F, 0.0f);

    float sc = sqrtf((float)fc->em.c.d_model);
    for (size_t i = 0; i < (size_t)T * F; i++) fbank[i] *= sc;
    for (size_t i = (size_t)T * F; i < fbank.size(); i++) fbank[i] = 0.0f;
    add_posenc(fbank, T_bucket, F);

    std::shared_ptr<EncGraphCache> cache;
    auto it = fc->enc_cache.find(T_bucket);
    if (it != fc->enc_cache.end()) {
        cache = it->second;
    } else {
        LOGI("Building encoder graph for T=%d (bucket=%d)", T, T_bucket);
        cache = build_encoder_graph(fc->em, T_bucket, F, fc->enc_backend);
        if (!cache) return {};
        if ((int)fc->enc_cache.size() >= FunasrCtx::MAX_ENC_CACHE)
            fc->enc_cache.erase(fc->enc_cache.begin());
        fc->enc_cache[T_bucket] = cache;
    }

    ggml_backend_tensor_set(cache->inp, fbank.data(), 0, ggml_nbytes(cache->inp));
    if (ggml_backend_graph_compute(fc->enc_backend, cache->gf) != GGML_STATUS_SUCCESS) {
        LOGE("encoder graph compute failed"); return {};
    }
    Dout = cache->D;
    std::vector<float> out((size_t)Dout * T);
    ggml_backend_tensor_get(cache->out, out.data(), 0, (size_t)Dout * T * sizeof(float));
    return out;
}

// ======================= JNI bridge =======================
extern "C" {

JNIEXPORT jlong JNICALL
Java_com_arm_aichat_funasr_FunasrLib_00024Companion_init(
        JNIEnv *env, jobject thiz, jstring enc_path, jstring llm_path, jboolean use_shared_model) {
    const char *enc_chars = env->GetStringUTFChars(enc_path, nullptr);
    const char *llm_chars = env->GetStringUTFChars(llm_path, nullptr);

    auto *fc = new FunasrCtx();

    // load encoder
    if (!load_enc(enc_chars, fc->em)) {
        LOGW("Failed to load encoder from %s", enc_chars);
        delete fc;
        env->ReleaseStringUTFChars(enc_path, enc_chars);
        env->ReleaseStringUTFChars(llm_path, llm_chars);
        return 0;
    }
    LOGI("Encoder loaded, d_model=%d n_head=%d num_blocks=%d", fc->em.c.d_model, fc->em.c.n_head, fc->em.c.num_blocks);

    // setup CPU backend for encoder graph
    ggml_backend_load_all();
    fc->enc_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!fc->enc_backend) {
        LOGW("Failed to init CPU backend for encoder");
    } else {
        ggml_backend_reg_t cpu_reg = ggml_backend_reg_by_name("CPU");
        if (cpu_reg) {
            fc->set_n_threads_fn = (FunasrCtx::set_n_threads_fn_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads");
        }
        if (fc->set_n_threads_fn) {
            fc->set_n_threads_fn(fc->enc_backend, 4);
            LOGI("Set encoder backend threads=4");
        } else {
            LOGW("Could not set encoder backend threads");
        }
    }

    // load LLM
    if (use_shared_model) {
        fc->model = get_shared_model();
        if (fc->model) {
            fc->owns_model = false;
            LOGI("Using shared LLM model");
        } else {
            LOGW("Shared model not available, falling back to file load");
        }
    }
    if (!fc->model) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        fc->model = llama_model_load_from_file(llm_chars, mp);
        fc->owns_model = true;
    }
    if (!fc->model) {
        LOGW("Failed to load LLM from %s", llm_chars);
        delete fc;
        env->ReleaseStringUTFChars(enc_path, enc_chars);
        env->ReleaseStringUTFChars(llm_path, llm_chars);
        return 0;
    }
    LOGI("LLM loaded");

    const llama_vocab *vocab = llama_model_get_vocab(fc->model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = 256;
    cp.n_ubatch = 256;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    fc->ctx = llama_init_from_model(fc->model, cp);
    if (!fc->ctx) {
        LOGW("Failed to create llama context");
        delete fc;
        env->ReleaseStringUTFChars(enc_path, enc_chars);
        env->ReleaseStringUTFChars(llm_path, llm_chars);
        return 0;
    }
    fc->mem = llama_get_memory(fc->ctx);

    auto sp = llama_sampler_chain_default_params();
    fc->smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(fc->smpl, llama_sampler_init_greedy());

    // tokenize prefix/suffix
    const char *prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n语音转写：";
    const char *suffix = "<|im_end|>\n<|im_start|>assistant\n";
    auto tokenize = [&](const char *s) {
        int n = -llama_tokenize(vocab, s, strlen(s), nullptr, 0, false, true);
        std::vector<llama_token> v(n);
        llama_tokenize(vocab, s, strlen(s), v.data(), n, false, true);
        return v;
    };
    fc->pre = tokenize(prefix);
    fc->suf = tokenize(suffix);

    env->ReleaseStringUTFChars(enc_path, enc_chars);
    env->ReleaseStringUTFChars(llm_path, llm_chars);

    LOGI("FunASR initialized, pre=%zu suf=%zu", fc->pre.size(), fc->suf.size());
    return (jlong) fc;
}

JNIEXPORT jstring JNICALL
Java_com_arm_aichat_funasr_FunasrLib_00024Companion_transcribe(
        JNIEnv *env, jobject thiz, jlong ptr, jfloatArray audio_data) {
    auto *fc = reinterpret_cast<FunasrCtx *>(ptr);
    if (!fc || !fc->ctx) {
        LOGW("transcribe: invalid context");
        return env->NewStringUTF("");
    }

    // get audio samples from Java
    jfloat *audio_arr = env->GetFloatArrayElements(audio_data, nullptr);
    jsize audio_len = env->GetArrayLength(audio_data);
    std::vector<float> wav(audio_arr, audio_arr + audio_len);
    env->ReleaseFloatArrayElements(audio_data, audio_arr, JNI_ABORT);

    int64_t t_start = get_time_ms();

    if (wav.size() < (size_t) WINLEN) {
        LOGW("audio too short: %zu", wav.size());
        return env->NewStringUTF("");
    }

    // compute fbank
    int64_t t0 = get_time_ms();
    int T = 0;
    auto fbank = compute_fbank(wav, T);
    int64_t t_fbank = get_time_ms() - t0;
    if (T == 0) {
        LOGW("fbank returned empty");
        return env->NewStringUTF("");
    }

    // run encoder
    t0 = get_time_ms();
    int D = 0;
    auto adp = run_encoder_cached(fc, fbank, T, 560, D);
    int64_t t_encoder = get_time_ms() - t0;
    int ol = 1 + (T - 3 + 2) / 2;
    ol = 1 + (ol - 3 + 2) / 2;
    int n_aud = (ol - 1) / 2 + 1;

    // LLM decode
    t0 = get_time_ms();
    const llama_vocab *vocab = llama_model_get_vocab(fc->model);
    int n_past = 0;
    if (!fc->prefix_cached) {
        llama_memory_clear(fc->mem, true);
        decode_batch(fc->ctx, fc->pre.size(), fc->pre.data(), nullptr, 0, n_past, false);
        fc->prefix_cached = true;
        fc->prefix_len = n_past;
    } else {
        llama_memory_seq_rm(fc->mem, 0, fc->prefix_len, -1);
        n_past = fc->prefix_len;
    }
    decode_batch(fc->ctx, n_aud, nullptr, adp.data(), D, n_past, false);
    decode_batch(fc->ctx, fc->suf.size(), fc->suf.data(), nullptr, 0, n_past, true);
    int64_t t_llm_decode = get_time_ms() - t0;

    // sample tokens
    t0 = get_time_ms();
    std::string result;
    int npred = std::min(256, n_aud * 4 + 32);
    if (npred < 64) npred = 64;
    llama_token tk = llama_sampler_sample(fc->smpl, fc->ctx, -1);
    for (int i = 0; i < npred; i++) {
        if (llama_vocab_is_eog(vocab, tk)) break;
        char buf[256];
        int k = llama_token_to_piece(vocab, tk, buf, sizeof(buf), 0, true);
        if (k > 0) result.append(buf, k);
        decode_batch(fc->ctx, 1, &tk, nullptr, 0, n_past, true);
        tk = llama_sampler_sample(fc->smpl, fc->ctx, -1);
    }
    int64_t t_sample = get_time_ms() - t0;
    int64_t t_total = get_time_ms() - t_start;

    LOGI("FunASR timing: total=%lldms fbank=%lldms encoder=%lldms llm_decode=%lldms sample=%lldms (T=%d n_aud=%d npred=%d)",
         (long long)t_total, (long long)t_fbank, (long long)t_encoder, (long long)t_llm_decode, (long long)t_sample,
         T, n_aud, npred);
    LOGI("FunASR result: %s", result.c_str());
    return env->NewStringUTF(result.c_str());
}

JNIEXPORT void JNICALL
Java_com_arm_aichat_funasr_FunasrLib_00024Companion_free(
        JNIEnv *env, jobject thiz, jlong ptr) {
    auto *fc = reinterpret_cast<FunasrCtx *>(ptr);
    if (fc) {
        delete fc;
        LOGI("FunASR freed");
    }
}

JNIEXPORT jstring JNICALL
Java_com_arm_aichat_funasr_FunasrLib_00024Companion_getSystemInfo(
        JNIEnv *env, jobject thiz) {
    return env->NewStringUTF("FunASR-Nano (encoder + Qwen3-0.6B) via ggml");
}

} // extern "C"
