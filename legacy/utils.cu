//
//  lmbspecialops - a collection of tensorflow ops
//  Copyright (C) 2017  Benjamin Ummenhofer, Huizhong Zhou
//  
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#define EIGEN_USE_GPU
#include "/usr/local/cuda/include/vector_types.h"
#include "/usr/local/cuda/include/cuda.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/platform.h"

//#include "tensorflow/core/platform/default/from_stream_executor_status.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/dnn.h"
#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/host/host_platform_id.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/scratch_allocator.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/kernel.h"
#include "tensorflow/stream_executor/stream_executor.h"

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/shape_inference.h"
//#include "tensorflow/core/util//cuda_kernel_helper.h"
#include "tensorflow/core/framework/register_types.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h"
Eigen::CudaStreamDevice stream;
//#include "tensorflow/core/util/cuda_launch_config.h"

//help functions
//using GPUDevice = Eigen::GpuDevice;
using namespace tensorflow;
inline int divup(int x,int y){
  div_t tmp = std::div(x,y);
  return tmp.quot+(tmp.rem!=0?1:0);
}
struct CudaLaunchConfig {
  int virtual_thread_count = -1;
  int thread_per_block = -1;
  int block_count = -1;
};

const int d_getNumGpuMultiProcessors=24;
const int d_maxGpuThreadsPerMultiProcessor=2048;
const int d_maxGpuThreadsPerBlock=1024;

inline CudaLaunchConfig GetCudaLaunchConfig(int work_element_count){
  CHECK_GT(work_element_count, 0);
  CudaLaunchConfig config;
  const int virtual_thread_count = work_element_count;
  const int physical_thread_count = std::min(
      d_getNumGpuMultiProcessors*d_maxGpuThreadsPerMultiProcessor,
      virtual_thread_count);
  const int thread_per_block = std::min(1024,d_maxGpuThreadsPerBlock);
  const int block_count =std::min(divup(physical_thread_count,thread_per_block),d_getNumGpuMultiProcessors);

  config.virtual_thread_count = virtual_thread_count;
  config.thread_per_block = thread_per_block;
  config.block_count = block_count;
  return config;
}



#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

inline const cudaStream_t& GetCudaStream(OpKernelContext* context) {
  const cudaStream_t* ptr = CHECK_NOTNULL(
      reinterpret_cast<const cudaStream_t*>(context->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->GpuStreamMemberHack()));
  return *ptr;
}

perftools::gputools::DeviceMemory<float> AsDeviceMemory(const float* cuda_memory) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<float*>(cuda_memory));
  perftools::gputools::DeviceMemory<float> typed(wrapped);
  return typed;
}

class CublasScratchAllocator : public perftools::gputools::ScratchAllocator {
 public:
  using Stream = ::perftools::gputools::Stream;
  using DeviceMemoryBytes = ::perftools::gputools::DeviceMemory<uint8>;

  CublasScratchAllocator(OpKernelContext* context) : context_(context) {}

  int64 GetMemoryLimitInBytes(Stream* stream) override { return -1; }

  perftools::gputools::port::StatusOr<DeviceMemoryBytes> AllocateBytes(
      Stream* stream, int64 byte_size) override {
    Tensor temporary_memory;

    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory));
    if (!allocation_status.ok()) {
      return perftools::gputools::port::StatusOr<DeviceMemoryBytes>(
          DeviceMemoryBytes::MakeFromByteSize(nullptr, 0));
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    return perftools::gputools::port::StatusOr<DeviceMemoryBytes>(
        DeviceMemoryBytes::MakeFromByteSize(
            temporary_memory.flat<uint8>().data(),
            temporary_memory.flat<uint8>().size()));
  }

 private:
  OpKernelContext* context_;
  std::vector<Tensor> allocated_tensors_;
};


REGISTER_OP("WarpComputation")
    .Input("imgs_flat:float")
    .Input("index:int32")
    .Input("w:float")
    .Output("output:float");

__global__ void warp_kernel(float* out, const float* in, const int* index,const float* weights,const int npixels,const int nchannels){
  int pixel   = blockIdx.x * blockDim.x + threadIdx.x;
  int channel = blockIdx.y * blockDim.y + threadIdx.y;
  if( channel >= nchannels||pixel >= npixels)
    return;
  out[nchannels*pixel+channel]=in[nchannels*index[4*pixel]+channel]*weights[4*pixel]
                              +in[nchannels*index[4*pixel+1]+channel]*weights[4*pixel+1]
                              +in[nchannels*index[4*pixel+2]+channel]*weights[4*pixel+2]
                              +in[nchannels*index[4*pixel+3]+channel]*weights[4*pixel+3];
}

class Warp: public OpKernel 
{
public:
  explicit Warp(OpKernelConstruction* context):OpKernel(context){}

  void Compute( OpKernelContext* context ) override 
  {
    //create shared output buffer
    //int gpu_id=context->op_device_context()->gpu_id();

    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();
    const TensorShape input_shape(input_tensor.shape());

    const Tensor& index_tensor = context->input(1);
    auto  index = index_tensor.flat<int>();
    const TensorShape index_shape(index_tensor.shape());

    const Tensor& weight_tensor= context->input(2);
    auto  weight = weight_tensor.flat<float>();

    int npixels   = index_shape.dim_size(0);
    int nchannels = input_shape.dim_size(1);

    TensorShape output_shape(input_shape);
    Tensor* output_tensor=NULL;
    output_shape.set_dim(0,npixels);
    OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));
    
    auto output = output_tensor->flat<float>();
    warp2d_gpu(GetCudaStream(context),output.data(),input.data(),index.data(),weight.data(),npixels,nchannels);
  }

  void warp2d_gpu(const cudaStream_t& stream,float* out,const float* in,const int* index,const float* weight,const int npixels,const int nchannels){

    dim3 block(64,16,1);
    dim3 grid;
    
    grid.x = divup(npixels,block.x);
    grid.y = divup(nchannels,block.y);
    warp_kernel<<<grid,block,0,stream>>>(out,in,index,weight,npixels,nchannels);
  }
};
REGISTER_KERNEL_BUILDER(Name("WarpComputation").Device(DEVICE_GPU),Warp);



REGISTER_OP("HessianMatrix")
     .Input("jacobian:float")
     .Output("hessian:float");

class HessianMatrix: public OpKernel 
{
public:

  explicit HessianMatrix(OpKernelConstruction* context):OpKernel(context){}

  void Compute( OpKernelContext* context ) override {
    
    const Tensor& jacobian = context->input(0);
    const TensorShape jacobian_shape(jacobian.shape());
    
    int batch_size=jacobian_shape.dim_size(0);
    int npixels   =jacobian_shape.dim_size(1);
    int jacobian_rows=jacobian_shape.dim_size(2);
    int jacobian_cols=jacobian_shape.dim_size(3);

    TensorShape hessian_shape(jacobian.shape());
    hessian_shape.set_dim(2,jacobian_cols);
    Tensor* hessian=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(0,hessian_shape,&hessian));

    typedef perftools::gputools::DeviceMemory<float> DeviceMemoryType;

    std::vector<DeviceMemoryType> jacobian_device_memory;
    std::vector<DeviceMemoryType> hessian_device_memory;
    std::vector<DeviceMemoryType*> jacobian_ptrs;
    std::vector<DeviceMemoryType*> hessian_ptrs;

    int n_matrix=batch_size*npixels;
    jacobian_device_memory.reserve(n_matrix);
    hessian_device_memory.reserve(n_matrix);
    jacobian_ptrs.reserve(n_matrix);
    hessian_ptrs.reserve(n_matrix);

    const float* jacobian_base_ptr = jacobian.flat<float>().data();
    const float* hessian_base_ptr  = hessian->flat<float>().data();
    for (int i = 0; i < n_matrix; ++i) {
      jacobian_device_memory.push_back(AsDeviceMemory(jacobian_base_ptr+i*jacobian_rows*jacobian_cols));
      hessian_device_memory.push_back(AsDeviceMemory(hessian_base_ptr+i*jacobian_cols*jacobian_cols));
      jacobian_ptrs.push_back(&jacobian_device_memory.back());
      hessian_ptrs.push_back(&hessian_device_memory.back());
    }
    
    CublasScratchAllocator scratch_allocator(context);
    bool blas_launch_status = context->op_device_context()
                                     ->stream()
                                     ->ThenBlasGemmBatchedWithScratch(
                                        perftools::gputools::blas::Transpose::kNoTranspose,
                                        perftools::gputools::blas::Transpose::kTranspose, 
                                        jacobian_cols,jacobian_cols,jacobian_rows,static_cast<float>(1.0), 
                                        jacobian_ptrs, jacobian_cols, 
                                        jacobian_ptrs, jacobian_cols, 
                                        static_cast<float>(0.0),hessian_ptrs,jacobian_cols,
                                        n_matrix, &scratch_allocator).ok();
  }
};
REGISTER_KERNEL_BUILDER(Name("HessianMatrix").Device(DEVICE_GPU),HessianMatrix);



REGISTER_OP("EquationConstruction")
     .Input("jacobian:float")
     .Input("gradient:float")
     .Input("difference:float")
     .Output("left:float")
     .Output("right:float")
     .SetShapeFn([](shape_inference::InferenceContext* c) {

      shape_inference::ShapeHandle batch_size=c->Vector(c->Dim(c->input(0),0));
      shape_inference::DimensionHandle jacobian_cols=c->Dim(c->input(0),3);
      shape_inference::DimensionHandle one=c->Dim(c->input(2),3);
      shape_inference::ShapeHandle output1_shape;
      shape_inference::ShapeHandle output2_shape;

      c->Concatenate(batch_size,c->Matrix(jacobian_cols,jacobian_cols),&output1_shape);
      c->Concatenate(batch_size,c->Matrix(jacobian_cols,one),&output2_shape);

      c->set_output(0,output1_shape);
      c->set_output(1,output2_shape);

      return Status::OK();
    });



struct Sum {
  __host__ __device__ float operator()(const float& a, const float& b) const {
    return a + b;
  }
};

__global__ void ColumnReduceSimpleKernel(const float* in,float* out, int num_planes,
                                         int num_rows, int num_cols) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int elems_per_plane = num_rows * num_cols;

  const int plane = gid / num_cols;
  const int col = gid % num_cols;

  if (plane >= num_planes) 
    return;

  float sum = in[plane * elems_per_plane + col]+in[plane * elems_per_plane + num_cols + col];
  for (int row = 2; row < num_rows; ++row) {
    sum = sum+in[plane * elems_per_plane + row * num_cols + col];
  }
  out[plane * num_cols + col] = sum;
}


//template <typename T, typename Op, typename OUT_T, typename IN_T>
void Launch3DYReduction(const cudaStream_t& cu_stream,float* out,const float* in, int extent_x,
                        int extent_y, int extent_z) {
  int threads_per_block = 128;
  int num_blocks =(extent_x * extent_z + threads_per_block - 1) / threads_per_block;
  ColumnReduceSimpleKernel<<<num_blocks, threads_per_block, 0, cu_stream>>>(
      in, out, extent_x, extent_y, extent_z);
}

#define MAX_BUFFERS 5
#define MAX_GPUS 4

typedef perftools::gputools::DeviceMemory<float> DeviceMemoryType;
static Tensor* buffer[MAX_GPUS]={nullptr};
static std::vector<DeviceMemoryType>  memory[MAX_BUFFERS*MAX_GPUS];
static std::vector<DeviceMemoryType*> memory_ptr[MAX_BUFFERS*MAX_GPUS];


class EquationConstruction: public OpKernel 
{

private:

  int gpu_id;

public:

  explicit EquationConstruction(OpKernelConstruction* context):OpKernel(context){
    gpu_id=context->device()->tensorflow_gpu_device_info()->gpu_id;
    //std::cout<<"gpu:"<<gpu_id<<std::endl;
  }

  void Compute( OpKernelContext* context ) override {
    //std::cout<<"computing:"<<gpu_id<<std::endl;
    
    const Tensor& jacobian=context->input(0);
    const TensorShape jacobian_shape(jacobian.shape());

    const Tensor& gradient=context->input(1);
    const TensorShape gradient_shape(gradient.shape());

    const Tensor& difference=context->input(2);
    const TensorShape difference_shape(difference.shape());

    int batch_size=jacobian_shape.dim_size(0);
    int npixels   =jacobian_shape.dim_size(1);
    int n_matrix  =batch_size*npixels;

    int jacobian_rows=jacobian_shape.dim_size(2);
    int jacobian_cols=jacobian_shape.dim_size(3);

    int gradient_rows=gradient_shape.dim_size(2);
    int gradient_cols=gradient_shape.dim_size(3);

    int difference_rows=difference_shape.dim_size(2);
    int difference_cols=difference_shape.dim_size(3);


    if (buffer[gpu_id]==nullptr){

      TensorShape buffer_shape(jacobian.shape());
      buffer_shape.set_dim(2,jacobian_rows+jacobian_cols);
      PersistentTensor* newtensor= new PersistentTensor();
      context->allocate_persistent(jacobian.dtype(),buffer_shape,newtensor,&buffer[gpu_id]);


      //std::cout<<"EquationConstruction"<<buffer[gpu_id]<<std::endl;


      const float* buffer_base_ptr1 = buffer[gpu_id]->flat<float>().data();
      const float* buffer_base_ptr2 = buffer_base_ptr1+(n_matrix*jacobian_cols*jacobian_cols);

      for(int i=0;i<MAX_BUFFERS;i++){
        memory[MAX_BUFFERS*gpu_id+i].clear();
        memory[MAX_BUFFERS*gpu_id+i].reserve(n_matrix);
        memory_ptr[MAX_BUFFERS*gpu_id+i].clear();
        memory_ptr[MAX_BUFFERS*gpu_id+i].reserve(n_matrix);
      }

      for(int i=0;i<n_matrix;i++){

        memory[MAX_BUFFERS*gpu_id].push_back(AsDeviceMemory(buffer_base_ptr1+i*gradient_cols*gradient_cols));
        memory[MAX_BUFFERS*gpu_id+1].push_back(AsDeviceMemory(buffer_base_ptr2+i*jacobian_rows*jacobian_cols));
        memory[MAX_BUFFERS*gpu_id+2].push_back(AsDeviceMemory(buffer_base_ptr1+i*jacobian_cols*jacobian_cols));
        
        memory[MAX_BUFFERS*gpu_id+3].push_back(AsDeviceMemory(buffer_base_ptr2+i*gradient_cols*difference_cols));
        memory[MAX_BUFFERS*gpu_id+4].push_back(AsDeviceMemory(buffer_base_ptr1+i*jacobian_cols));

        memory_ptr[MAX_BUFFERS*gpu_id].push_back(&memory[MAX_BUFFERS*gpu_id].back());
        memory_ptr[MAX_BUFFERS*gpu_id+1].push_back(&memory[MAX_BUFFERS*gpu_id+1].back());
        memory_ptr[MAX_BUFFERS*gpu_id+2].push_back(&memory[MAX_BUFFERS*gpu_id+2].back());
        memory_ptr[MAX_BUFFERS*gpu_id+3].push_back(&memory[MAX_BUFFERS*gpu_id+3].back());
        memory_ptr[MAX_BUFFERS*gpu_id+4].push_back(&memory[MAX_BUFFERS*gpu_id+4].back());
      }
      //std::cout<<gpu_id<<" initialized"<<std::endl;
    }

    const float* jacobian_base_ptr = jacobian.flat<float>().data();
    const float* gradient_base_ptr = gradient.flat<float>().data();
    const float* difference_base_ptr=difference.flat<float>().data();

    std::vector<DeviceMemoryType>  jacobian_device_memory;
    std::vector<DeviceMemoryType>  gradient_device_memory;
    std::vector<DeviceMemoryType>  difference_device_memory;

    std::vector<DeviceMemoryType*> jacobian_ptrs;
    std::vector<DeviceMemoryType*> gradient_ptrs;
    std::vector<DeviceMemoryType*> difference_ptrs;

    jacobian_device_memory.reserve(n_matrix);
    gradient_device_memory.reserve(n_matrix);
    difference_device_memory.reserve(n_matrix);

    jacobian_ptrs.reserve(n_matrix);
    gradient_ptrs.reserve(n_matrix);
    difference_ptrs.reserve(n_matrix);

    for (int i = 0; i < n_matrix; ++i) {

      jacobian_device_memory.push_back(AsDeviceMemory(jacobian_base_ptr+i*jacobian_rows*jacobian_cols));
      gradient_device_memory.push_back(AsDeviceMemory(gradient_base_ptr+i*gradient_rows*gradient_cols));
      difference_device_memory.push_back(AsDeviceMemory(difference_base_ptr+i*difference_rows*difference_cols));

      jacobian_ptrs.push_back(&jacobian_device_memory.back());
      gradient_ptrs.push_back(&gradient_device_memory.back());
      difference_ptrs.push_back(&difference_device_memory.back());
    }

    //std::cout<<"computing0 done:"<<gpu_id<<std::endl; 
    CublasScratchAllocator scratch_allocator(context);
    bool blas_launch_status = context->op_device_context()
                                     ->stream()
                                     ->ThenBlasGemmBatchedWithScratch(
                                        perftools::gputools::blas::Transpose::kNoTranspose,
                                        perftools::gputools::blas::Transpose::kTranspose, 
                                        gradient_cols, gradient_cols,gradient_rows,static_cast<float>(1.0), 
                                        gradient_ptrs, gradient_cols, 
                                        gradient_ptrs, gradient_cols, 
                                        static_cast<float>(0.0),memory_ptr[MAX_BUFFERS*gpu_id],gradient_cols,
                                        n_matrix, &scratch_allocator).ok();

    //std::cout<<"computing1 done:"<<gpu_id<<std::endl;                           

    blas_launch_status = context->op_device_context()
                                 ->stream()
                                 ->ThenBlasGemmBatchedWithScratch(
                                    perftools::gputools::blas::Transpose::kNoTranspose,
                                    perftools::gputools::blas::Transpose::kNoTranspose, 
                                    jacobian_cols,gradient_cols,jacobian_rows,static_cast<float>(1.0), 
                                    jacobian_ptrs, jacobian_cols, 
                                    memory_ptr[MAX_BUFFERS*gpu_id],gradient_cols, 
                                    static_cast<float>(0.0),memory_ptr[MAX_BUFFERS*gpu_id+1],jacobian_cols,
                                    n_matrix, &scratch_allocator).ok();
    //std::cout<<"computing2 done:"<<gpu_id<<std::endl;                             

    blas_launch_status = context->op_device_context()
                                 ->stream()
                                 ->ThenBlasGemmBatchedWithScratch(
                                    perftools::gputools::blas::Transpose::kNoTranspose,
                                    perftools::gputools::blas::Transpose::kTranspose, 
                                    jacobian_cols,jacobian_cols,gradient_cols,static_cast<float>(1.0), 
                                    memory_ptr[MAX_BUFFERS*gpu_id+1], jacobian_cols, 
                                    jacobian_ptrs, jacobian_cols, 
                                    static_cast<float>(0.0),memory_ptr[MAX_BUFFERS*gpu_id+2],jacobian_cols,
                                    n_matrix, &scratch_allocator).ok();
    //std::cout<<"computing3 done:"<<gpu_id<<std::endl;
    
    TensorShape output_shape;
    output_shape.AddDim(batch_size);
    output_shape.AddDim(jacobian_cols);
    output_shape.AddDim(jacobian_cols);

    Tensor* output_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));
    
    auto output = output_tensor->flat<float>();
    auto input  = buffer[gpu_id]->flat<float>();
    int nelements=jacobian_cols*jacobian_cols;

    Launch3DYReduction(GetCudaStream(context),output.data(),input.data(),batch_size,npixels,nelements);

    blas_launch_status = context->op_device_context()
                           ->stream()
                           ->ThenBlasGemmBatchedWithScratch(
                              perftools::gputools::blas::Transpose::kNoTranspose,
                              perftools::gputools::blas::Transpose::kTranspose, 
                              difference_cols,gradient_cols,gradient_rows,static_cast<float>(1.0), 
                              difference_ptrs,difference_cols, 
                              gradient_ptrs,gradient_cols, 
                              static_cast<float>(0.0),memory_ptr[MAX_BUFFERS*gpu_id+3],difference_cols,
                              n_matrix, &scratch_allocator).ok();

    blas_launch_status = context->op_device_context()
                       ->stream()
                       ->ThenBlasGemmBatchedWithScratch(
                          perftools::gputools::blas::Transpose::kNoTranspose,
                          perftools::gputools::blas::Transpose::kTranspose, 
                          difference_cols,jacobian_cols,jacobian_rows,static_cast<float>(1.0), 
                          memory_ptr[MAX_BUFFERS*gpu_id+3],difference_cols, 
                          jacobian_ptrs, jacobian_cols, 
                          static_cast<float>(0.0),memory_ptr[MAX_BUFFERS*gpu_id+4],difference_cols,
                          n_matrix, &scratch_allocator).ok();

    TensorShape output2_shape;
    output2_shape.AddDim(batch_size);
    output2_shape.AddDim(jacobian_cols);
    output2_shape.AddDim(difference_cols);

    Tensor* output2_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(1,output2_shape,&output2_tensor));

    auto output2 = output2_tensor->flat<float>();
    nelements=jacobian_cols*difference_cols;
    Launch3DYReduction(GetCudaStream(context),output2.data(),input.data(),batch_size,npixels,nelements);
  }
};
REGISTER_KERNEL_BUILDER(Name("EquationConstruction").Device(DEVICE_GPU),EquationConstruction);


REGISTER_OP("EquationConstructionGrad")
     .Input("jacobian:float")
     .Input("gradient:float")
     .Input("difference:float")
     .Input("left_grad:float")
     .Input("right_grad:float")
     .Output("jacobian_grad:float")
     .Output("gradient_grad:float")
     .Output("difference_grad:float");
     // .Output("test_output:float");

#define GRAD_MAX_BUFFERS 4 
enum{
  GRAD_0,
  GRAD_1,
  JACOBIAN_GRAD,
  JACOBIAN_GRAD_GRAD
};
static Tensor* grad_buffer[MAX_GPUS]={nullptr};
static std::vector<DeviceMemoryType>  grad_memory[GRAD_MAX_BUFFERS*MAX_GPUS];
static std::vector<DeviceMemoryType*> grad_memory_ptr[GRAD_MAX_BUFFERS*MAX_GPUS];

__global__ void tile_kernel(const float* in,float* out, 
                           int num_planes, int num_rows, int num_cols) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int elems_per_plane = num_rows * num_cols;

  const int plane = gid / num_rows;
  const int row   = gid % num_rows;

  if (plane >= num_planes) 
    return;

  for (int col=0;col<num_cols; ++col){
    out[plane * elems_per_plane + row * num_cols + col]=in[plane*num_cols+col];
  }
}

void tile_gpu(const cudaStream_t& stream,float* out,const float* in,const int batch_size,const int npixels,const int nelements){
  int threads_per_block = 128;
  int num_blocks =(batch_size * npixels+ threads_per_block - 1) / threads_per_block;
  tile_kernel<<<num_blocks,threads_per_block,0,stream>>>(in,out,batch_size,npixels,nelements);
}

class EquationConstructionGrad: public OpKernel 
{
private:

  int gpu_id;

public:

  explicit EquationConstructionGrad(OpKernelConstruction* context):OpKernel(context){
    gpu_id=context->device()->tensorflow_gpu_device_info()->gpu_id;
  }
  void Compute( OpKernelContext* context ) override {

    const Tensor& jacobian=context->input(0);
    const TensorShape jacobian_shape(jacobian.shape());

    const Tensor& gradient=context->input(1);
    const TensorShape gradient_shape(gradient.shape());

    const Tensor& difference=context->input(2);
    const TensorShape difference_shape(difference.shape());

    const Tensor& grad0=context->input(3);
    const Tensor& grad1=context->input(4);

    Tensor* jacobian_grad_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(0,jacobian_shape,&jacobian_grad_tensor));

    Tensor* gradient_grad_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(1,gradient_shape,&gradient_grad_tensor));

    Tensor* difference_grad_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(2,difference_shape,&difference_grad_tensor));


    int batch_size=jacobian_shape.dim_size(0);
    int npixels   =jacobian_shape.dim_size(1);
    int n_matrix  =batch_size*npixels;

    int jacobian_rows=jacobian_shape.dim_size(2);
    int jacobian_cols=jacobian_shape.dim_size(3);

    int gradient_rows=gradient_shape.dim_size(2);
    int gradient_cols=gradient_shape.dim_size(3);

    int difference_rows=difference_shape.dim_size(2);
    int difference_cols=difference_shape.dim_size(3);


    // std::cout<<buffer[gpu_id]<<std::endl;
    float* tiled_grad0_ptr=buffer[gpu_id]->flat<float>().data();
    float* tiled_grad1_ptr=tiled_grad0_ptr+(n_matrix*jacobian_cols*jacobian_cols);
    

    if (grad_buffer[gpu_id]==nullptr){

      TensorShape buffer_shape(jacobian.shape());
      buffer_shape.set_dim(2,2*gradient_rows);
      PersistentTensor* newtensor= new PersistentTensor();
      context->allocate_persistent(jacobian.dtype(),buffer_shape,newtensor,&grad_buffer[gpu_id]);
        
      float* buffer_base_ptr1 = grad_buffer[gpu_id]->flat<float>().data();
      float* buffer_base_ptr2 = buffer_base_ptr1+(n_matrix*gradient_rows*jacobian_cols);

      for(int i=0;i<GRAD_MAX_BUFFERS;i++){
        grad_memory[GRAD_MAX_BUFFERS*gpu_id+i].clear();
        grad_memory[GRAD_MAX_BUFFERS*gpu_id+i].reserve(n_matrix);
        grad_memory_ptr[GRAD_MAX_BUFFERS*gpu_id+i].clear();
        grad_memory_ptr[GRAD_MAX_BUFFERS*gpu_id+i].reserve(n_matrix);
      }

      for(int i=0;i<n_matrix;i++){

        grad_memory[GRAD_MAX_BUFFERS*gpu_id+JACOBIAN_GRAD].push_back(AsDeviceMemory(buffer_base_ptr2+i*gradient_rows*jacobian_cols));
        grad_memory[GRAD_MAX_BUFFERS*gpu_id+JACOBIAN_GRAD_GRAD].push_back(AsDeviceMemory(buffer_base_ptr1+i*gradient_rows*jacobian_cols));

        grad_memory[GRAD_MAX_BUFFERS*gpu_id+GRAD_0].push_back(AsDeviceMemory(tiled_grad0_ptr+i*jacobian_cols*jacobian_cols));
        grad_memory[GRAD_MAX_BUFFERS*gpu_id+GRAD_1].push_back(AsDeviceMemory(tiled_grad1_ptr+i*difference_cols*jacobian_cols));

        grad_memory_ptr[GRAD_MAX_BUFFERS*gpu_id+JACOBIAN_GRAD].push_back(&grad_memory[GRAD_MAX_BUFFERS*gpu_id+JACOBIAN_GRAD].back());
        grad_memory_ptr[GRAD_MAX_BUFFERS*gpu_id+JACOBIAN_GRAD_GRAD].push_back(&grad_memory[GRAD_MAX_BUFFERS*gpu_id+JACOBIAN_GRAD_GRAD].back());

        grad_memory_ptr[GRAD_MAX_BUFFERS*gpu_id+GRAD_0].push_back(&grad_memory[GRAD_MAX_BUFFERS*gpu_id+GRAD_0].back());
        grad_memory_ptr[GRAD_MAX_BUFFERS*gpu_id+GRAD_1].push_back(&grad_memory[GRAD_MAX_BUFFERS*gpu_id+GRAD_1].back());

      }
    }

    const float* jacobian_base_ptr = jacobian.flat<float>().data();
    const float* gradient_base_ptr = gradient.flat<float>().data();
    const float* difference_base_ptr=difference.flat<float>().data();

    const float* jacobian_grad_base_ptr=jacobian_grad_tensor->flat<float>().data();
    const float* gradient_grad_base_ptr=gradient_grad_tensor->flat<float>().data();
    const float* difference_grad_base_ptr=difference_grad_tensor->flat<float>().data();

    std::vector<DeviceMemoryType>  jacobian_device_memory;
    std::vector<DeviceMemoryType>  gradient_device_memory;
    std::vector<DeviceMemoryType>  difference_device_memory;

    std::vector<DeviceMemoryType>  jacobian_grad_device_memory;
    std::vector<DeviceMemoryType>  gradient_grad_device_memory;
    std::vector<DeviceMemoryType>  difference_grad_device_memory;

    std::vector<DeviceMemoryType*> jacobian_ptrs;
    std::vector<DeviceMemoryType*> gradient_ptrs;
    std::vector<DeviceMemoryType*> difference_ptrs;

    std::vector<DeviceMemoryType*> jacobian_grad_ptrs;
    std::vector<DeviceMemoryType*> gradient_grad_ptrs;
    std::vector<DeviceMemoryType*> difference_grad_ptrs;

    jacobian_device_memory.reserve(n_matrix);
    gradient_device_memory.reserve(n_matrix);
    difference_device_memory.reserve(n_matrix);

    jacobian_grad_device_memory.reserve(n_matrix);
    gradient_grad_device_memory.reserve(n_matrix);
    difference_grad_device_memory.reserve(n_matrix);

    jacobian_ptrs.reserve(n_matrix);
    gradient_ptrs.reserve(n_matrix);
    difference_ptrs.reserve(n_matrix);

    jacobian_grad_ptrs.reserve(n_matrix);
    gradient_grad_ptrs.reserve(n_matrix);
    difference_grad_ptrs.reserve(n_matrix);

    for (int i = 0; i < n_matrix; ++i) {

      jacobian_device_memory.push_back(AsDeviceMemory(jacobian_base_ptr+i*jacobian_rows*jacobian_cols));
      gradient_device_memory.push_back(AsDeviceMemory(gradient_base_ptr+i*gradient_rows*gradient_cols));
      difference_device_memory.push_back(AsDeviceMemory(difference_base_ptr+i*difference_rows*difference_cols));

      jacobian_ptrs.push_back(&jacobian_device_memory.back());
      gradient_ptrs.push_back(&gradient_device_memory.back());
      difference_ptrs.push_back(&difference_device_memory.back());


      jacobian_grad_device_memory.push_back(AsDeviceMemory(jacobian_grad_base_ptr+i*jacobian_rows*jacobian_cols));
      gradient_grad_device_memory.push_back(AsDeviceMemory(gradient_grad_base_ptr+i*gradient_rows*gradient_cols));
      difference_grad_device_memory.push_back(AsDeviceMemory(difference_grad_base_ptr+i*difference_rows*difference_cols));

      jacobian_grad_ptrs.push_back(&jacobian_grad_device_memory.back());
      gradient_grad_ptrs.push_back(&gradient_grad_device_memory.back());
      difference_grad_ptrs.push_back(&difference_grad_device_memory.back());
    }
    
    int grad0_nelements=jacobian_cols*jacobian_cols;
    tile_gpu(GetCudaStream(context),tiled_grad0_ptr,grad0.flat<float>().data(),batch_size,npixels,grad0_nelements);

    int grad1_nelements=jacobian_cols*difference_cols;
    tile_gpu(GetCudaStream(context),tiled_grad1_ptr,grad1.flat<float>().data(),batch_size,npixels,grad1_nelements);


    



    CublasScratchAllocator scratch_allocator(context);
    bool blas_launch_status = context->op_device_context()
                                                ->stream()
                                                ->ThenBlasGemmBatchedWithScratch(
                                                   perftools::gputools::blas::Transpose::kNoTranspose,
                                                   perftools::gputools::blas::Transpose::kNoTranspose, 
                                                   jacobian_cols,gradient_rows,jacobian_rows,static_cast<float>(1.0), 
                                                   jacobian_ptrs, jacobian_cols, 
                                                   gradient_ptrs, gradient_cols, 
                                                   static_cast<float>(0.0),grad_memory_ptr[GRAD_MAX_BUFFERS*gpu_id+JACOBIAN_GRAD],jacobian_cols,
                                                   n_matrix, &scratch_allocator).ok();

    blas_launch_status = context->op_device_context()
                                           ->stream()
                                           ->ThenBlasGemmBatchedWithScratch(
                                              perftools::gputools::blas::Transpose::kNoTranspose,
                                              perftools::gputools::blas::Transpose::kNoTranspose, 
                                              difference_cols,gradient_rows,jacobian_cols,static_cast<float>(1.0), 
                                              grad_memory_ptr[GRAD_MAX_BUFFERS*gpu_id+GRAD_1],difference_cols, 
                                              grad_memory_ptr[GRAD_MAX_BUFFERS*gpu_id+JACOBIAN_GRAD],jacobian_cols, 
                                              static_cast<float>(0.0),difference_grad_ptrs,difference_cols,
                                              n_matrix, &scratch_allocator).ok();

    
    blas_launch_status = context->op_device_context()
                                           ->stream()
                                           ->ThenBlasGemmBatchedWithScratch(
                                              perftools::gputools::blas::Transpose::kNoTranspose,
                                              perftools::gputools::blas::Transpose::kNoTranspose, 
                                              jacobian_cols,gradient_rows,jacobian_cols,static_cast<float>(2.0), 
                                              grad_memory_ptr[GRAD_MAX_BUFFERS*gpu_id+GRAD_0],jacobian_cols, 
                                              grad_memory_ptr[GRAD_MAX_BUFFERS*gpu_id+JACOBIAN_GRAD],jacobian_cols, 
                                              static_cast<float>(0.0),grad_memory_ptr[GRAD_MAX_BUFFERS*gpu_id+JACOBIAN_GRAD_GRAD],jacobian_cols,
                                              n_matrix, &scratch_allocator).ok();

    blas_launch_status = context->op_device_context()
                                           ->stream()
                                           ->ThenBlasGemmBatchedWithScratch(
                                              perftools::gputools::blas::Transpose::kTranspose,
                                              perftools::gputools::blas::Transpose::kNoTranspose, 
                                              jacobian_cols,gradient_rows,difference_cols,static_cast<float>(1.0), 
                                              grad_memory_ptr[GRAD_MAX_BUFFERS*gpu_id+GRAD_1],difference_cols, 
                                              difference_ptrs,difference_cols, 
                                              static_cast<float>(1.0),grad_memory_ptr[GRAD_MAX_BUFFERS*gpu_id+JACOBIAN_GRAD_GRAD],jacobian_cols,
                                              n_matrix, &scratch_allocator).ok();

    blas_launch_status = context->op_device_context()
                                       ->stream()
                                       ->ThenBlasGemmBatchedWithScratch(
                                          perftools::gputools::blas::Transpose::kNoTranspose,
                                          perftools::gputools::blas::Transpose::kTranspose, 
                                          jacobian_cols,gradient_cols,gradient_rows,static_cast<float>(1.0), 
                                          grad_memory_ptr[GRAD_MAX_BUFFERS*gpu_id+JACOBIAN_GRAD_GRAD],jacobian_cols, 
                                          gradient_ptrs,gradient_cols, 
                                          static_cast<float>(0.0),jacobian_grad_ptrs,jacobian_cols,
                                          n_matrix, &scratch_allocator).ok(); 

    blas_launch_status = context->op_device_context()
                                       ->stream()
                                       ->ThenBlasGemmBatchedWithScratch(
                                          perftools::gputools::blas::Transpose::kTranspose,
                                          perftools::gputools::blas::Transpose::kNoTranspose, 
                                          gradient_cols,gradient_rows,jacobian_cols,static_cast<float>(1.0), 
                                          jacobian_ptrs,jacobian_cols, 
                                          grad_memory_ptr[GRAD_MAX_BUFFERS*gpu_id+JACOBIAN_GRAD_GRAD],jacobian_cols, 
                                          static_cast<float>(0.0),gradient_grad_ptrs,gradient_cols,
                                          n_matrix, &scratch_allocator).ok();                                                                                                            

  }
};
REGISTER_KERNEL_BUILDER(Name("EquationConstructionGrad").Device(DEVICE_GPU),EquationConstructionGrad);



REGISTER_OP("TestSum")
     .Input("input:float")
     .Output("output:float");

class TestSum: public OpKernel 
{
public:

  explicit TestSum(OpKernelConstruction* context):OpKernel(context){}

  void Compute( OpKernelContext* context ) override {
    
    const Tensor& input_tensor=context->input(0);
    const TensorShape input_shape(input_tensor.shape());
    int batch_size=input_shape.dim_size(0);
    int npixels=input_shape.dim_size(1);
    int rows=input_shape.dim_size(2);
    int cols=input_shape.dim_size(3);


    TensorShape output_shape;
    output_shape.AddDim(batch_size);
    output_shape.AddDim(rows);
    output_shape.AddDim(cols);

    Tensor* output_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));
    
    auto output = output_tensor->flat<float>();
    auto input  = input_tensor.flat<float>();
    int nelements=rows*cols;

    Launch3DYReduction(GetCudaStream(context),output.data(),input.data(),batch_size,npixels,nelements);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("TestSum").Device(DEVICE_GPU),TestSum);




REGISTER_OP("TestTile")
           .Input("input:float")
           .Attr("npixels:int")
           .Output("output:float");

class TestTile: public OpKernel 
{
private:
  int npixels;
public:

  explicit TestTile(OpKernelConstruction* context):OpKernel(context){
    context->GetAttr("npixels",&npixels);
  }

  void Compute( OpKernelContext* context ) override {
    
    const Tensor& input_tensor=context->input(0);
    const TensorShape input_shape(input_tensor.shape());
    int batch_size=input_shape.dim_size(0);    
    int rows=input_shape.dim_size(1);
    int cols=input_shape.dim_size(2);


    TensorShape output_shape;
    output_shape.AddDim(batch_size);
    output_shape.AddDim(npixels);
    output_shape.AddDim(rows);
    output_shape.AddDim(cols);

    Tensor* output_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));
    
    auto output = output_tensor->flat<float>();
    auto input  = input_tensor.flat<float>();
    int nelements=rows*cols;
    tile_gpu(GetCudaStream(context),output.data(),input.data(),batch_size,npixels,nelements);
  }

};
REGISTER_KERNEL_BUILDER(Name("TestTile").Device(DEVICE_GPU),TestTile);


REGISTER_OP("JacobianConstruction")
     .Input("points:float")
     .Input("intrinsics:float")
     .Input("rotation:float")
     .Input("translation:float")
     .Output("jacobian:float")
     .Output("projection:float")
     .SetShapeFn([](shape_inference::InferenceContext* c) {

      std::cout<<"set_shape"<<std::endl;
      shape_inference::ShapeHandle     point_shape=c->input(0);
      std::vector<shape_inference::DimensionHandle> dims1(4);
      dims1[0]=c->Dim(c->input(0),0);
      dims1[1]=c->Dim(c->input(0),1);
      dims1[2]=c->MakeDim(2);
      dims1[3]=c->MakeDim(6);
      shape_inference::ShapeHandle output1_shape=c->MakeShape(dims1);
      c->set_output(0,output1_shape);


      std::vector<shape_inference::DimensionHandle> dims2(3);
      dims2[0]=c->Dim(c->input(0),0);
      dims2[1]=c->Dim(c->input(0),1);
      dims2[2]=c->MakeDim(2);
      shape_inference::ShapeHandle output2_shape=c->MakeShape(dims2);
      c->set_output(1,output2_shape);
      std::cout<<"set_shape_done"<<std::endl;
      return Status::OK();
    });

__global__ void jacobian_construction_kernel(const int nsamples,const int npoints,const float* points,
                                             const float* intrinsics,const float* rotation,const float* translation,
                                             float* jacobian,float* projection) {

  const int gid=threadIdx.x+blockIdx.x*blockDim.x;
  int sample_offset=gid/npoints;
  int point_offset =gid%npoints;

  if (sample_offset>= nsamples){
    return;
  }

  Eigen::Matrix3f rotation_matrix(rotation+9*sample_offset);
  Eigen::Vector3f transation_vector(translation+3*sample_offset);
  Eigen::Vector3f point_vector(points+3*npoints*sample_offset+3*point_offset);
  Eigen::Vector3f point_vector_transformed=rotation_matrix*point_vector+transation_vector;

  float x=(point_vector_transformed(0)/point_vector_transformed(2)),
        y=(point_vector_transformed(1)/point_vector_transformed(2));

  int offset=2*gid;
  projection[offset]  =intrinsics[4*sample_offset  ]*x+intrinsics[4*sample_offset+2],
  projection[offset+1]=intrinsics[4*sample_offset+1]*y+intrinsics[4*sample_offset+3];

  offset=12*gid;
  jacobian[offset]  = intrinsics[4*sample_offset]*x*y;
  jacobian[offset+1]=-intrinsics[4*sample_offset]*(1.0+x*x); 
  jacobian[offset+2]= intrinsics[4*sample_offset]*y; 
  jacobian[offset+3]=-intrinsics[4*sample_offset]/point_vector_transformed(2); 
  jacobian[offset+4]= 0; 
  jacobian[offset+5]= intrinsics[4*sample_offset]*x/point_vector_transformed(2); 

  jacobian[offset+6] = intrinsics[4*sample_offset+1]*(1.0+y*y);  
  jacobian[offset+7] =-intrinsics[4*sample_offset+1]*x*y;
  jacobian[offset+8] =-intrinsics[4*sample_offset+1]*x;  
  jacobian[offset+9] = 0; 
  jacobian[offset+10]=-intrinsics[4*sample_offset+1]/point_vector_transformed(2);
  jacobian[offset+11]= intrinsics[4*sample_offset+1]*y/point_vector_transformed(2); 
}

typedef Eigen::GpuDevice GPUDevice;

class JacobianConstruction:public OpKernel 
{

public:

  explicit JacobianConstruction(OpKernelConstruction* context):OpKernel(context){

  }

  void Compute(OpKernelContext* context ) override {

    const Tensor&     points=context->input(0);
    const TensorShape points_shape(points.shape());
    const Tensor& intrinsic  =context->input(1);
    const Tensor& rotation   =context->input(2);
    const Tensor& translation=context->input(3);
    
    int nbatch =points_shape.dim_size(0);
    int npoints=points_shape.dim_size(1);

    CudaLaunchConfig config=GetCudaLaunchConfig(nbatch*npoints);

    TensorShape jacobian_shape(points_shape);
    jacobian_shape.set_dim(2,2);
    jacobian_shape.AddDim(6);

    Tensor* jacobian_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(0,jacobian_shape,&jacobian_tensor));

    TensorShape projection_shape(points_shape);
    projection_shape.set_dim(2,2);
    Tensor* projection_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(1,projection_shape,&projection_tensor));
    
    auto projection= projection_tensor->flat<float>();
    auto jacobian  = jacobian_tensor->flat<float>();
    jacobian_construction_kernel<<<config.block_count,config.thread_per_block,0,GetCudaStream(context)>>>(nbatch,npoints,points.flat<float>().data(),
                                                                                intrinsic.flat<float>().data(),rotation.flat<float>().data(),translation.flat<float>().data(),
                                                                                jacobian.data(),projection.data());
  }
};
REGISTER_KERNEL_BUILDER(Name("JacobianConstruction").Device(DEVICE_GPU),JacobianConstruction);



REGISTER_OP("EquationConstructionPrepare")
     .Input("points:float")
     .Input("intrinsics:float")
     .Input("rotation:float")
     .Input("translation:float")
     .Input("conv1:float")
     .Input("conv2:float")
     .Input("gradx:float")
     .Input("grady:float")
     .Output("jacobian:float")
     .Output("grad:float")
     .Output("diff:float")
     .Output("valid:float")
     .SetShapeFn([](shape_inference::InferenceContext* c) {
      //std::cout<<"set_shape_2"<<std::endl;
      shape_inference::ShapeHandle     point_shape=c->input(0);
      std::vector<shape_inference::DimensionHandle> dims1(4);
      dims1[0]=c->Dim(point_shape,0);
      dims1[1]=c->Dim(point_shape,1);
      dims1[2]=c->MakeDim(2);
      dims1[3]=c->MakeDim(6);
      shape_inference::ShapeHandle output1_shape=c->MakeShape(dims1);
      c->set_output(0,output1_shape);

      shape_inference::ShapeHandle     conv_shape=c->input(4);
      dims1[2]=c->Dim(conv_shape,2);
      dims1[3]=c->MakeDim(2);
      shape_inference::ShapeHandle output2_shape=c->MakeShape(dims1);
      c->set_output(1,output2_shape);

      dims1[3]=c->MakeDim(1);
      shape_inference::ShapeHandle output3_shape=c->MakeShape(dims1);
      c->set_output(2,output3_shape);

      dims1.resize(3);
      dims1[2]=c->MakeDim(1);
      shape_inference::ShapeHandle output4_shape=c->MakeShape(dims1);
      c->set_output(3,output4_shape);

      return Status::OK();
    });


__global__ void jacobian_construction_kernel2(const int height,const int width,
                                              const int nsamples,const int npoints,const float* points,
                                              const float* intrinsics,const float* rotation,const float* translation,
                                              float* jacobian,int* projection,float* weights,float* in_boundary) {

  const int gid=threadIdx.x+blockIdx.x*blockDim.x;
  int sample_offset=gid/npoints;
  int point_offset =gid%npoints;

  if (sample_offset>= nsamples){
    return;
  }

  Eigen::Matrix3f rotation_matrix(rotation+9*sample_offset);
  Eigen::Vector3f transation_vector(translation+3*sample_offset);
  Eigen::Vector3f point_vector(points+3*npoints*sample_offset+3*point_offset);
  Eigen::Vector3f point_vector_transformed=rotation_matrix*point_vector+transation_vector;

  float x=(point_vector_transformed(0)/point_vector_transformed(2)),
        y=(point_vector_transformed(1)/point_vector_transformed(2));

  int offset=2*gid;
  float px=intrinsics[4*sample_offset  ]*x+intrinsics[4*sample_offset+2],
        py=intrinsics[4*sample_offset+1]*y+intrinsics[4*sample_offset+3];

  if (px<0||px>width-2||py<0||py>height-2){
    projection[offset]  =-1;
    projection[offset+1]=-1;
    in_boundary[gid]=0.0;
    return;
  }else{
    in_boundary[gid]=1.0;
    projection[offset]  =static_cast<int>px;
    projection[offset+1]=static_cast<int>py;
    float ax=px-projection[offset],
          ay=py-projection[offset+1];
    offset=4*gid;
    weights[offset]=(1.0-ax)*(1.0-ay);
    weights[offset+1]=(ax)*(1.0-ay);
    weights[offset+2]=(1.0-ax)*ay;
    weights[offset+3]=ax*ay;
  }

  offset=12*gid;
  jacobian[offset]  = intrinsics[4*sample_offset]*x*y;
  jacobian[offset+1]=-intrinsics[4*sample_offset]*(1.0+x*x); 
  jacobian[offset+2]= intrinsics[4*sample_offset]*y; 
  jacobian[offset+3]=-intrinsics[4*sample_offset]/point_vector_transformed(2); 
  jacobian[offset+4]= 0; 
  jacobian[offset+5]= intrinsics[4*sample_offset]*x/point_vector_transformed(2); 

  jacobian[offset+6] = intrinsics[4*sample_offset+1]*(1.0+y*y);  
  jacobian[offset+7] =-intrinsics[4*sample_offset+1]*x*y;
  jacobian[offset+8] =-intrinsics[4*sample_offset+1]*x;  
  jacobian[offset+9] = 0; 
  jacobian[offset+10]=-intrinsics[4*sample_offset+1]/point_vector_transformed(2);
  jacobian[offset+11]= intrinsics[4*sample_offset+1]*y/point_vector_transformed(2); 
}


#define GET_DATA_POINT(data,x,y) data[batch_id*data_batch_stride+data_channels*(y*data_width+x)+chan]

__global__ void feature_construction_kernel2(const int data_height,const int data_width,const int data_channels,
                                             const int output_data_size,
                                             const int data_batch_stride,
                                             const int warp_batch_stride,
                                             const int output_batch_stride,
                                             const int* projection,const float* weights,
                                             const float* conv1,
                                             const float* conv2,
                                             const float* gradx,
                                             const float* grady,
                                             float* grad,
                                             float* diff) {

  
  
  CUDA_KERNEL_LOOP(index,output_data_size){
    const int out_index      = index;
    const int batch_id       = index / output_batch_stride;
    const int index_in_batch = index % output_batch_stride;

    const int sample_id      = index_in_batch / data_channels;
    const int chan           = index_in_batch % data_channels;
    

    // Get coords of 2D point where data will be resampled
    const int fx = projection[batch_id*warp_batch_stride+sample_id*2];
    const int fy = projection[batch_id*warp_batch_stride+sample_id*2+1];

    if (fx==-1) {
      grad[2*out_index  ]=0.0;
      grad[2*out_index+1]=0.0;
      diff[out_index]    =0.0;
    } else {
      
      const int cx = fx + 1;
      const int cy = fy + 1;
      
      const int weight_offset=2*batch_id*warp_batch_stride+sample_id*4;

      grad[2*out_index  ]=weights[weight_offset]*GET_DATA_POINT(gradx,fx,fy)
                         +weights[weight_offset+1]*GET_DATA_POINT(gradx,cx,fy)
                         +weights[weight_offset+2]*GET_DATA_POINT(gradx,fx,cy)
                         +weights[weight_offset+3]*GET_DATA_POINT(gradx,cx,cy);

      grad[2*out_index+1]=weights[weight_offset]  *GET_DATA_POINT(grady,fx,fy)
                         +weights[weight_offset+1]*GET_DATA_POINT(grady,cx,fy)
                         +weights[weight_offset+2]*GET_DATA_POINT(grady,fx,cy)
                         +weights[weight_offset+3]*GET_DATA_POINT(grady,cx,cy);

      diff[out_index]    =weights[weight_offset]  *GET_DATA_POINT(conv2,fx,fy)
                         +weights[weight_offset+1]*GET_DATA_POINT(conv2,cx,fy)
                         +weights[weight_offset+2]*GET_DATA_POINT(conv2,fx,cy)
                         +weights[weight_offset+3]*GET_DATA_POINT(conv2,cx,cy)-conv1[out_index];
    }
  }
}



class EquationConstructionPrepare:public OpKernel 
{

public:

  explicit EquationConstructionPrepare(OpKernelConstruction* context):OpKernel(context){

  }

  void Compute(OpKernelContext* context ) override {
    //std::cout<<"prepare_start_"<<std::endl;
    const Tensor& points     =context->input(0);
    const Tensor& intrinsic  =context->input(1);
    const Tensor& rotation   =context->input(2);
    const Tensor& translation=context->input(3);
    const Tensor& conv1      =context->input(4);
    const Tensor& conv2      =context->input(5);
    const Tensor& gradx      =context->input(6);
    const Tensor& grady      =context->input(7);
    

    const TensorShape points_shape(points.shape());
    const TensorShape conv1_shape(conv1.shape());
    const TensorShape conv2_shape(conv2.shape());

    int nbatch   =points_shape.dim_size(0);
    int npoints  =points_shape.dim_size(1);
    int nchannels=conv1_shape.dim_size(2);
    int height   =conv2_shape.dim_size(1);
    int width    =conv2_shape.dim_size(2);

    TensorShape jacobian_shape(points_shape);
    jacobian_shape.set_dim(2,2);
    jacobian_shape.AddDim(6);

    Tensor* jacobian_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(0,jacobian_shape,&jacobian_tensor));


    //std::cout<<"compute_prepare_done"<<std::endl;

    TensorShape projection_shape(points_shape);
    projection_shape.set_dim(2,2);
    Tensor projection_tensor;
    OP_REQUIRES_OK(context,context->allocate_temp(DT_INT32,projection_shape,&projection_tensor));

    TensorShape weights_shape(points_shape);
    weights_shape.set_dim(2,4);
    Tensor weights_tensor;
    OP_REQUIRES_OK(context,context->allocate_temp(DT_FLOAT,weights_shape,&weights_tensor));


    TensorShape valid_shape(points_shape);
    valid_shape.set_dim(2,1);
    Tensor* valid_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(3,valid_shape,&valid_tensor));

    float* jacobian  = jacobian_tensor->flat<float>().data();
    int*   projection= projection_tensor.flat<int>().data();
    float* weights   = weights_tensor.flat<float>().data();
    float* valid     = valid_tensor->flat<float>().data();

    CudaLaunchConfig config =GetCudaLaunchConfig(nbatch*npoints);
    jacobian_construction_kernel2<<<config.block_count,config.thread_per_block,0,GetCudaStream(context)>>>(
                                                                          height,width,nbatch,npoints,points.flat<float>().data(),
                                                                          intrinsic.flat<float>().data(),rotation.flat<float>().data(),translation.flat<float>().data(),
                                                                          jacobian,projection,weights,valid);

    TensorShape grad_shape(points_shape);
    grad_shape.set_dim(2,nchannels);
    grad_shape.AddDim(2);
    Tensor* grad_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(1,grad_shape,&grad_tensor));


    TensorShape diff_shape(points_shape);
    diff_shape.set_dim(2,nchannels);
    diff_shape.AddDim(1);

    Tensor* diff_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(2,diff_shape,&diff_tensor));

    const float* conv1_ptr= conv1.flat<float>().data();
    const float* conv2_ptr= conv2.flat<float>().data();
    const float* gradx_ptr= gradx.flat<float>().data();
    const float* grady_ptr= grady.flat<float>().data();
    float*       grad     = grad_tensor->flat<float>().data();
    float*       diff     = diff_tensor->flat<float>().data();
    
    const int output_data_size    = nbatch * npoints * nchannels;
    const int data_batch_stride   = height * width * nchannels;
    const int warp_batch_stride   = npoints* 2;
    const int output_batch_stride = npoints* nchannels;


    config=GetCudaLaunchConfig(output_data_size);
    feature_construction_kernel2<<<config.block_count,config.thread_per_block,0,GetCudaStream(context)>>>(height,width,nchannels,output_data_size,data_batch_stride,warp_batch_stride,output_batch_stride,
                                                                                                          projection,weights,conv1_ptr,conv2_ptr,gradx_ptr,grady_ptr,
                                                                                                          grad,diff);

  }
};
REGISTER_KERNEL_BUILDER(Name("EquationConstructionPrepare").Device(DEVICE_GPU),EquationConstructionPrepare);


REGISTER_OP("EquationConstructionFused")
     .Input("points:float")
     .Input("intrinsics:float")
     .Input("rotation:float")
     .Input("translation:float")
     .Input("conv1:float")
     .Input("conv2:float")
     .Input("gradx:float")
     .Input("grady:float")
     .Output("jacobian:float")
     .Output("grad:float")
     .Output("diff:float")
     .Output("valid:float")
     .SetShapeFn([](shape_inference::InferenceContext* c) {

      shape_inference::ShapeHandle     point_shape=c->input(0);
      std::vector<shape_inference::DimensionHandle> dims1(4);
      dims1[0]=c->Dim(point_shape,0);
      dims1[1]=c->Dim(point_shape,1);
      dims1[2]=c->MakeDim(2);
      dims1[3]=c->MakeDim(6);
      shape_inference::ShapeHandle output1_shape=c->MakeShape(dims1);
      c->set_output(0,output1_shape);

      shape_inference::ShapeHandle     conv_shape=c->input(4);
      dims1[2]=c->MakeDim(2);
      dims1[3]=c->MakeDim(2);
      shape_inference::ShapeHandle output2_shape=c->MakeShape(dims1);
      c->set_output(1,output2_shape);

      dims1[3]=c->MakeDim(1);
      shape_inference::ShapeHandle output3_shape=c->MakeShape(dims1);
      c->set_output(2,output3_shape);

      dims1.resize(3);
      dims1[2]=c->MakeDim(1);
      shape_inference::ShapeHandle output4_shape=c->MakeShape(dims1);
      c->set_output(3,output4_shape);

      return Status::OK();
    });


__global__ void jacobian_construction_fused(const int height,const int width,
                                            const int nsamples,const int npoints,const float* points,
                                            const float* intrinsics,const float* rotation,const float* translation,
                                            float* jacobian,int* projection,float* weights,float* in_boundary,float* grad,float* diff) {

  const int gid=threadIdx.x+blockIdx.x*blockDim.x;
  int sample_offset=gid/npoints;
  int point_offset =gid%npoints;

  if (sample_offset>= nsamples){
    return;
  }


  Eigen::Matrix3f rotation_matrix(rotation+9*sample_offset);
  Eigen::Vector3f transation_vector(translation+3*sample_offset);
  Eigen::Vector3f point_vector(points+3*npoints*sample_offset+3*point_offset);
  Eigen::Vector3f point_vector_transformed=rotation_matrix*point_vector+transation_vector;

  float x=(point_vector_transformed(0)/point_vector_transformed(2)),
        y=(point_vector_transformed(1)/point_vector_transformed(2));

  int offset=2*gid;
  float px=intrinsics[4*sample_offset  ]*x+intrinsics[4*sample_offset+2],
        py=intrinsics[4*sample_offset+1]*y+intrinsics[4*sample_offset+3];
  
  memset(diff+offset  ,0,2*sizeof(float));
  memset(grad+2*offset,0,4*sizeof(float));
  
  if (px<0||px>width-2||py<0||py>height-2){
    projection[offset]  =-1;
    projection[offset+1]=-1;
    in_boundary[gid]=0.0;
    memset(jacobian+12*gid,0,12*sizeof(float));
    return;
  }else{
    in_boundary[gid]=1.0;
    projection[offset]  =static_cast<int>px;
    projection[offset+1]=static_cast<int>py;
    float ax=px-projection[offset],
          ay=py-projection[offset+1];
    offset=4*gid;
    weights[offset]=(1.0-ax)*(1.0-ay);
    weights[offset+1]=(ax)*(1.0-ay);
    weights[offset+2]=(1.0-ax)*ay;
    weights[offset+3]=ax*ay;
  }

  offset=12*gid;
  jacobian[offset]  = intrinsics[4*sample_offset]*x*y;
  jacobian[offset+1]=-intrinsics[4*sample_offset]*(1.0+x*x); 
  jacobian[offset+2]= intrinsics[4*sample_offset]*y; 
  jacobian[offset+3]=-intrinsics[4*sample_offset]/point_vector_transformed(2); 
  jacobian[offset+4]= 0; 
  jacobian[offset+5]= intrinsics[4*sample_offset]*x/point_vector_transformed(2); 

  jacobian[offset+6] = intrinsics[4*sample_offset+1]*(1.0+y*y);  
  jacobian[offset+7] =-intrinsics[4*sample_offset+1]*x*y;
  jacobian[offset+8] =-intrinsics[4*sample_offset+1]*x;  
  jacobian[offset+9] = 0; 
  jacobian[offset+10]=-intrinsics[4*sample_offset+1]/point_vector_transformed(2);
  jacobian[offset+11]= intrinsics[4*sample_offset+1]*y/point_vector_transformed(2); 
}

__global__ void feature_construction_fused(const int data_height,const int data_width,const int data_channels,
                                             const int output_data_size,
                                             const int data_batch_stride,
                                             const int warp_batch_stride,
                                             const int output_batch_stride,
                                             const int* projection,const float* weights,
                                             const float* conv1,
                                             const float* conv2,
                                             const float* gradx,
                                             const float* grady,
                                             float* grad,
                                             float* diff) {
  
  CUDA_KERNEL_LOOP(index,output_data_size){
    const int out_index      = index;
    const int batch_id       = index / output_batch_stride;
    const int index_in_batch = index % output_batch_stride;

    const int sample_id      = index_in_batch / data_channels;
    const int chan           = index_in_batch % data_channels;
    

    // Get coords of 2D point where data will be resampled
    const int proj_offset=batch_id*warp_batch_stride+sample_id*2;
    const int fx = projection[proj_offset];
    const int fy = projection[proj_offset+1];

    if (fx!=-1) {
      
      const int cx = fx + 1;
      const int cy = fy + 1;
      
      const int weight_offset=2*batch_id*warp_batch_stride+sample_id*4;

      float gx=weights[weight_offset]  *GET_DATA_POINT(gradx,fx,fy)
              +weights[weight_offset+1]*GET_DATA_POINT(gradx,cx,fy)
              +weights[weight_offset+2]*GET_DATA_POINT(gradx,fx,cy)
              +weights[weight_offset+3]*GET_DATA_POINT(gradx,cx,cy);

      float gy=weights[weight_offset]  *GET_DATA_POINT(grady,fx,fy)
              +weights[weight_offset+1]*GET_DATA_POINT(grady,cx,fy)
              +weights[weight_offset+2]*GET_DATA_POINT(grady,fx,cy)
              +weights[weight_offset+3]*GET_DATA_POINT(grady,cx,cy);

      
      float dif=weights[weight_offset]*GET_DATA_POINT(conv2,fx,fy)
               +weights[weight_offset+1]*GET_DATA_POINT(conv2,cx,fy)
               +weights[weight_offset+2]*GET_DATA_POINT(conv2,fx,cy)
               +weights[weight_offset+3]*GET_DATA_POINT(conv2,cx,cy)-conv1[out_index];

      atomicAdd(grad+weight_offset  ,gx*gx);
      atomicAdd(grad+weight_offset+1,gx*gy);
      atomicAdd(grad+weight_offset+2,gx*gy);
      atomicAdd(grad+weight_offset+3,gy*gy);


      atomicAdd(diff+proj_offset,  dif*gx);
      atomicAdd(diff+proj_offset+1,dif*gy);
    }
  }
}



class EquationConstructionFused:public OpKernel 
{

public:

  static 

  explicit EquationConstructionFused(OpKernelConstruction* context):OpKernel(context){

  }

  void Compute(OpKernelContext* context ) override {
    //std::cout<<"prepare_start_"<<std::endl;
    const Tensor& points     =context->input(0);
    const Tensor& intrinsic  =context->input(1);
    const Tensor& rotation   =context->input(2);
    const Tensor& translation=context->input(3);
    const Tensor& conv1      =context->input(4);
    const Tensor& conv2      =context->input(5);
    const Tensor& gradx      =context->input(6);
    const Tensor& grady      =context->input(7);
    

    const TensorShape points_shape(points.shape());
    const TensorShape conv1_shape(conv1.shape());
    const TensorShape conv2_shape(conv2.shape());

    int nbatch   =points_shape.dim_size(0);
    int npoints  =points_shape.dim_size(1);
    int nchannels=conv1_shape.dim_size(2);
    int height   =conv2_shape.dim_size(1);
    int width    =conv2_shape.dim_size(2);

    // TensorShape jacobian_shape(points_shape);
    // jacobian_shape.set_dim(2,2);
    // jacobian_shape.AddDim(6);

    // Tensor jacobian_tensor;
    // OP_REQUIRES_OK(context,context->allocate_temp(DT_FLOAT,jacobian_shape,&jacobian_tensor));


    TensorShape jacobian_shape(points_shape);
    jacobian_shape.set_dim(2,2);
    jacobian_shape.AddDim(6);
    Tensor* jacobian_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(0,jacobian_shape,&jacobian_tensor));



    TensorShape projection_shape(points_shape);
    projection_shape.set_dim(2,2);
    Tensor projection_tensor;
    OP_REQUIRES_OK(context,context->allocate_temp(DT_INT32,projection_shape,&projection_tensor));

    TensorShape weights_shape(points_shape);
    weights_shape.set_dim(2,4);
    Tensor weights_tensor;
    OP_REQUIRES_OK(context,context->allocate_temp(DT_FLOAT,weights_shape,&weights_tensor));


    // TensorShape valid_shape(points_shape);
    // valid_shape.set_dim(2,1);
    // Tensor* valid_tensor=NULL;
    // OP_REQUIRES_OK(context,context->allocate_output(3,valid_shape,&valid_tensor));


    // TensorShape grad_shape(points_shape);
    // grad_shape.set_dim(2,2);
    // grad_shape.AddDim(2);
    // Tensor grad_tensor;
    // OP_REQUIRES_OK(context,context->allocate_temp(DT_FLOAT,grad_shape,&grad_tensor));


    // TensorShape diff_shape(points_shape);
    // diff_shape.set_dim(2,2);
    // diff_shape.AddDim(1);
    // Tensor diff_tensor;
    // OP_REQUIRES_OK(context,context->allocate_temp(DT_FLOAT,diff_shape,&diff_tensor));

    TensorShape valid_shape(points_shape);
    valid_shape.set_dim(2,1);
    Tensor* valid_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(3,valid_shape,&valid_tensor));


    TensorShape grad_shape(points_shape);
    grad_shape.set_dim(2,2);
    grad_shape.AddDim(2);
    Tensor* grad_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(1,grad_shape,&grad_tensor));

    TensorShape diff_shape(points_shape);
    diff_shape.set_dim(2,2);
    diff_shape.AddDim(1);
    Tensor* diff_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(2,diff_shape,&diff_tensor));


    float* jacobian  = jacobian_tensor->flat<float>().data();
    int*   projection= projection_tensor.flat<int>().data();
    float* weights   = weights_tensor.flat<float>().data();
    float* valid     = valid_tensor->flat<float>().data();
    float*  grad     = grad_tensor->flat<float>().data();
    float*  diff     = diff_tensor->flat<float>().data();

    CudaLaunchConfig config =GetCudaLaunchConfig(nbatch*npoints);
    jacobian_construction_fused<<<config.block_count,config.thread_per_block,0,GetCudaStream(context)>>>(
                                                                          height,width,nbatch,npoints,points.flat<float>().data(),
                                                                          intrinsic.flat<float>().data(),rotation.flat<float>().data(),translation.flat<float>().data(),
                                                                          jacobian,projection,weights,valid,grad,diff);

    const float* conv1_ptr= conv1.flat<float>().data();
    const float* conv2_ptr= conv2.flat<float>().data();
    const float* gradx_ptr= gradx.flat<float>().data();
    const float* grady_ptr= grady.flat<float>().data();
    // float*       grad     = grad_tensor->flat<float>().data();
    // float*       diff     = diff_tensor->flat<float>().data();
    
    const int output_data_size    = nbatch * npoints * nchannels;
    const int data_batch_stride   = height * width * nchannels;
    const int warp_batch_stride   = npoints* 2;
    const int output_batch_stride = npoints* nchannels;


    config=GetCudaLaunchConfig(output_data_size);
    feature_construction_fused<<<config.block_count,config.thread_per_block,0,GetCudaStream(context)>>>(height,width,nchannels,output_data_size,data_batch_stride,warp_batch_stride,output_batch_stride,
                                                                                                        projection,weights,conv1_ptr,conv2_ptr,gradx_ptr,grady_ptr,
                                                                                                        grad,diff);

  }
};
REGISTER_KERNEL_BUILDER(Name("EquationConstructionFused").Device(DEVICE_GPU),EquationConstructionFused);



__global__ void hessian_indexing(const int jacobian_rows,const int index_size,int *indexing) {
  const int gid=threadIdx.x+blockIdx.x*blockDim.x;
  int sample_offset=gid;
  int offset=2*(sample_offset*index_size);
  for (int i = 0; i < jacobian_rows; ++i){
    for (int j = i; j < jacobian_rows; ++j){
      indexing[offset]  =i;
      indexing[offset+1]=j;
      offset+=2;
    }
  }
}

__global__ void left_construction_fused(const int jacobian_rows,
                                        const int output_data_size,
                                        const int output_batch_stride,
                                        const int* hessian_index,
                                        const float* jacobians,
                                        const float* grads,
                                        float* lefts) {

  CUDA_KERNEL_LOOP(index,output_data_size){

    const int out_index      = index;
    const int batch_id       = index / output_batch_stride;
    const int index_in_batch = index % output_batch_stride;

    const int sample_id      = index_in_batch / data_channels;
    const int chan           = index_in_batch % data_channels;

    const int chan_offset    = 2*chan;
    const int row_index      = hessian_index[chan_offset];
    const int col_index      = hessian_index[chan_offset+1];

    
  }
}


// __global__ void equation_construction_fused(const int jacobian_rows,
//                                             const int output_data_size,
//                                             const int output_batch_stride,
//                                             const int* hessian_index,
//                                             const float* jacobians,
//                                             const float* grads,
//                                             const float* diffs
//                                             float* lefts,
//                                             float* rights) {

//   CUDA_KERNEL_LOOP(index,output_data_size){

//     int jaocbian_offset=12*index;
//     Eigen::Matrix<float,6,2,Eigen::ColMajor> Jt(jacobian+jaocbian_offset);
//     Eigen::Matrix<float,2,6,Eigen::RowMajor> J (jacobian+jaocbian_offset);
    
//     int grad_offset=4*index;
//     Eigen::Matrix2f grad(grads+grad_offset);

//     int diff_offset=2*index;
//     Eigen::Vector2f diff(diffs+diff_offset);

//     Eigen::Matrix<float,6,6> JtJ=Jt*grad*J;
//     Eigen::Vector2f Jtb         =Jt*diff;
//   }
// }


REGISTER_OP("EquationConstructionFused2")
     .Input("points:float")
     .Input("intrinsics:float")
     .Input("rotation:float")
     .Input("translation:float")
     .Input("conv1:float")
     .Input("conv2:float")
     .Input("gradx:float")
     .Input("grady:float")
     .Output("jacobian:float")
     .Output("grad:float")
     .Output("diff:float")
     .Output("valid:float")
     .SetShapeFn([](shape_inference::InferenceContext* c) {

      shape_inference::ShapeHandle     point_shape=c->input(0);
      std::vector<shape_inference::DimensionHandle> dims1(4);
      dims1[0]=c->Dim(point_shape,0);
      dims1[1]=c->Dim(point_shape,1);
      dims1[2]=c->MakeDim(2);
      dims1[3]=c->MakeDim(6);
      shape_inference::ShapeHandle output1_shape=c->MakeShape(dims1);
      c->set_output(0,output1_shape);

      shape_inference::ShapeHandle     conv_shape=c->input(4);
      dims1[2]=c->MakeDim(2);
      dims1[3]=c->MakeDim(2);
      shape_inference::ShapeHandle output2_shape=c->MakeShape(dims1);
      c->set_output(1,output2_shape);

      dims1[3]=c->MakeDim(1);
      shape_inference::ShapeHandle output3_shape=c->MakeShape(dims1);
      c->set_output(2,output3_shape);

      dims1.resize(3);
      dims1[2]=c->MakeDim(1);
      shape_inference::ShapeHandle output4_shape=c->MakeShape(dims1);
      c->set_output(3,output4_shape);

      return Status::OK();
    });


class EquationConstructionFused2:public OpKernel 
{

public:

  static 

  explicit EquationConstructionFused2(OpKernelConstruction* context):OpKernel(context){

  }

  void Compute(OpKernelContext* context ) override {
    //std::cout<<"prepare_start_"<<std::endl;
    const Tensor& points     =context->input(0);
    const Tensor& intrinsic  =context->input(1);
    const Tensor& rotation   =context->input(2);
    const Tensor& translation=context->input(3);
    const Tensor& conv1      =context->input(4);
    const Tensor& conv2      =context->input(5);
    const Tensor& gradx      =context->input(6);
    const Tensor& grady      =context->input(7);
    

    const TensorShape points_shape(points.shape());
    const TensorShape conv1_shape(conv1.shape());
    const TensorShape conv2_shape(conv2.shape());

    int nbatch   =points_shape.dim_size(0);
    int npoints  =points_shape.dim_size(1);
    int nchannels=conv1_shape.dim_size(2);
    int height   =conv2_shape.dim_size(1);
    int width    =conv2_shape.dim_size(2);

    // TensorShape jacobian_shape(points_shape);
    // jacobian_shape.set_dim(2,2);
    // jacobian_shape.AddDim(6);

    // Tensor jacobian_tensor;
    // OP_REQUIRES_OK(context,context->allocate_temp(DT_FLOAT,jacobian_shape,&jacobian_tensor));


    TensorShape jacobian_shape(points_shape);
    jacobian_shape.set_dim(2,2);
    jacobian_shape.AddDim(6);
    Tensor* jacobian_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(0,jacobian_shape,&jacobian_tensor));



    TensorShape projection_shape(points_shape);
    projection_shape.set_dim(2,2);
    Tensor projection_tensor;
    OP_REQUIRES_OK(context,context->allocate_temp(DT_INT32,projection_shape,&projection_tensor));

    TensorShape weights_shape(points_shape);
    weights_shape.set_dim(2,4);
    Tensor weights_tensor;
    OP_REQUIRES_OK(context,context->allocate_temp(DT_FLOAT,weights_shape,&weights_tensor));


    // TensorShape valid_shape(points_shape);
    // valid_shape.set_dim(2,1);
    // Tensor* valid_tensor=NULL;
    // OP_REQUIRES_OK(context,context->allocate_output(3,valid_shape,&valid_tensor));


    // TensorShape grad_shape(points_shape);
    // grad_shape.set_dim(2,2);
    // grad_shape.AddDim(2);
    // Tensor grad_tensor;
    // OP_REQUIRES_OK(context,context->allocate_temp(DT_FLOAT,grad_shape,&grad_tensor));


    // TensorShape diff_shape(points_shape);
    // diff_shape.set_dim(2,2);
    // diff_shape.AddDim(1);
    // Tensor diff_tensor;
    // OP_REQUIRES_OK(context,context->allocate_temp(DT_FLOAT,diff_shape,&diff_tensor));

    TensorShape valid_shape(points_shape);
    valid_shape.set_dim(2,1);
    Tensor* valid_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(3,valid_shape,&valid_tensor));


    TensorShape grad_shape(points_shape);
    grad_shape.set_dim(2,2);
    grad_shape.AddDim(2);
    Tensor* grad_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(1,grad_shape,&grad_tensor));

    TensorShape diff_shape(points_shape);
    diff_shape.set_dim(2,2);
    diff_shape.AddDim(1);
    Tensor* diff_tensor=NULL;
    OP_REQUIRES_OK(context,context->allocate_output(2,diff_shape,&diff_tensor));


    float* jacobian  = jacobian_tensor->flat<float>().data();
    int*   projection= projection_tensor.flat<int>().data();
    float* weights   = weights_tensor.flat<float>().data();
    float* valid     = valid_tensor->flat<float>().data();
    float*  grad     = grad_tensor->flat<float>().data();
    float*  diff     = diff_tensor->flat<float>().data();

    CudaLaunchConfig config =GetCudaLaunchConfig(nbatch*npoints);
    jacobian_construction_fused<<<config.block_count,config.thread_per_block,0,GetCudaStream(context)>>>(
                                                                          height,width,nbatch,npoints,points.flat<float>().data(),
                                                                          intrinsic.flat<float>().data(),rotation.flat<float>().data(),translation.flat<float>().data(),
                                                                          jacobian,projection,weights,valid,grad,diff);

    const float* conv1_ptr= conv1.flat<float>().data();
    const float* conv2_ptr= conv2.flat<float>().data();
    const float* gradx_ptr= gradx.flat<float>().data();
    const float* grady_ptr= grady.flat<float>().data();
    // float*       grad     = grad_tensor->flat<float>().data();
    // float*       diff     = diff_tensor->flat<float>().data();
    
    const int output_data_size    = nbatch * npoints * nchannels;
    const int data_batch_stride   = height * width * nchannels;
    const int warp_batch_stride   = npoints* 2;
    const int output_batch_stride = npoints* nchannels;


    config=GetCudaLaunchConfig(output_data_size);
    feature_construction_fused<<<config.block_count,config.thread_per_block,0,GetCudaStream(context)>>>(height,width,nchannels,output_data_size,data_batch_stride,warp_batch_stride,output_batch_stride,
                                                                                                        projection,weights,conv1_ptr,conv2_ptr,gradx_ptr,grady_ptr,
                                                                                                        grad,diff);

    TensorShape element_shape(points_shape);
    projection_shape.set_dim(2,2);
    Tensor projection_tensor;
    OP_REQUIRES_OK(context,context->allocate_temp(DT_INT32,projection_shape,&projection_tensor));

  }
};
REGISTER_KERNEL_BUILDER(Name("EquationConstructionFused").Device(DEVICE_GPU),EquationConstructionFused);





