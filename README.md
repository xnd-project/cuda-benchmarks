
### Cuda benchmarks for unified vs. explicit memory


## Unified memory

Unified memory has been a feature of game consoles for many years. It simplifies game development because it frees the programmer from having to track whether a memory block is on CPU or GPU memory.


Starting with the Pascal architecture, Nvidia also offers [advanced unified memory support](https://devblogs.nvidia.com/unified-memory-cuda-beginners/).


Game consoles can have tighter hardware control, but Nvidia also has the [Jetson product line](https://en.wikipedia.org/wiki/Nvidia_Jetson) with physically unified memory that has been [reported](https://devtalk.nvidia.com/default/topic/1029853/does-unified-memory-and-zero-copy-always-better-than-cudamemcpy-/) to have better performance than explicit memory management.


## Unified memory reputation

There have been several reports, many of them from around 2014, that unified memory has a lower performance in many scenarios. The reports usually do not come with links to actual code.

Recent reports appear to address only specialty scenarios like [8 NVlink GPUs](https://devtalk.nvidia.com/default/topic/1029706/cuda-programming-and-performance/partial-fail-of-peer-access-in-8-volta-gpu-instance-p3-16xlarge-on-aws-gt-huge-slowdown-/).

The report also has no code link.



## Unified memory reality

This benchmark suite attempts to provide actual code so that people can check for themselves. It is incomplete and does not yet address scenarios like IPC or multiple GPUs. Benchmarks that show the superiority of explicit memory management are welcome.


## Examples

The examples are run on Linux with Cuda release 9.2, V9.2.148, using a GeForce 1060 with 6GB of memory.

# simpleManaged vs. simpleMemcpy

This benchmark tests initializing three arrays in host memory, running a kernel and accessing the result. Most of the time is spent in copying, the kernel runtime is negligible.

With N=200000000, explicit memory performs slightly better:


```
$ ./simpleManaged 
host: malloc: 0.210889
host: init arrays: 0.655248
device: uvm+compute+synchronize: 0.954883
host: access all arrays: 0.928161
host: access all arrays a second time: 0.244277
host: free: 0.175481
total: 3.169012

$ ./simpleMemcpy 
host: malloc: 0.926996
host: init arrays: 0.269095
device: malloc+copy+compute: 1.108133
host: access all arrays: 0.238951
host: access all arrays a second time: 0.239062
host: free: 0.000005
total: 2.782294
```

With N=500000000, managed memory has no issues, but explicit memory does not run at all:


```
$ ./simpleManaged 
host: malloc: 0.208101
host: init arrays: 1.631877
device: uvm+compute+synchronize: 2.262058
host: access all arrays: 1.643403
host: access all arrays a second time: 0.607834
host: free: 0.380680
total: 6.734027
$ 
$ ./simpleMemcpy 
host: malloc: 1.970601
host: init arrays: 0.675828
cudaErrorMemoryAllocation

```

# cuBLAS: gemmManaged vs. gemmMemcpy

This benchmark calls the cublasSgemm() function.

With N=8000, managed memory is considerably faster:

```
$ ./gemmManaged
host: cudaMallocManaged+init: 0.195686
cublasSgemm: 1.421310
host: access all arrays: 0.000100
host: access all arrays a second time: 0.000012
host: free: 0.029193
total: 1.965400

$ ./gemmMemcpy 
host: cudaMallocHost+init: 0.238533
cublasSgemm: 3.342471
host: access all arrays: 0.000031
host: access all arrays a second time: 0.000008
host: free: 0.000002
total: 3.899770
```

With N=16000, managed memory is not only considerably faster, but explicit memory performance is catastrophic:

```
$ ./gemmManaged
host: cudaMallocManaged+init: 0.761136
cublasSgemm: 3.388252
host: access all arrays: 0.000108
host: access all arrays a second time: 0.000030
host: free: 0.083837
total: 4.546474
$ 
$ ./gemmMemcpy 
host: cudaMallocHost+init: 0.943743
cublasSgemm: 36.860626
host: access all arrays: 0.000039
host: access all arrays a second time: 0.000016
host: free: 0.000002
total: 38.120901
```


# cuBlas+Managed vs. cuBlasXt+HostMemory

cuBlasXt handles out-of-core computations for memory that is allocated with cudaMallocHost(). This benchmark compares the cublasSgemm() function running on managed memory vs. the cublasXtSgemm() function running on host allocated memory.

Note that cublasXtSgemm() is designed to run on host allocated memory and handles optimized tiled memory transfers. Also note that cuBlasXt has more functionality (multiple cards), so the slightly worse performance is not surprising.

The point of this comparison, however, is that managed memory performs very well using the standard cuBlas function.

```
$ ./gemmManagedOutOfCore 
host: cudaMallocManaged+init: 3.041447
cublasSgemm: 20.852501
$ 
$ ./gemmXtOutOfCore 
host: cudaMallocHost+init: 3.749518
cublasXtSgemm: 25.600282
```
