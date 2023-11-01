# 操作系统实验3：遍历地址空间
1120210526 马煜

## 运行
```cmd
$ cd <your-file-path>
$ g++ main.cpp -o main.exe -lpsapi && .\main.exe>out.txt
```

结果输出在了out.txt中

# 1.获取系统当前内存设置信息
## GetSystemInfo
[GetSystemInfo()](https://learn.microsoft.com/zh-cn/windows/win32/api/sysinfoapi/ns-sysinfoapi-system_info)

*在32位机中为GetNativeSystemInfo函数

[SYSTEM_INFO 结构](https://learn.microsoft.com/zh-cn/windows/win32/api/sysinfoapi/ns-sysinfoapi-system_info)

```c++
typedef struct _SYSTEM_INFO {
  union {
    DWORD dwOemId;
    struct {
      WORD wProcessorArchitecture;
      WORD wReserved;
    } DUMMYSTRUCTNAME;
  } DUMMYUNIONNAME;
  DWORD     dwPageSize;
  LPVOID    lpMinimumApplicationAddress;
  LPVOID    lpMaximumApplicationAddress;
  DWORD_PTR dwActiveProcessorMask;
  DWORD     dwNumberOfProcessors;
  DWORD     dwProcessorType;
  DWORD     dwAllocationGranularity;
  WORD      wProcessorLevel;
  WORD      wProcessorRevision;
} SYSTEM_INFO, *LPSYSTEM_INFO;
```
### 成员含义
#### dwPageSize
 Granularity（粒度）是虚拟内存分配中重要的概念之一，它是指内存页面大小的单位。在 Windows 中，虚拟内存分配时，系统会把实际的物理内存空间分成大小相等的块，称为页面。这些页面的大小通常是2的幂次方，如4KB、8KB、16KB等。粒度越小，可以更精细地管理内存分配，但也会增加内存分配的开销；而粒度越大，则可以节省内存分配的开销，但可能会浪费部分内存空间。

在 Windows 中，虚拟内存的基本单位是虚拟页（Virtual Page），虚拟页与物理页（Physical Page）是一一对应的。当进程需要访问某个虚拟地址时，系统会根据该地址对应的虚拟页号，找到对应的物理页号，从而访问相应的物理内存。

在虚拟内存的分配和管理中，粒度的大小对系统性能和资源利用效率都有一定的影响。通常情况下，Windows 中的默认页面大小为4KB，这是一种比较折中的选择，能够平衡内存管理的效率和开销，同时也便于与其他操作系统进行交互。但在特定场景下，可能需要调整页面大小，以更好地适应应用程序的需求。

#### lpMinimumApplicationAddress
指向应用程序和动态链接库(DLL)可以访问的最低内存地址。

#### dwAllocationGranularity
可以分配虚拟内存的起始地址的粒度。

dwAllocationGranularity 是在使用 VirtualAlloc 函数分配虚拟内存时的一项重要参数。它表示虚拟内存的分配粒度，也称为虚拟内存页面大小，它决定了分配的虚拟内存空间将以多大的粒度进行分配。

具体来说，dwAllocationGranularity 是一个系统定义的常量，通常在 Windows 操作系统中的大部分系统上都是 64 KB（65536 字节）。这意味着，当您调用 VirtualAlloc 分配虚拟内存时，系统会以 64 KB 的倍数来分配内存块，而不是按照字节精确分配。

例如，如果您请求分配 4 KB 的虚拟内存，实际上系统可能会分配 64 KB 的虚拟内存，并将其余的内存留作备用。这是因为虚拟内存的分配以页面为单位进行操作，而页面的大小由 dwAllocationGranularity 决定。

### 实验代码
```c++
SYSTEM_INFO sysInfo;
    // 系统信息
    GetSystemInfo(&sysInfo);
    std::cout << "System Memory Information:\n";
    std::cout << "Page Size: " << float(sysInfo.dwPageSize) / DIV << " KB\n";
    std::cout << "Minimum Application Address: " << sysInfo.lpMinimumApplicationAddress << "\n";
    std::cout << "Maximum Application Address: " << sysInfo.lpMaximumApplicationAddress << "\n";
    std::cout << "Allocation Granularity: " << sysInfo.dwAllocationGranularity / DIV << " KB\n\n";

```
### 实验结果
```txt
Page Size: 4 KB
Minimum Application Address: 0x10000
Maximum Application Address: 0x7ffffffeffff
Allocation Granularity: 64 KB
```
根据所学和结果可以看到，如果地址对应的内存已经被保留了，那么将向下偏移至64K的整数倍，如果这块内存已经被提交，那么地址将向下偏移至4K的整数倍，也就是说保留页面的最小粒度是64K，而提交的最小粒度是一页4K。
 
## GlobalMemoryStatusEx
[GlobalMemoryStatus(弃用)](https://learn.microsoft.com/zh-cn/windows/win32/api/winbase/nf-winbase-globalmemorystatus)

[GlobalMemoryStatusEx](https://learn.microsoft.com/zh-cn/windows/win32/api/sysinfoapi/nf-sysinfoapi-globalmemorystatusex)

[MEMORYSTATUSEX 结构](https://learn.microsoft.com/zh-cn/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex)

```c++
typedef struct _MEMORYSTATUSEX {
  DWORD     dwLength;
  DWORD     dwMemoryLoad;
  DWORDLONG ullTotalPhys;
  DWORDLONG ullAvailPhys;
  DWORDLONG ullTotalPageFile;
  DWORDLONG ullAvailPageFile;
  DWORDLONG ullTotalVirtual;
  DWORDLONG ullAvailVirtual;
  DWORDLONG ullAvailExtendedVirtual;
} MEMORYSTATUSEX, *LPMEMORYSTATUSEX;
```

### 成员解释
#### dwLength
通常在调用 API 函数前需要设置结构体的 dwLength 成员，以确保函数能够正确识别结构体的版本和大小。这是一种用于处理不同版本的 Windows API 的通用做法，以保持向后兼容性。

#### ullTotalPhys
实际物理内存量（以字节为单位）。
#### ullAvailPhys
当前可用的物理内存量（以字节为单位）。 这是可以立即重复使用的物理内存量，而无需先将其内容写入磁盘。 它是备用列表、可用列表和零列表的大小之和。（It is the sum of the size of the standby, free, and zero lists.）
#### ullTotalVirtual
调用进程的虚拟地址空间的用户模式部分的大小（以字节为单位）。 此值取决于进程类型、处理器类型和操作系统的配置。 例如，对于 x86 处理器上的大多数 32 位进程，此值约为 2 GB，对于在启用了 4 GB 优化 的系统上运行的大地址感知的 32 位进程，此值约为 3 GB。

#### ullAvailVirtual
当前位于调用进程的虚拟地址空间的用户模式部分中的未保留和未提交的内存量（以字节为单位）。

#### ullTotalPageFile
系统或当前进程的当前已提交内存限制，以字节为单位，以较小者为准。

#### ullAvailPageFile
当前进程可以提交的最大内存量（以字节为单位）。 此值等于或小于系统范围的可用提交值。 若要计算系统范围的可用提交值，请调用 GetPerformanceInfo，并从 CommitLimit 的值中减去 CommitTotal 的值。


### 实验代码
```c++
MEMORYSTATUSEX memoryStatus;
    memoryStatus.dwLength = sizeof(MEMORYSTATUSEX); // 通常在调用 API 函数前需要设置结构体的 dwLength 成员，以确保函数能够正确识别结构体的版本和大小。
    // 这是一种用于处理不同版本的 Windows API 的通用做法，以保持向后兼容性。

    if (GlobalMemoryStatusEx(&memoryStatus))
    {
        std::cout << "There is " << memoryStatus.dwMemoryLoad << " percent of memory in use.\n";

        std::cout << "Total Physical Memory: " << float(memoryStatus.ullTotalPhys) / DIV / DIV / DIV << " GB" << std::endl;
        std::cout << "Available Physical Memory: " << float(memoryStatus.ullAvailPhys) / DIV / DIV / DIV << " GB" << std::endl;
        std::cout << "Total Virtual Memory: " << float(memoryStatus.ullTotalVirtual) / DIV / DIV / DIV / DIV << " TB" << std::endl;
        std::cout << "Available Virtual Memory: " << float(memoryStatus.ullAvailVirtual) / DIV / DIV / DIV / DIV << " TB" << std::endl;
        std::cout << "Total Page File: " << float(memoryStatus.ullTotalPageFile) / DIV / DIV / DIV << " GB" << std::endl;
        std::cout << "Available Page File: " << float(memoryStatus.ullAvailPageFile) / DIV / DIV / DIV << " GB" << std::endl
                  << std::endl;
    }
    else
    {
        std::cerr << "Failed to retrieve memory status." << std::endl;
    }
```
### 实验结果
```txt
There is 57 percent of memory in use.
Total Physical Memory: 23.6848 GB
Available Physical Memory: 10.0346 GB
Total Virtual Memory: 128 TB
Available Virtual Memory: 127.996 TB
Total Page File: 45.6848 GB
Available Page File: 25.2585 GB
```

针对程序得到巨大虚拟地址空间的解释:
```txt
Total Virtual Memory: 128 TB
Available Virtual Memory: 127.996TB
```
底层的Intel第四代CPU架构支持完全的64位虚拟和物理地址空间，但现在的（以及可预见的未来的）Core i7实现支持48位（256TB）虚拟地址空间和52位（4PB）物理地址空间，还有一个兼容模式支持32位虚拟和物理地址空间。


## getPerformanceInfo
[getPerformanceInfo](https://learn.microsoft.com/zh-cn/windows/win32/api/psapi/nf-psapi-getperformanceinfo)

[PERFORMANCE_INFORMATION 结构](https://learn.microsoft.com/zh-cn/windows/win32/api/psapi/ns-psapi-performance_information)

```c++
typedef struct _PERFORMANCE_INFORMATION {
  DWORD  cb;
  SIZE_T CommitTotal;
  SIZE_T CommitLimit;
  SIZE_T CommitPeak;
  SIZE_T PhysicalTotal;
  SIZE_T PhysicalAvailable;
  SIZE_T SystemCache;
  SIZE_T KernelTotal;
  SIZE_T KernelPaged;
  SIZE_T KernelNonpaged;
  SIZE_T PageSize;
  DWORD  HandleCount;
  DWORD  ProcessCount;
  DWORD  ThreadCount;
} PERFORMANCE_INFORMATION, *PPERFORMANCE_INFORMATION, PERFORMACE_INFORMATION, *PPERFORMACE_INFORMATION;
```
### 成员解释
#### CommitTotal

系统当前提交的页数。 请注意， (使用 VirtualAlloc 和 MEM_COMMIT) 提交页面会立即更改此值;但是，在访问页面之前，不会对物理内存收费。

#### CommitLimit

系统在不扩展分页文件 () 的情况下可以提交的当前最大页数。 如果添加或删除内存，或者页面文件已增长、收缩或已添加，则此数字可能会更改。 如果可以扩展分页文件，则这是一个软限制。

#### CommitPeak

自上次系统重新启动以来同时处于已提交状态的最大页数。

#### SystemCache

系统缓存内存量（以页为单位）。 这是备用列表加上系统工作集的大小。

#### KernelTotal

分页内核池和非分页内核池中的当前内存总和（以页为单位）。

#### KernelPaged

分页内核池中的当前内存（以页为单位）。

#### KernelNonpaged

当前位于非分页内核池中的内存（以页为单位）。

### 实验代码
```c++
PERFORMANCE_INFORMATION pi;
    GetPerformanceInfo(&pi, sizeof(pi));
    DWORDLONG page_size = pi.PageSize;

    std::cout << "Commit Total              \t" << float(pi.CommitTotal) / DIV / DIV << " M-pages" << std::endl;
    std::cout << "Commit Total              \t" << float(pi.CommitTotal * page_size) / DIV / DIV / DIV << " GB" << std::endl;
    std::cout << "Commit Limit              \t" << float(pi.CommitLimit) / DIV / DIV << " M-pages" << std::endl;
    std::cout << "Commit Limit              \t" << float(pi.CommitLimit * page_size) / DIV / DIV / DIV << " GB" << std::endl;
    std::cout << "Commit Peak               \t" << float(pi.CommitPeak) / DIV / DIV << " M-pages" << std::endl;
    std::cout << "Commit Peak              \t" << float(pi.CommitPeak * page_size) / DIV / DIV / DIV << " GB" << std::endl;
    std::cout << "Physical Memory           \t" << float(pi.PhysicalTotal) / DIV / DIV << " M-pages" << std::endl;
    std::cout << "Physical Memory           \t" << float(pi.PhysicalTotal * page_size) / DIV / DIV / DIV << " GB" << std::endl;
    std::cout << "Physical Memory Available \t" << float(pi.PhysicalAvailable) / DIV / DIV << " M-pages" << std::endl;
    std::cout << "Physical Memory Available \t" << float(pi.PhysicalAvailable * page_size) / DIV / DIV / DIV << " GB" << std::endl;
    std::cout << "System Cache              \t" << float(pi.SystemCache) / DIV / DIV << " M-pages" << std::endl;
    std::cout << "Kernel Total              \t" << float(pi.KernelTotal) / DIV / DIV << " M-pages" << std::endl;
    std::cout << "Kernel Paged              \t" << float(pi.KernelPaged) / DIV / DIV << " M-pages" << std::endl;
    std::cout << "Kernel Nonpaged           \t" << float(pi.KernelNonpaged) / DIV / DIV << " M-pages" << std::endl;
    std::cout << "Handle Count              \t" << pi.HandleCount << std::endl;
    std::cout << "Process Count             \t" << pi.ProcessCount << std::endl;
    std::cout << "Thread Count              \t" << pi.ThreadCount << std::endl
              << std::endl;
```
### 实验结果
```txt
Commit Total              	5.10657 M-pages
Commit Total              	20.4263 GB
Commit Limit              	11.4212 M-pages
Commit Limit              	45.6848 GB
Commit Peak               	5.65533 M-pages
Commit Peak              	22.6213 GB
Physical Memory           	5.92119 M-pages
Physical Memory           	23.6848 GB
Physical Memory Available 	2.50864 M-pages
Physical Memory Available 	10.0345 GB
System Cache              	1.21428 M-pages
Kernel Total              	0.394457 M-pages
Kernel Paged              	0.169461 M-pages
Kernel Nonpaged           	0.224996 M-pages
Handle Count              	234602
Process Count             	398
Thread Count              	5530
```
可以从逻辑上看出这个函数主要从页的角度来看内存的使用情况，而不是从内存的大小角度来看，所以这里的内存使用情况和上面的函数得到的结果有所不同。但转化之后是相同的
# 2.遍历当前进程地址空间(虚拟内存)，显示每个虚拟内存区域的特性
## VirtualQuery
[VirtualQuery](https://learn.microsoft.com/zh-cn/windows/win32/api/memoryapi/nf-memoryapi-virtualquery)

[MEMORY_BASIC_INFORMATION 结构](https://learn.microsoft.com/zh-cn/windows/win32/api/winnt/ns-winnt-memory_basic_information)

```c++
typedef struct _MEMORY_BASIC_INFORMATION {
  PVOID  BaseAddress;
  PVOID  AllocationBase;
  DWORD  AllocationProtect;
  WORD   PartitionId;
  SIZE_T RegionSize;
  DWORD  State;
  DWORD  Protect;
  DWORD  Type;
} MEMORY_BASIC_INFORMATION, *PMEMORY_BASIC_INFORMATION;
```

### 成员含义
#### BaseAddress

指向页区域的基址的指针。起始地址

#### AllocationBase

#### RegionSize

从基址开始的区域大小，其中所有页都具有相同的属性（以字节为单位）。
虽然在同一虚拟内存区域内所有页都具有相同的属性，但在系统中的不同虚拟内存区域之间，它们的大小可以各不相同。

#### State
表示该地址的状态，分为提交，保留和FREE

#### Protect
读写权限

#### Type
表示该地址的类型，分为私有，映射文件

### 实验结果
```txt
Executable File: E:\MAYU\Documents\A-Colt-Crossing-a-River\system\HW3\main.exe

Base Address: 0x10000
Allocation Base: 0x10000
Region Size: 64 KB
End Address: 0x20000
State: Committed
Protect: Read Write
Type: Mapped
Module Name: 

……
Base Address: 0x2780000
Allocation Base: 0
Region Size: 1.4812 GB
End Address: 0x61440000
State: Free
Protect: No Access
Module Name: E:\MAYU\Documents\A-Colt-Crossing-a-River\system\HW3\main.exe

……

Base Address: 0x6494d000
Allocation Base: 0x64940000
Region Size: 4 KB
End Address: 0x6494e000
State: Committed
Protect: Read Write
Type: Image
Module Name: C:\Users\MaYu\anaconda3\envs\joyrl\Library\mingw-w64\bin\libwinpthread-1.dll


……
```
# 3.在当前进程地址空间，分配 1GB 虚拟内存，对其中 1MB 清0
## VirtualAlloc
[VirtualAlloc](https://learn.microsoft.com/zh-cn/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc)

### 参数描述
#### MEM_COMMIT
	

提交（区域状态从RESERVE到COMMIT），也就是说将虚拟地址映射到对应的真实物理内存中，这样这块内存就可以正常使用

#### MEM_RESERVE
	

保留（区域状态从FREE到RESERVE），告知系统以这个地址开始到后面的dwSize大小的连续的虚拟内存程序要使用，进程其他分配内存的操作不得使用这段内存。

### 实验代码
以上逻辑可以从以下代码段检验
```c++
if (mbi.RegionSize > 1 * GB && mbi.State == MEM_FREE)
        {
            // buffer = (char *)VirtualAlloc((LPVOID)p, 1 * GB, MEM_RESERVE, PAGE_READWRITE);
            buffer = (char *)VirtualAlloc((LPVOID)p, 1 * GB, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);//这里如果只是MEM_COMMIT，那么后面的VirtualAlloc会失败。MEM_RESERVE | MEM_COMMIT同时保留和提交可用页面区域
            if (buffer == NULL)
            {
                std::cerr << "VirtualAlloc failed\n";
                return 1;
            }
```

关于这部分，文档中有这样的解释
    
    每个页面都有一个关联的 页状态。 VirtualAlloc 函数可以执行以下操作：

        提交保留页的区域
        保留可用页面区域
        同时保留和提交可用页面区域

    VirtualAlloc 无法保留保留页。 它可以提交已提交的页面。 这意味着，无论页面是否已提交，都可以提交一系列页面，并且函数不会失败。

    可以使用“VirtualAlloc”保留一个页面块，然后对“VirtualAlloc”进行其他调用，以提交保留块中的各个页面。 这使进程能够保留其虚拟地址空间的范围，而无需使用物理存储，直到需要为止。

### 实验结果
```txt
1GB of virtual memory allocated， Region information:

Base Address: 0x2780000
Allocation Base: 0x2780000
Region Size: 1 GB
End Address: 0x42780000
State: Committed
Protect: Read Write
Type: Private
Module Name: 

Added Virtual Page File: 1.00321 GB

1MB cleared to 0
```
可以看到程序在第一段足够大的FREE区域分配了1GB的虚拟内存空间

## VirtualFree
[VirtualFree](https://learn.microsoft.com/zh-cn/windows/win32/api/memoryapi/nf-memoryapi-virtualfree)
函数可以取消提交已提交的页面、释放页面的存储，也可以同时取消提交和释放已提交的页面。 它还可以释放保留页，使其成为FREE页面。


