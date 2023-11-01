#include <iostream>
#include <Windows.h>
#include <Psapi.h>
#include <shlwapi.h>

// Use to convert bytes to KB and KB to MB ……
#define DIV 1024

SIZE_T PrintRegion(char *p)
{
    MEMORY_BASIC_INFORMATION mbi;
    VirtualQuery(p, &mbi, sizeof(mbi));
    std::cout << "Base Address: " << (void *)mbi.BaseAddress << "\n";
    std::cout << "Allocation Base: " << (void *)mbi.AllocationBase << "\n";
    if (mbi.RegionSize >= DIV * DIV * DIV)
    {

        std::cout << "Region Size: " << float(mbi.RegionSize) / DIV / DIV / DIV << " GB\n";
    }
    else if (DIV < mbi.RegionSize <DIV * DIV)
    {
        std::cout << "Region Size: " << float(mbi.RegionSize) / DIV << " KB\n";
    }
    else
    {
        std::cout << "Region Size: " << mbi.RegionSize << " bytes\n";
    }

    void *endAddress = reinterpret_cast<void *>(reinterpret_cast<char *>(mbi.BaseAddress) + mbi.RegionSize);
    std::cout << "End Address: " << endAddress << "\n";
    // std::cout << "State: ";
    switch (mbi.State)
    {
    case MEM_COMMIT:
        std::cout << "State: Committed\n";
        break;
    case MEM_FREE:
        std::cout << "State: Free\n";
        break;
    case MEM_RESERVE:
        std::cout << "State: Reserved\n";
        break;
    default:
        break;
    }
    // std::cout << "Protect: ";
    switch (mbi.Protect)
    {
    case PAGE_EXECUTE:
        std::cout << "Protect: Execute\n";
        break;
    case PAGE_EXECUTE_READ:
        std::cout << "Protect: Execute Read\n";
        break;
    case PAGE_EXECUTE_READWRITE:
        std::cout << "Protect: Execute Read Write\n";
        break;
    case PAGE_EXECUTE_WRITECOPY:
        std::cout << "Protect: Execute Write Copy\n";
        break;
    case PAGE_NOACCESS:
        std::cout << "Protect: No Access\n";
        break;
    case PAGE_READONLY:
        std::cout << "Protect: Read Only\n";
        break;
    case PAGE_READWRITE:
        std::cout << "Protect: Read Write\n";
        break;
    case PAGE_WRITECOPY:
        std::cout << "Protect: Write Copy\n";
        break;
    default:
        break;
    }
    // std::cout << "Type: ";
    switch (mbi.Type)
    {
    case MEM_IMAGE:
        std::cout << "Type: Image\n";
        break;
    case MEM_MAPPED:
        std::cout << "Type: Mapped\n";
        break;
    case MEM_PRIVATE:
        std::cout << "Type: Private\n";
        break;
    default:
        break;
    }
    HMODULE hMod = (HMODULE)p;
    TCHAR szModName[MAX_PATH];
    GetModuleFileName(hMod, szModName, sizeof(szModName) / sizeof(TCHAR));
    std::cout << "Module Name: " << szModName << std::endl;

    std::cout << "\n";
    return mbi.RegionSize;
}
int main()
{
    // 1.获取系统当前内存设置信息。
    SYSTEM_INFO sysInfo;
    // 系统信息
    GetSystemInfo(&sysInfo);
    std::cout << "System Memory Information:\n";
    std::cout << "Page Size: " << float(sysInfo.dwPageSize) / DIV << " KB\n";
    std::cout << "Minimum Application Address: " << sysInfo.lpMinimumApplicationAddress << "\n";
    std::cout << "Maximum Application Address: " << sysInfo.lpMaximumApplicationAddress << "\n";
    std::cout << "Allocation Granularity: " << sysInfo.dwAllocationGranularity / DIV << " KB\n\n";

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

    // 2.遍历当前进程地址空间(虚拟内存)，显示每个虚拟内存区域的特性
    
    std::cout << "----------------------------------------\n";
    std::cout << "Virtual Memory Information:\n";
    TCHAR fileName[MAX_PATH];
    GetModuleFileName(NULL, fileName, MAX_PATH);
    printf("Executable File: %s\n\n", fileName);

    SYSTEM_INFO si;
    GetSystemInfo(&si);
    char *p = (char *)si.lpMinimumApplicationAddress;
    while (p < (char *)si.lpMaximumApplicationAddress)
    {
        SIZE_T RegionSize = PrintRegion(p);
        p += RegionSize;
    }
    std::cout << std::endl;

    // 3.在当前进程地址空间，分配 1GB 虚拟内存，对其中 1MB 清0。
    std::cout << "----------------------------------------\n";
    const int GB = DIV * DIV * DIV;
    const int MB = DIV * DIV;
    p = (char *)si.lpMinimumApplicationAddress;
    char *buffer = NULL;
    while (p < (char *)si.lpMaximumApplicationAddress)
    {
        MEMORY_BASIC_INFORMATION mbi;
        VirtualQuery(p, &mbi, sizeof(mbi));
        if (mbi.RegionSize > 1 * GB && mbi.State == MEM_FREE)
        {
            // buffer = (char *)VirtualAlloc((LPVOID)p, 1 * GB, MEM_RESERVE, PAGE_READWRITE);
            buffer = (char *)VirtualAlloc((LPVOID)p, 1 * GB, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            if (buffer == NULL)
            {
                std::cerr << "VirtualAlloc failed\n";
                return 1;
            }
            else
            {
                std::cout << "1GB of virtual memory allocated， Region information:\n\n";
                SIZE_T RegionSize = PrintRegion(p);
                break;
            }
        }
        p += mbi.RegionSize;
    }

    MEMORYSTATUSEX memoryStatus1;
    memoryStatus1.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memoryStatus1))
    {
        std::cout << "Added Virtual Page File: " << float(memoryStatus.ullAvailPageFile - memoryStatus1.ullAvailPageFile) / DIV / DIV / DIV << " GB" << std::endl
                  << std::endl;
    }
    else
    {
        std::cerr << "Failed to retrieve memory status." << std::endl;
    }
    memset(buffer, 0, MB);
    std::cout << "1MB cleared to 0\n";
    VirtualFree(buffer, 0, MEM_RELEASE);

    return 0;
}
