//生产者和消费者实验主程序
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>

#define TIME_PRODUCER 6
#define TIME_CONSUMER 4

struct buffer
{//定义缓冲区
    int s[3];//要求是三个共享区
    int head;
    int tail;
    int is_empty;
};

struct sharedmemory
{//定义共享内存
    struct buffer data;//缓冲区
    HANDLE full;//有数据的缓冲区个数，初值为0
    HANDLE empty;//表示空缓冲区的个数，初值为k
    HANDLE mutex;//互斥访问临界区的信号量，初值为1
} ;

HANDLE MakeShared()
{ //创建共享内存，由filemapping实现
    //创建一个临时文件映射对象
    HANDLE hMapping = CreateFileMapping(INVALID_HANDLE_VALUE,
                                        NULL, PAGE_READWRITE, 0, sizeof(struct sharedmemory), "BUFFER");
    if (hMapping == NULL)
    {//映射对象无退出程序
        printf("CreateFileMapping error\n");
        exit(0);
    }
    //在文件映射上创建视图，返回起始虚地址
    LPVOID pData = MapViewOfFile(hMapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (pData == NULL)
    {
        printf("MapViewOfFile error\n");
        exit(0);
    }
    if (pData != NULL)
    {
        ZeroMemory(pData, sizeof(struct sharedmemory));
    }
    //解除当前地址空间映射
    UnmapViewOfFile(pData);
    return (hMapping);
}

int main()
{
    HANDLE hMapping = MakeShared();

    //打开文件映射
    HANDLE hFileMapping = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, "BUFFER");
    if (hFileMapping == NULL)
    {
        printf("OpenFileMapping error\n");
        exit(0);
    }

    LPVOID pFile = MapViewOfFile(hFileMapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (pFile == NULL)
    {
        printf("MapViewOfFile error\n");
        exit(0);
    }

    // 创建共享内存
    struct sharedmemory *addr = (struct sharedmemory *) (pFile);
    addr->data.head = 0;
    addr->data.tail = 0;
    addr->data.is_empty = 1;

    HANDLE empty = CreateSemaphore(NULL, 3, 3, "EMPTY");
    HANDLE full = CreateSemaphore(NULL, 0, 3, "FULL");
    HANDLE mutex = CreateMutex(NULL, FALSE, "MUTEX");

    UnmapViewOfFile(pFile);//停止当前程序的一个内存映射
    pFile = NULL;
    CloseHandle(hFileMapping);//关闭现有已打开句柄

    //创建子进程
    PROCESS_INFORMATION sub[5];

    for (int i = 0; i < 2; i++)
    {//生产者
        printf("Produce %d created.\n", i + 1);
        TCHAR szFilename[MAX_PATH];
        TCHAR szCmdLine[MAX_PATH];
        
        sprintf(szFilename, "./p.exe");
        sprintf(szCmdLine, "\"%s\"", szFilename);
        PROCESS_INFORMATION pi;
        STARTUPINFO si;
        ZeroMemory(&si, sizeof(STARTUPINFO));
        si.cb = sizeof(si);
        //创建子进程
        BOOL bCreatOK = CreateProcess(szFilename, NULL, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi);
        sub[i] = pi;
    }
    //消费者
    for (int i = 2; i < 5; i++)
    {
        printf("Consume %d created.\n", i - 1);
        TCHAR szFilename[MAX_PATH];
        TCHAR szCmdLine[MAX_PATH];
        PROCESS_INFORMATION pi;
        sprintf(szFilename, "./c.exe");
        sprintf(szCmdLine, "\"%s\"", szFilename);

        STARTUPINFO si;
        ZeroMemory(&si, sizeof(STARTUPINFO));
        si.cb = sizeof(si);
        //创建子进程
        BOOL bCreatOK = CreateProcess(szFilename, NULL, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi);
        sub[i] = pi;
    }
    //等待子进程结束
    for (int i = 0; i < 5; i++)
    {
        WaitForSingleObject(sub[i].hProcess, INFINITE);
    }
    //关闭子进程句柄
    for (int i = 0; i < 5; i++)
    {
        CloseHandle(sub[i].hProcess);
    }

    CloseHandle(hMapping);
    hMapping = INVALID_HANDLE_VALUE;
    return 0;
}
