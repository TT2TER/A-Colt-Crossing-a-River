#ifndef COLOR_H
#define COLOR_H

#include <iostream>
#include <windows.h>
#include <time.h>
#include <vector>

#define BUFFER_SIZE 6
typedef char Bffer[10];
typedef struct sharedmemory
{
    Bffer s[BUFFER_SIZE];
    int head;
    int tail;
    int is_empty;
} SHM;

class ShareBuffer
{
public:
    ShareBuffer()
        : hMapping(NULL), pData(NULL)
    {
    }

    ~ShareBuffer()
    {
        UnmapViewOfData();

        if (hMapping != NULL)
        {
            CloseHandle(hMapping);
        }
    }

    void CreateSharedMemory()
    {
        // 创建共享内存
        hMapping = CreateFileMapping(
            INVALID_HANDLE_VALUE,
            NULL,
            PAGE_READWRITE,
            0,
            sizeof(SHM),
            "BUFFER");

        if (hMapping == NULL)
        {
            printf("CreateFileMapping error\n");
        }
    }

    bool MapViewOfData()
    {
        // 打开共享内存
        if (pData != NULL)
        {
            return true;
        }
        pData = MapViewOfFile(hMapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);

        if (pData == NULL)
        {
            printf("MapViewOfFile error\n");
            return false;
        }
        return true;
    }

    void InitializeSharedMemory()
    {
        if (MapViewOfData())
        {
            ZeroMemory(pData, sizeof(SHM));
            SHM *pSHM = (SHM *)pData;
            pSHM->head = 0;
            pSHM->tail = 0;
            pSHM->is_empty = 1;

            HANDLE empty = CreateSemaphore(NULL, 6, 6, "EMPTY");
            HANDLE full = CreateSemaphore(NULL, 0, 6, "FULL");
            HANDLE mutex = CreateMutex(NULL, FALSE, "MUTEX");
            printf("InitializedSharedMemory\n");
        }
    }

    void UnmapViewOfData()
    {
        // 提前结束视图
        if (pData != NULL)
        {
            UnmapViewOfFile(pData);
            pData = NULL; // 将 pData 设置为 NULL，以避免重复调用 UnmapViewOfFile
        }
    }

private:
    HANDLE hMapping;
    LPVOID pData;
};

class SubProcessManager
{
    // 用来创建和关闭子进程
public:
    SubProcessManager(int numProducers, int numConsumers) : NUM_PRODUCER(numProducers), NUM_CONSUMER(numConsumers)
    {
        subProcess.resize(NUM_PRODUCER + NUM_CONSUMER);
    }

    void CreateSubProcess()
    {
        for (int i = 0; i < NUM_PRODUCER; i++)
        {
            TCHAR FileName[MAX_PATH];
            TCHAR Cmd[MAX_PATH];
            sprintf(FileName, "./p.exe");
            char arg[16];          // 预留足够的空间来存储整数转换后的字符串
            sprintf(arg, "%d", i); // 将整数转换为字符串

            sprintf(Cmd, "./p.exe %s", arg); // 拼接字符串参数
            STARTUPINFO si;
            ZeroMemory(&si, sizeof(si));
            si.cb = sizeof(si);
            if (CreateProcess(FileName, Cmd, NULL, NULL, FALSE, 0, NULL, NULL, &si, &subProcess[i]))
            {
                printf("Producer %d created.\n", i + 1);
            }
        }
        for (int i = NUM_PRODUCER; i < NUM_PRODUCER + NUM_CONSUMER; i++)
        {
            TCHAR FileName[MAX_PATH];
            TCHAR Cmd[MAX_PATH];
            char arg[16];                         // 预留足够的空间来存储整数转换后的字符串
            sprintf(FileName, "./c.exe");
            sprintf(arg, "%d", i - NUM_PRODUCER); // 将整数转换为字符串

            sprintf(Cmd, "./c.exe %s", arg); // 拼接字符串参数s

            STARTUPINFO si;
            ZeroMemory(&si, sizeof(si));
            si.cb = sizeof(si);
            if (CreateProcess(FileName, Cmd, NULL, NULL, FALSE, 0, NULL, NULL, &si, &subProcess[i]))
            {
                printf("Consumer %d created.\n", i + 1);
            }
        }
    }

    void CloseSubProcess()
    {
        for (int i = 0; i < NUM_PRODUCER + NUM_CONSUMER; i++)
        {
            WaitForSingleObject(subProcess[i].hProcess, INFINITE);
        }

        for (int i = 0; i < NUM_PRODUCER + NUM_CONSUMER; i++)
        {
            CloseHandle(subProcess[i].hProcess);
        }
    }

private:
    const int NUM_PRODUCER;
    const int NUM_CONSUMER;
    std::vector<PROCESS_INFORMATION> subProcess;
};
#endif