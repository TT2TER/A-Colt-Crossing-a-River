#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>

#define TIME_PRODUCER 6

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
};


//显示缓冲区数据
void CurrentStatus(struct sharedmemory *a)
{
    printf("Current Data: ");
    for (int i = a->data.head;;)
    {
        printf("%d ", a->data.s[i]);
        i++;
        i %= 3;
        if (i == a->data.tail)
        {
            printf("\n");
            return;
        }
    }
}

int main()
{
    HANDLE hMap = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, "BUFFER");
    if (hMap == NULL)
    {
        printf("OpenFileMapping error!\n");
        exit(0);
    }
    LPVOID pFile = MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (pFile == NULL)
    {
        printf("MapViewOfFile error!\n");
        exit(0);
    }
    struct sharedmemory *addr = (struct sharedmemory *) (pFile);
    HANDLE full = OpenSemaphore(SEMAPHORE_ALL_ACCESS, FALSE, "FULL"); // 为现有的一个已命名信号机对象创建一个新句柄
    HANDLE empty = OpenSemaphore(SEMAPHORE_ALL_ACCESS, FALSE, "EMPTY");
    HANDLE mutex = OpenMutex(SEMAPHORE_ALL_ACCESS, FALSE, "MUTEX"); // 为现有的一个已命名互斥体对象创建一个新句柄。
    for (int i = 0; i < TIME_PRODUCER; i++)
    {
        srand(GetCurrentProcessId() + i);
        Sleep(rand() % 1000);
        WaitForSingleObject(empty, INFINITE); //P(empty) 申请空缓冲
        WaitForSingleObject(mutex, INFINITE); //P(mutex) 申请进入缓冲区
        //向缓冲区添加数据
        int num = rand() % 1000;
        addr->data.s[addr->data.tail] = num;
        addr->data.tail = (addr->data.tail + 1) % 3;
        addr->data.is_empty = 0;
        SYSTEMTIME time;
        GetLocalTime(&time);
        printf("\nTime: %02d:%02d:%02d:%d\n", time.wHour, time.wMinute, time.wSecond, time.wMilliseconds);
        printf("Producer %d putting %d\n", GetCurrentProcessId(), num);

        if (addr->data.is_empty)
            printf("Empty\n");
        else
            CurrentStatus(addr);

        ReleaseSemaphore(full, 1, NULL); //V(full) 释放一个产品
        ReleaseMutex(mutex);             //V(mutex) 退出缓冲区
    }
    UnmapViewOfFile(pFile); // 停止当前程序的一个内存映射
    pFile = NULL;
    CloseHandle(hMap); // 关闭句柄
    CloseHandle(mutex);
    CloseHandle(empty);
    CloseHandle(full);
    return 0;
}
