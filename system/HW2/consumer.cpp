#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>

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
    HANDLE full = OpenSemaphore(SEMAPHORE_ALL_ACCESS, FALSE, "FULL");
    HANDLE empty = OpenSemaphore(SEMAPHORE_ALL_ACCESS, FALSE, "EMPTY");
    HANDLE mutex = OpenMutex(SEMAPHORE_ALL_ACCESS, FALSE, "MUTEX");
    for (int i = 0; i < TIME_CONSUMER; i++)
    {
        srand(GetCurrentProcessId() + i);
        Sleep(rand() % 1000);
        WaitForSingleObject(full, INFINITE);  //P(full) 申请一个产品
        WaitForSingleObject(mutex, INFINITE); //P(mutex) 申请进入缓冲区

        int num = addr->data.s[addr->data.head];
        addr->data.head = (addr->data.head + 1) % 3;
        if (addr->data.head == addr->data.tail)
            addr->data.is_empty = 1;
        else
            addr->data.is_empty = 0;
        SYSTEMTIME time;
        GetLocalTime(&time);
        printf("\nTime: %02d:%02d:%02d:%d\n", time.wHour, time.wMinute, time.wSecond, time.wMilliseconds);
        printf("Consumer %d removing %d\n", GetCurrentProcessId(), num);

        if (addr->data.is_empty)
            printf("Empty\n");
        else
            CurrentStatus(addr);

        ReleaseSemaphore(empty, 1, NULL); //V(empty) 释放一个空缓冲
        ReleaseMutex(mutex);//V(mutex) 退出缓冲区
    }
    UnmapViewOfFile(pFile);
    pFile = NULL;
    CloseHandle(hMap);
    return 0;
}
