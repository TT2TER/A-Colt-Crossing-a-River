#include <iostream>
#include <windows.h>
#include <time.h>
#include "mayu.h"

#define TIMES_OF_CONSUMER 8
#define P WaitForSingleObject
#define Vs ReleaseSemaphore
#define Vm ReleaseMutex

void PrintBufferContents(SHM *pSHM)
{

    if (pSHM->is_empty)
    {
        printf("Buffer is empty\n");
    }
    else
    {
        printf("Buffer Contents: ");
        for (int i = pSHM->head;;)
        {
            printf("%s  ", pSHM->s[i]);
            i = (i + 1) % BUFFER_SIZE;
            if (i == pSHM->tail)
            {
                printf("\n");
                return;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <ProgramNumber>" << std::endl;
        return 1;
    }

    int programNumber = std::atoi(argv[1]);
    
    HANDLE hMapping = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, "BUFFER");
    if (hMapping == NULL)
    {
        printf("OpenFileMapping error!\n");
        exit(0);
    }
    LPVOID pFile = MapViewOfFile(hMapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (pFile == NULL)
    {
        printf("MapViewOfFile error!\n");
        exit(0);
    }
    SHM *addr = (SHM *)pFile;

    HANDLE full = OpenSemaphore(SEMAPHORE_ALL_ACCESS, FALSE, "FULL");
    HANDLE empty = OpenSemaphore(SEMAPHORE_ALL_ACCESS, FALSE, "EMPTY");
    HANDLE mutex = OpenMutex(SEMAPHORE_ALL_ACCESS, FALSE, "MUTEX");

    for (int i = 0; i < TIMES_OF_CONSUMER; i++)
    {
        // 随机睡很久
        srand(GetCurrentProcessId() + i);
        Sleep(rand() % 2000);
        P(full, INFINITE);
        P(mutex, INFINITE);

        SYSTEMTIME time;
        GetLocalTime(&time);
        printf("\nTime: %02d:%02d:%02d:%d\n", time.wHour, time.wMinute, time.wSecond, time.wMilliseconds);
        printf("Consumer %d removing %s\n", programNumber, addr->s[addr->head]);

        addr->head = (addr->head + 1) % BUFFER_SIZE;
        if (addr->head == addr->tail)
        {
            addr->is_empty = 1;
        }
        else
        {
            addr->is_empty = 0;
        }
        PrintBufferContents(addr);
        Vm(mutex);
        Vs(empty, 1, NULL);
    }

    UnmapViewOfFile(pFile);
    pFile = NULL;
    CloseHandle(hMapping);
    CloseHandle(mutex);
    CloseHandle(empty);
    CloseHandle(full);
    return 0;
}
