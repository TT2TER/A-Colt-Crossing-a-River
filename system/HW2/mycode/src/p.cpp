#include <iostream>
#include <windows.h>
#include <time.h>
#include "mayu.h"

#define TIMES_OF_PRODUCER 8
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

const char *GetRandomFoodNameString()
{
    int randomIndex = std::rand() % 30;

    const char *foodName;

    switch (randomIndex)
    {
    case 0:
        foodName = "Pizza";
        break;
    case 1:
        foodName = "Burger";
        break;
    case 2:
        foodName = "Pasta";
        break;
    case 3:
        foodName = "Sushi";
        break;
    case 4:
        foodName = "Salad";
        break;
    case 5:
        foodName = "Steak";
        break;
    case 6:
        foodName = "Soup";
        break;
    case 7:
        foodName = "Taco";
        break;
    case 8:
        foodName = "Curry";
        break;
    case 9:
        foodName = "Rice";
        break;
    case 10:
        foodName = "Pancake";
        break;
    case 11:
        foodName = "Donut";
        break;
    case 12:
        foodName = "Sandwich";
        break;
    case 13:
        foodName = "Noodle";
        break;
    case 14:
        foodName = "Cheese";
        break;
    case 15:
        foodName = "Fish";
        break;
    case 16:
        foodName = "Chicken";
        break;
    case 17:
        foodName = "Egg";
        break;
    case 18:
        foodName = "Bacon";
        break;
    case 19:
        foodName = "Potato";
        break;
    case 20:
        foodName = "Tomato";
        break;
    case 21:
        foodName = "Cucumber";
        break;
    case 22:
        foodName = "Carrot";
        break;
    case 23:
        foodName = "Onion";
        break;
    case 24:
        foodName = "Pineapple";
        break;
    case 25:
        foodName = "Orange";
        break;
    case 26:
        foodName = "Banana";
        break;
    case 27:
        foodName = "Grapes";
        break;
    case 28:
        foodName = "Apple";
        break;
    case 29:
        foodName = "Chocolate";
        break;
    default:
        foodName = "Unknown";
        break;
    }

    return foodName;
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

    for (int i = 0; i < TIMES_OF_PRODUCER; i++)
    {
        // 随机睡很久
        srand(GetCurrentProcessId() + i);
        Sleep(rand() % 1000);
        P(empty, INFINITE);
        P(mutex, INFINITE);
        strcpy(addr->s[addr->tail], (char *)GetRandomFoodNameString());
        SYSTEMTIME time;
        GetLocalTime(&time);
        printf("\nTime: %02d:%02d:%02d:%d\n", time.wHour, time.wMinute, time.wSecond, time.wMilliseconds);
        printf("Producer %d putting %s\n", programNumber, addr->s[addr->tail]);
        addr->tail = (addr->tail + 1) % BUFFER_SIZE;
        addr->is_empty = 0;

        PrintBufferContents(addr);
        Vm(mutex);
        Vs(full, 1, NULL);
    }

    UnmapViewOfFile(pFile);
    pFile = NULL;
    CloseHandle(hMapping);
    CloseHandle(mutex);
    CloseHandle(empty);
    CloseHandle(full);
    return 0;
}