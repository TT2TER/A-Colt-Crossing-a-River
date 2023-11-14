// 生产者和消费者实验主程序
#include <iostream>
#include <windows.h>
#include "mayu.h"
#define NUM_PRODUCER 4
#define NUM_CONSUMER 4

int main()
{
    ShareBuffer shareBuffer;
    shareBuffer.CreateSharedMemory();
    shareBuffer.InitializeSharedMemory();
    SubProcessManager subProcess1(NUM_CONSUMER, NUM_PRODUCER);
    subProcess1.CreateSubProcess();
    subProcess1.CloseSubProcess();
    return 0;
}