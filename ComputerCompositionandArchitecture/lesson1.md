# 体系结构

## 指令集体系结构（ISA）
- 指令系统
- 机器代码汇编语言

## 微体系结构

# 
## 存储程序（Stored Program）
五大组成部分：运算器、控制器、存储器、输入设备、输出设备

内部采用二进制表示数据和指令

将变成好的程序和数据存储在计算机的存储器中，计算机按照程序的顺序执行指令，从而实现对数据的处理


## 计算机系统组成
**硬件和软件在逻辑功能上是等效的**
固件是具有软件功能的硬件

主机包括cpu（运算器+控制器）和主存储器
### 运算器

### 控制器

### 存储器
#### 主存储器（内存）
#### 辅助存储器（外设）

### 输入设备

### 输出设备

### 系统总线
#### 数据总线
#### 地址总线
#### 控制总线

## 计算机编程语言与软件
1. 机器语言编写程序，记录在纸带和卡片上
1. 用汇编语言开发程序（操作系统移植，算法优化）
    - 助记符表示操作码
    - 用标号表示位置
    - 汇编程序将汇编语言转换为机器语言，与机器指令一一对应
1. 高级语言编写程序
    - 面向算法
    - 一条高级语言语句对应多条机器语言指令
    - 有面向过程和面向对象两种
    - 处理逻辑分为三种结构：顺序结构、选择结构、循环结构
    - 有两种转换方式：编译和解释
        - 编译：将高级语言源程序一次性转换为机器语言目标程序，再执行目标程序
        - 解释：将高级语言源程序**逐条**翻译为机器语言并执行，不生成目标文件

## 计算机的软件
### 系统软件 
简化编程过程、使得计算机易于使用、有效利用计算机资源
- 操作系统：用户接口、资源管理、作业管理、文件管理、存储管理、设备管理、网络管理
- 语言处理系统  语言实现两种技术：翻译和解释
    - 翻译程序（translator）有两类：
        - 汇编程序（assembler）：将汇编语言源程序翻译成机器语言目标程序
        - 编译程序（compiler）：将高级语言源程序翻译成机器语言目标程序

## 系列机和软件兼容
系列机载指令系统、数据格式、字符编码、中断系统、控制方式、输入输出方式保持统一

### 系列机
同一个厂家生产，具有相同体系结构
### 兼容机
不同厂家生产，具有相同体系结构
### 软件兼容
向上兼容
向后兼容：新机器只能加指令

## 计算机的工作过程
1. 程序的执行过程
    1. 从外存中读取程序
    1. 将程序和数据送入主存
    1. 从主存中取出指令和数据送入CPU
    1. CPU执行指令
    1. 将结果送入主存
    1. 将结果送入外存

## 计算机主要性能指标
1. 机器字长
指参与运算的二进制数的位数，与加法器寄存器数据总线位数有关
字节（**B**yte）：8位(**b**it)
字（word）：机器字长,与系列机有关
2. 数据通路宽度（外部数据总线宽度）
数据总线一次能并行传送的二进制位数
3. 主存容量
一个主存储器能存储的全部信息量

容量单位有两种：
- 字节数（字节编址计算机）
 $2^{10}=1K; 2^{20}=1M; 2^{30}=1G; 2^{40}=1T$
- 字数*字长（字编址计算机）

4. 运算速度
- 吞吐量和响应时间：响应时间包括CPU执行时间和等待时间（I/O等）
- 主频和时钟周期：CPU主频是时钟频率（CPU内数字脉冲信号震荡速度），主频倒数是时钟周期；每个动作至少需要一个时钟周期
- 机器周期：CPU执行一条指令所需要的时间
- CPI（Cycles Per Instruction）：每条指令平均需要的时钟周期数  $CPI=\frac{CPU时钟周期数}{指令条数(IC)}$
- 并行常用IPC（Instructions Per Cycle）每个时钟周期执行指令数：$IPC=\frac{1}{CPI}$
- 假设计算机系统有$n$种指令，第i种指令的处理时间为$CPI_i$,第i种指令出现的次数为$IC_i$，则CPU执行时间为：$CPU执行时间=\sum_{i=1}^n{CPI_i*IC_i}$
- CPU性能公式：$CPU执行时间=\frac{CPU时钟周期数}{时钟频率}=\frac{指令数*CPI}{时钟频率}=指令数*CPI*时钟周期$
- MIPS（Million Instructions Per Second）：每秒执行百万条指令数 $MIPS=\frac{指令条数}{执行时间*10^6}=\frac{时钟频率}{CPI*10^6}$
- MFLOPS（Million Floating Point Operations Per Second）：每秒执行百万次浮点运算数 $MFLOPS=\frac{浮点运算次数}{执行时间*10^6}$


