`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/12/05 22:02:31
// Design Name: 
// Module Name: Ins_Mem
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 指令存储器Instruction Memory
// 指令存储器是计算机中的一个关键组件，它存储了计算机需要执行的所有指令
//////////////////////////////////////////////////////////////////////////////////

`include "macro.vh"
module Ins_Mem(
    input [11:2] addr, // 指令存储器由32位寄存器数组表示,直接取到数组的下标；
    //指令地址也是按 4 字节对齐的。这意味着指令地址的最低 2 位总是 0，因此在实际设计中，我们可以忽略这两位，只需要用剩下的位来索引指令存储器
    output [31:0] inst
    );

reg[31:0] imem[`IM_SIZE:0];//IM_SIZE=1023, 定义一个寄存器数组，用于模拟指令存储器
//reg[31:0] 表示每个数组元素都是一个 32 位的寄存器，可以存储一个 32 位的数据，这对应于 RISC-V 架构中的指令长度
//imem[IM_SIZE:0] 定义了数组的大小
assign inst=imem[addr];   //从指令存储器中的地址 addr 读取一个指令，并将这个指令赋值给 inst

endmodule
