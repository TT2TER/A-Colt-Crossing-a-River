`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/12/06 00:42:42
// Design Name: 
// Module Name: Reg_File
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
// 寄存器堆的抽象，使用 verilog 中内置的寄存器数组模拟存储器，一共 32 个寄存器。
//////////////////////////////////////////////////////////////////////////////////

`include "macro.vh"
module Reg_File(
    input clk,  //时钟和
    input rst,  //复位信号
    input we,           //写使能,只有在 we 值 1 且 clk 上升沿时，才进行数据写入
    input [4:0] RA1,    //读地址1 (寄存器数组的下标)
    input [4:0] RA2,    //读地址2
    input [4:0] WA,     //写地址
    input [31:0] WD,    //写数据 (32 位),待写入的数值

    output [31:0] RD1,  //读出的数据1
    output [31:0] RD2,  //读出的数据2
    output [31:0] debug_reg1,
    output [31:0] debug_reg2,
    output [31:0] debug_reg3
    );

reg [31:0] regFiles [0:31];     //寄存器堆，索引从 0 到 31

always @(posedge clk) begin //时钟上升沿写数据
    if(rst==0 && we && WA>0 ) begin   //写使能置1且非0号寄存器 **注意** 0号寄存器不可写，x0恒为0，不可修改。
        regFiles[WA] <= WD; //写数据
    end
end

// 读寄存器数据
assign RD1=(RA1==0)? `DEFAULT_VAL: regFiles[RA1]; //如果读地址为0，则输出默认值，否则输出寄存器堆中的值
assign RD2=(RA2==0)? `DEFAULT_VAL: regFiles[RA2]; //同上
assign debug_reg1=regFiles[1]; 
assign debug_reg2=regFiles[2];
assign debug_reg3=regFiles[3];
endmodule
