`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/12/06 00:39:24
// Design Name: 
// Module Name: Data_Mem
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
// 数据存储器
//////////////////////////////////////////////////////////////////////////////////

`include "macro.vh"
module Data_Mem(
    input clk,
    input rst,
    input we, //写使能信号，信号为真时，WD 写入到 dmem[addr] 中
    input [11:2] addr, //地址
    input [31:0] WD,   //写入的数据
    output [31:0] RD,   //读出的数据
    //debug
    output [31:0] data1,
    output [31:0] data2,
    output [31:0] data3,
    output [31:0] data4,
    output [31:0] data5
    );
// 存储器
reg[31:0] dmem[`DM_SIZE:0]; //DM_SIZE=1023, 定义一个寄存器数组，用于模拟数据存储器

assign RD=dmem[addr];  //从数据存储器中的地址 addr 读取一个数据，并将这个数据赋值给 RD

// debug
assign data1=dmem[0];
assign data2=dmem[1];
assign data3=dmem[2];
assign data4=dmem[3];
assign data5=dmem[4];

always @(posedge clk) begin
    if (rst==0 && we ) begin //写使能信号为真时，WD 写入到 dmem[addr] 中
        dmem[addr] <= WD;
    end
end

endmodule
