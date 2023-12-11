`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/12/05 20:29:04
// Design Name: 
// Module Name: pc
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
// 
//////////////////////////////////////////////////////////////////////////////////

`include "macro.vh"
module pc(
    input clk,
    input  rst, //复位信号
    input [31:0] npc, // 下一个指令地址
    output reg [31:0] pc  //输出的pc值
    );

    always @(posedge clk, posedge rst) begin
        if (rst) 
            pc <= `DEFAULT_VAL; //如果复位信号置1，则pc设为0
        else 
            pc <= npc;  // 否则正常取下一个指令地址
    end
endmodule
