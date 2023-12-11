`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/12/08 21:40:11
// Design Name: 
// Module Name: Reg_Mux
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
// 本项目中
// 写回寄存器的值有 3 种不同的可能: alu 的运算的结果 op_res、数据存储器读出的值 d_mem, pc+4 的值 pc
//////////////////////////////////////////////////////////////////////////////////


module Reg_Mux(
    //register multiplexers 多路选择器
    //根据控制信号 reg_src 从输入op_res、dmem、pc中选择一个输出
    input [1:0] reg_src,
    input [31:0] op_res,
    input [31:0] dmem,
    input [31:0] pc,
    output reg [31:0] res
);

always @(*) begin
    case (reg_src)
        0: res = op_res;
        1: res = dmem;
        2: res = pc; //对应npc模块中的pc+4
    endcase
end
endmodule
