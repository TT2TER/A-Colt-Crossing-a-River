`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/12/06 00:53:39
// Design Name: 
// Module Name: Alu_Mux
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
// alu 的 2 个输入值中，第二个输入值根据指令类型不同，可以取第二个寄存
// 器或是指令中立即数的值，本模块根据控制信号 alu_src 完成这一选择功能。
//////////////////////////////////////////////////////////////////////////////////


module Alu_Mux(
    //alu multiplexers 多路选择器
    //根据控制信号 alu_src 从寄存器堆或是立即数选择一个输出
    input alu_src,
    input [31:0] RD2,
    input [31:0] imm,
    output reg [31:0] res
    );

always @(*) begin
    case (alu_src)
        0: res=RD2;
        1: res=imm;
    endcase
end
endmodule
