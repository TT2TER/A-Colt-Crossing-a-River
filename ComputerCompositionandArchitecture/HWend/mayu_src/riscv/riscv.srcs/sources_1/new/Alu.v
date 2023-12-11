`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/12/06 00:41:20
// Design Name: 
// Module Name: Alu
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

module Alu(
    // 算数逻辑单元 Arithmetic Logic Unit 执行算数和逻辑运算
    input [1:0] op, 
    input [31:0] in1,
    input [31:0] in2,
    output ZF, //zero flag
    output SF, //sign flag
    output reg [31:0] res
    );
    assign ZF=(res==0)?1:0;
    assign SF=res[31];//最高位为符号位

    always @(*) begin
        case (op)
            0: res=in1+in2;
            1: res=in1-in2;
            2: res=in1 | in2;//按位或
        endcase        
    end
endmodule
