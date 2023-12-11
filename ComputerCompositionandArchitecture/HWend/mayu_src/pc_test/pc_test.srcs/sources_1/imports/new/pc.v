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
    input  rst, //��λ�ź�
    input [31:0] npc, // ��һ��ָ���ַ
    output reg [31:0] pc  //�����pcֵ
    );

    always @(posedge clk, posedge rst) begin
        if (rst) 
            pc <= `DEFAULT_VAL; //�����λ�ź���1����pc��Ϊ0
        else 
            pc <= npc;  // ��������ȡ��һ��ָ���ַ
    end
endmodule
