`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/12/06 00:54:50
// Design Name: 
// Module Name: Imm_Gen
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
// 由于在RISCV中，指令的立即数是在指令中的，所以需要对指令进行解析，提取出立即数
// 指令中分布位置不同 
// 并且拓展为32位   
//////////////////////////////////////////////////////////////////////////////////

`include "macro.vh"
module Imm_Gen(

    input [31:0] inst, //32 位，指令值
    output reg [31:0] imm //32 位，立即数，提取并且拓展位数的立即数值
    );
//指令结构见链接：https://nju-projectn.github.io/dlco-lecture-note/_images/riscvisa.png
always@(*) begin //always@(*)表示任何一个输入信号变化时，都会执行一次
    case(inst[6:0]) //inst[6:0]表示指令的6到0位，即指令的操作码
       `OPCODE_I1,          // addi,ori
       `OPCODE_I2: begin    // lw
        // 12位立即数符号拓展
           if(inst[31] == 1) //如果立即数的最高位为1，则拓展为全1
               imm = {20'hfffff,inst[31:20]}; //立即数是指令的20到31位
           else imm =  {20'h0,inst[31:20]};
       end

       `OPCODE_S: begin     //sw
           if(inst[31] == 1) //如果立即数的最高位为1，则拓展为全1
               imm = {20'hfffff,inst[31:25],inst[11:7]};//立即数是指令的25到31位和7到11位
           else imm = {20'h0,inst[31:25],inst[11:7]}; //否则拓展为全0
       end

       `OPCODE_B: begin     //beq,blt
           if(inst[31] == 1) imm[31:13] = 19'h7ffff;
           else imm[31:13] = 19'h0;
           imm[12:0] = {inst[31],inst[7],inst[30:25],inst[11:8],1'b0};
       end
       
       `OPCODE_J: begin     //jal
           if(inst[31] == 1) 
                imm = {11'h7ff,inst[31],inst[19:12],inst[20],inst[30:21],1'h0};
           else imm = {11'h0,inst[31],inst[19:12],inst[20],inst[30:21],1'h0};
       end      
    endcase
end      
endmodule
