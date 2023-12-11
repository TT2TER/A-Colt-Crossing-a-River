`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/12/05 20:59:30
// Design Name: 
// Module Name: control_unit
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
// 控制单元，生成各种控制信号
//
//////////////////////////////////////////////////////////////////////////////////

`include "macro.vh"
module control_unit(
    input [6:0] opcode,//指令的操作码
    input [6:0] func7,//指令的功能码
    input [2:0] func3,//指令的功能码

    //跳转分支指令的选择信号,0表示不跳转,1表示jal,2表示beq,3表示blt
            //jal功能：将pc+4写入寄存器rd，将pc+imm写入pc
            //beq(相等时分支）功能：如果rs1==rs2，将pc+imm写入pc
            //blt（小于时分支）功能：如果rs1<rs2，将pc+imm写入pc
    output reg [1:0] branch,   
    // ALU第二个操作数的选择信号,0表示第二个寄存器，1表示立即数
    output reg alu_src,         
    // 寄存器写回数据的选择信号, 0表示alu结果，1表示数据存储器，2表示 pc+4
    output reg [1:0] reg_src,   
    // 运算符的选择信号,0为+，1为-,2为|
    output reg [1:0] alu_op,  
    // 存储器写使能
    output reg dmem_we,        
    // 寄存器堆写使能
    output reg reg_we           
    );

// 进行指令译码，并输出控制信号
always@(*) begin //always@(*)表示任何一个输入信号变化时，都会执行一次
    case(opcode) //case语句，类似于switch语句,根据opcode的值，选择不同的case
       `OPCODE_R: begin //add ,sub
            if(func7 == `FUNC7_ADD) alu_op = 0; //alu_op表示运算符的选择信号
            else if(func7 == `FUNC7_SUB) alu_op = 1; //0为+，1为-
            branch = 0; //0表示不跳转
            reg_src = 0; //寄存器写回数据选择Alu结果
            alu_src = 0; //alu_src 0表示第二个寄存器
            reg_we = 1; //reg_we表示寄存器堆写使能
            dmem_we = 0; //dmem_we表示禁止数据存储器写使能
        end 
        `OPCODE_I1: begin //addi,ori
            if (func3==`FUNC3_ADDI) alu_op = 0;
            else if(func3==`FUNC3_ORI) alu_op=2;
            branch=0;
            reg_src=0;
            alu_src = 1;
            reg_we = 1;
            dmem_we = 0;
       end
       `OPCODE_I2: begin// lw
            alu_op = 0;
            branch=0;
            reg_src = 1;
            alu_src = 1;
            reg_we = 1;
            dmem_we = 0;
        end
       `OPCODE_S: begin// sw
            alu_op = 0;
            branch=0;
            reg_src = 0;
            alu_src = 1;
            reg_we = 0;
            dmem_we = 1;
        end
       `OPCODE_B: begin
           if(func3 == `FUNC3_BEQ) begin// beq
               alu_op = 1;
               branch=2;
               reg_src = 2;
               alu_src = 0;
               reg_we = 0;
               dmem_we = 0;
            end
            else if(func3 == `FUNC3_BLT) begin //blt
               alu_op = 1;
               branch=3;
               reg_src = 2;
               alu_src = 0;
               reg_we = 0;
               dmem_we = 0;
            end
       end
       `OPCODE_J: begin//jal
           alu_op = 0;
           branch=1;
           reg_src = 2;
           alu_src = 0;
           reg_we = 1;
           dmem_we = 0;
       end
    endcase
end                
endmodule

