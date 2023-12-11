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
// ���Ƶ�Ԫ�����ɸ��ֿ����ź�
//
//////////////////////////////////////////////////////////////////////////////////

`include "macro.vh"
module control_unit(
    input [6:0] opcode,//ָ��Ĳ�����
    input [6:0] func7,//ָ��Ĺ�����
    input [2:0] func3,//ָ��Ĺ�����

    //��ת��ָ֧���ѡ���ź�,0��ʾ����ת,1��ʾjal,2��ʾbeq,3��ʾblt
            //jal���ܣ���pc+4д��Ĵ���rd����pc+immд��pc
            //beq(���ʱ��֧�����ܣ����rs1==rs2����pc+immд��pc
            //blt��С��ʱ��֧�����ܣ����rs1<rs2����pc+immд��pc
    output reg [1:0] branch,   
    // ALU�ڶ�����������ѡ���ź�,0��ʾ�ڶ����Ĵ�����1��ʾ������
    output reg alu_src,         
    // �Ĵ���д�����ݵ�ѡ���ź�, 0��ʾalu�����1��ʾ���ݴ洢����2��ʾ pc+4
    output reg [1:0] reg_src,   
    // �������ѡ���ź�,0Ϊ+��1Ϊ-,2Ϊ|
    output reg [1:0] alu_op,  
    // �洢��дʹ��
    output reg dmem_we,        
    // �Ĵ�����дʹ��
    output reg reg_we           
    );

// ����ָ�����룬����������ź�
always@(*) begin //always@(*)��ʾ�κ�һ�������źű仯ʱ������ִ��һ��
    case(opcode) //case��䣬������switch���,����opcode��ֵ��ѡ��ͬ��case
       `OPCODE_R: begin //add ,sub
            if(func7 == `FUNC7_ADD) alu_op = 0; //alu_op��ʾ�������ѡ���ź�
            else if(func7 == `FUNC7_SUB) alu_op = 1; //0Ϊ+��1Ϊ-
            branch = 0; //0��ʾ����ת
            reg_src = 0; //�Ĵ���д������ѡ��Alu���
            alu_src = 0; //alu_src 0��ʾ�ڶ����Ĵ���
            reg_we = 1; //reg_we��ʾ�Ĵ�����дʹ��
            dmem_we = 0; //dmem_we��ʾ��ֹ���ݴ洢��дʹ��
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

