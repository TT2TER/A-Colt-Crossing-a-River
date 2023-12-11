`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/12/05 20:52:45
// Design Name: 
// Module Name: npc
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
//////////////////////////////////////////////////////////////////////////////////

module npc(
    input [31: 0] imm,  // ��ת��pc ƫ����������������
    input jump,   //�Ƿ���Ҫ��ת,1 λ�����ź�,1��Ҫ��ת
    input  [31:0] pc, //��ǰpcֵ
    output [31:0] npc, //��һ��pcֵ
    output [31:0] pc_4 //pc+4������ jal ����ת������ʱ���淵�ص�ַ�ã��Ա���֮�󷵻�
    );
assign pc_4=pc+4; //����������pc+4�����������������pc�仯ʱ����ִ��
    
assign npc = jump==0 ?  pc+4 : pc+imm ;  //���jumpΪ0������ת��Ҳ����������pc+4)��������ת��pc+imm

endmodule
