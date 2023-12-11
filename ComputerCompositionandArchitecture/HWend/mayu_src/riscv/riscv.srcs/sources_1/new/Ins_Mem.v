`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/12/05 22:02:31
// Design Name: 
// Module Name: Ins_Mem
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
// ָ��洢��Instruction Memory
// ָ��洢���Ǽ�����е�һ���ؼ���������洢�˼������Ҫִ�е�����ָ��
//////////////////////////////////////////////////////////////////////////////////

`include "macro.vh"
module Ins_Mem(
    input [11:2] addr, // ָ��洢����32λ�Ĵ��������ʾ,ֱ��ȡ��������±ꣻ
    //ָ���ַҲ�ǰ� 4 �ֽڶ���ġ�����ζ��ָ���ַ����� 2 λ���� 0�������ʵ������У����ǿ��Ժ�������λ��ֻ��Ҫ��ʣ�µ�λ������ָ��洢��
    output [31:0] inst
    );

reg[31:0] imem[`IM_SIZE:0];//IM_SIZE=1023, ����һ���Ĵ������飬����ģ��ָ��洢��
//reg[31:0] ��ʾÿ������Ԫ�ض���һ�� 32 λ�ļĴ��������Դ洢һ�� 32 λ�����ݣ����Ӧ�� RISC-V �ܹ��е�ָ���
//imem[IM_SIZE:0] ����������Ĵ�С
assign inst=imem[addr];   //��ָ��洢���еĵ�ַ addr ��ȡһ��ָ��������ָ�ֵ�� inst

endmodule
