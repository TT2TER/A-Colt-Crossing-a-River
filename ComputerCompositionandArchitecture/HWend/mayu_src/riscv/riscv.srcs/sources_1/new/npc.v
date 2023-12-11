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
    input [31: 0] imm,  // 跳转的pc 偏移量（立即数），
    input jump,   //是否需要跳转,1 位控制信号,1需要跳转
    input  [31:0] pc, //当前pc值
    output [31:0] npc, //下一个pc值
    output [31:0] pc_4 //pc+4，用于 jal 中跳转并链接时保存返回地址用，以便在之后返回
    );
assign pc_4=pc+4; //大多数情况下pc+4，这个操作会在输入pc变化时立即执行
    
assign npc = jump==0 ?  pc+4 : pc+imm ;  //如果jump为0，不跳转（也就是正常的pc+4)，否则跳转到pc+imm

endmodule
