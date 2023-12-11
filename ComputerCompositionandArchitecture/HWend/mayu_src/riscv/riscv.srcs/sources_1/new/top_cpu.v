`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/12/08 21:55:01
// Design Name: 
// Module Name: top_cpu
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
// 顶层模块
// 将子模块接线
//////////////////////////////////////////////////////////////////////////////////

`include "macro.vh"
module top_cpu(
    input clk,
    input rst,

    // debug
    output [31:0] data1,
    output [31:0] data2,
    output [31:0] data3,
    output [31:0] data4,
    output [31:0] data5,
    output [31:0] debug_reg1,
    output [31:0] debug_reg2,
    output [31:0] debug_reg3,
    output [31:0] nowpc,
    output [31:0] inst_d,
    output [31:0] res_d,
    output [31:0] imm_d,
    output alu_src_d,
    output [31:0] RD22_d
    );

//线
// wire 关键字用于声明一个线网类型的变量。
// 线网类型的变量主要用于模块间的连接，
// 它们可以看作是电路中的物理线路。
// wire 变量可以连接模块的输出到其他模块的输入，或者连接到连续赋值语句的右侧。它们通常用于传递信号，但不能存储值，也就是说，你不能在过程赋值语句（如 always 块或 initial 块）中改变 wire 变量的值。
wire [31:0] pc, npc, pc_4;
wire [31:0] inst; //当前指令
wire [1:0] branch; //分支选择信号
wire [1:0] alu_op;
wire [1:0] reg_src;
wire alu_src;
wire ZF,SF;
wire dmem_we,reg_we;
//寄存器
reg jump;//存储跳转信号

wire [31:0] RD1,RD2,WD; //寄存器堆读写数据?
wire [31:0] imm;    //imm_gen
wire [31:0] res;    // 计算结果
wire [31:0] RD22;    //alu_mux
wire [31:0] RD;     //寄存器堆读出数据
wire [31:0] RF_RB;  // reg_mux

// debug
assign nowpc=pc;
assign inst_d=inst;
assign res_d=res;
assign imm_d=imm;
assign alu_src_d=alu_src;
assign RD22_d=RD22;

always @(*) begin
    //根据分支选择信号，选择跳转信号
    case (branch)
        0:  jump=0;
        1:  jump=1;
        2:  jump=ZF?1:0;
        3:  jump=SF?1:0;
    endcase
end

//实例化子模块
//程序计数器
pc my_pc(
    .clk(clk),//.clk()端口连接到了clk线上？
    .rst(rst),
    .npc(npc),
    .pc(pc)
    );

//下一个程序计数器
npc my_npc(
    .imm(imm),
    .jump(jump),
    .pc(pc),
    .npc(npc),
    .pc_4(pc_4)
    );

//指令存储器
Ins_Mem my_im(
    .addr(pc[11:2]),
    .inst(inst)
    );

//指令译码
control_unit my_cu(
    .opcode(inst[6:0]),
    .func3(inst[14:12]),
    .func7(inst[31:25]),
    .branch(branch),
    .alu_src(alu_src),
    .alu_op(alu_op),
    .reg_src(reg_src),
    .dmem_we(dmem_we),
    .reg_we(reg_we)
    );

//寄存器堆
Reg_File my_rf(
    .clk(clk),
    .rst(rst),
    .we(reg_we),
    .RA1(inst[19:15]),
    .RA2(inst[24:20]),
    .WA(inst[11:7]),
    .WD(RF_RB),
    .RD1(RD1),
    .RD2(RD2),
    .debug_reg1(debug_reg1),
    .debug_reg2(debug_reg2),
    .debug_reg3(debug_reg3)
);

//立即数生成器
Imm_Gen my_imm_gen(
    .inst(inst),
    .imm(imm)
    );

//运算器第二个操作数选择器
Alu_Mux my_alu_mux(
    .alu_src(alu_src),
    .RD2(RD2),
    .imm(imm),
    .res(RD22)
    );

//运算器
Alu my_alu(
    .op(alu_op),
    .in1(RD1),
    .in2(RD22),
    .ZF(ZF),
    .SF(SF),
    .res(res)
    );

//数据存储器
Data_Mem my_dm(
    .clk(clk),
    .rst(rst),
    .we(dmem_we),
    .addr(res[11:2]),
    .WD(RD2),
    .RD(RD),
    .data1(data1),
    .data2(data2),
    .data3(data3),
    .data4(data4),
    .data5(data5)
    );

//寄存器堆写数据选择器
Reg_Mux my_reg_mux(
    .reg_src(reg_src),
    .op_res(res),
    .dmem(RD),
    .pc(pc_4),
    .res(RF_RB)
    );

endmodule
