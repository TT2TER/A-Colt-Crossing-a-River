`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/12/09 17:22:17
// Design Name: 
// Module Name: testbench
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
module testbench();
reg clk;
reg rst;

wire [31:0] debug_reg1;
wire [31:0] debug_reg2;
wire [31:0] debug_reg3;
wire [31:0] pc;
wire [31:0] inst_d;
wire [31:0] data1;
wire [31:0] data2;
wire [31:0] data3;
wire [31:0] data4;
wire [31:0] data5;
// wire [31:0] res_d;
// wire [31:0] imm_d;
// wire alu_src_d;
// wire [31:0] RD22_d;

top_cpu my_top(
.clk(clk), 
.rst(rst),
//debug
.debug_reg1(debug_reg1),
.debug_reg2(debug_reg2),
.debug_reg3(debug_reg3),
.nowpc(pc),
.inst_d(inst_d),
.data1(data1),
.data2(data2),
.data3(data3),
.data4(data4),
.data5(data5)
);

initial begin
    // Load instructions
    $readmemh("E:/MAYU/Documents/A-Colt-Crossing-a-River/ComputerCompositionandArchitecture/HWend/riscv/test/instructions.txt", my_top.my_im.imem);
    // Load register initial values
    $readmemh("E:/MAYU/Documents/A-Colt-Crossing-a-River/ComputerCompositionandArchitecture/HWend/riscv/test/register.txt", my_top.my_rf.regFiles);
    // Load memory data initial values
    $readmemh("E:/MAYU/Documents/A-Colt-Crossing-a-River/ComputerCompositionandArchitecture/HWend/riscv/test/data_memory.txt", my_top.my_dm.dmem);
    $display("value: %h",my_top.my_im.imem[0]);
    rst = 1;
    clk = 0;
    #30 rst = 0; // 30ns 前为 rst 阶段
end

always
    #20 clk = ~clk; // 20ns 时钟周期
endmodule
