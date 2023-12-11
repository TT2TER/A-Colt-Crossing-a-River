// Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2019.2 (win64) Build 2708876 Wed Nov  6 21:40:23 MST 2019
// Date        : Sat Dec  9 18:45:13 2023
// Host        : LAPTOP-GTMGGBUO running 64-bit major release  (build 9200)
// Command     : write_verilog -mode funcsim -nolib -force -file
//               E:/MAYU/Documents/A-Colt-Crossing-a-River/ComputerCompositionandArchitecture/HWend/riscv/riscv.sim/sim_1/impl/func/xsim/testbench_func_impl.v
// Design      : top_cpu
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xcku035-ffva1156-1LV-i
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module pc
   (nowpc,
    clk,
    rst);
  output [31:0]nowpc;
  input clk;
  input rst;

  wire clk;
  wire [31:0]nowpc;
  wire \pc[0]_i_2_n_0 ;
  wire \pc_reg[0]_i_1_n_0 ;
  wire \pc_reg[0]_i_1_n_10 ;
  wire \pc_reg[0]_i_1_n_11 ;
  wire \pc_reg[0]_i_1_n_12 ;
  wire \pc_reg[0]_i_1_n_13 ;
  wire \pc_reg[0]_i_1_n_14 ;
  wire \pc_reg[0]_i_1_n_15 ;
  wire \pc_reg[0]_i_1_n_8 ;
  wire \pc_reg[0]_i_1_n_9 ;
  wire \pc_reg[16]_i_1_n_0 ;
  wire \pc_reg[16]_i_1_n_10 ;
  wire \pc_reg[16]_i_1_n_11 ;
  wire \pc_reg[16]_i_1_n_12 ;
  wire \pc_reg[16]_i_1_n_13 ;
  wire \pc_reg[16]_i_1_n_14 ;
  wire \pc_reg[16]_i_1_n_15 ;
  wire \pc_reg[16]_i_1_n_8 ;
  wire \pc_reg[16]_i_1_n_9 ;
  wire \pc_reg[24]_i_1_n_10 ;
  wire \pc_reg[24]_i_1_n_11 ;
  wire \pc_reg[24]_i_1_n_12 ;
  wire \pc_reg[24]_i_1_n_13 ;
  wire \pc_reg[24]_i_1_n_14 ;
  wire \pc_reg[24]_i_1_n_15 ;
  wire \pc_reg[24]_i_1_n_8 ;
  wire \pc_reg[24]_i_1_n_9 ;
  wire \pc_reg[8]_i_1_n_0 ;
  wire \pc_reg[8]_i_1_n_10 ;
  wire \pc_reg[8]_i_1_n_11 ;
  wire \pc_reg[8]_i_1_n_12 ;
  wire \pc_reg[8]_i_1_n_13 ;
  wire \pc_reg[8]_i_1_n_14 ;
  wire \pc_reg[8]_i_1_n_15 ;
  wire \pc_reg[8]_i_1_n_8 ;
  wire \pc_reg[8]_i_1_n_9 ;
  wire rst;
  wire [6:0]\NLW_pc_reg[0]_i_1_CO_UNCONNECTED ;
  wire [6:0]\NLW_pc_reg[16]_i_1_CO_UNCONNECTED ;
  wire [7:0]\NLW_pc_reg[24]_i_1_CO_UNCONNECTED ;
  wire [6:0]\NLW_pc_reg[8]_i_1_CO_UNCONNECTED ;

  LUT1 #(
    .INIT(2'h1)) 
    \pc[0]_i_2 
       (.I0(nowpc[2]),
        .O(\pc[0]_i_2_n_0 ));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[0]_i_1_n_15 ),
        .Q(nowpc[0]));
  (* OPT_MODIFIED = "SWEEP" *) 
  CARRY8 #(
    .CARRY_TYPE("SINGLE_CY8")) 
    \pc_reg[0]_i_1 
       (.CI(1'b0),
        .CI_TOP(1'b0),
        .CO({\pc_reg[0]_i_1_n_0 ,\NLW_pc_reg[0]_i_1_CO_UNCONNECTED [6:0]}),
        .DI(nowpc[7:0]),
        .O({\pc_reg[0]_i_1_n_8 ,\pc_reg[0]_i_1_n_9 ,\pc_reg[0]_i_1_n_10 ,\pc_reg[0]_i_1_n_11 ,\pc_reg[0]_i_1_n_12 ,\pc_reg[0]_i_1_n_13 ,\pc_reg[0]_i_1_n_14 ,\pc_reg[0]_i_1_n_15 }),
        .S({nowpc[7:3],\pc[0]_i_2_n_0 ,nowpc[1:0]}));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[10] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[8]_i_1_n_13 ),
        .Q(nowpc[10]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[11] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[8]_i_1_n_12 ),
        .Q(nowpc[11]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[12] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[8]_i_1_n_11 ),
        .Q(nowpc[12]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[13] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[8]_i_1_n_10 ),
        .Q(nowpc[13]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[14] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[8]_i_1_n_9 ),
        .Q(nowpc[14]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[15] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[8]_i_1_n_8 ),
        .Q(nowpc[15]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[16] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[16]_i_1_n_15 ),
        .Q(nowpc[16]));
  (* OPT_MODIFIED = "SWEEP" *) 
  CARRY8 #(
    .CARRY_TYPE("SINGLE_CY8")) 
    \pc_reg[16]_i_1 
       (.CI(\pc_reg[8]_i_1_n_0 ),
        .CI_TOP(1'b0),
        .CO({\pc_reg[16]_i_1_n_0 ,\NLW_pc_reg[16]_i_1_CO_UNCONNECTED [6:0]}),
        .DI({1'b0,1'b0,1'b0,1'b0,nowpc[19:16]}),
        .O({\pc_reg[16]_i_1_n_8 ,\pc_reg[16]_i_1_n_9 ,\pc_reg[16]_i_1_n_10 ,\pc_reg[16]_i_1_n_11 ,\pc_reg[16]_i_1_n_12 ,\pc_reg[16]_i_1_n_13 ,\pc_reg[16]_i_1_n_14 ,\pc_reg[16]_i_1_n_15 }),
        .S(nowpc[23:16]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[17] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[16]_i_1_n_14 ),
        .Q(nowpc[17]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[18] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[16]_i_1_n_13 ),
        .Q(nowpc[18]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[19] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[16]_i_1_n_12 ),
        .Q(nowpc[19]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[0]_i_1_n_14 ),
        .Q(nowpc[1]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[20] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[16]_i_1_n_11 ),
        .Q(nowpc[20]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[21] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[16]_i_1_n_10 ),
        .Q(nowpc[21]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[22] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[16]_i_1_n_9 ),
        .Q(nowpc[22]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[23] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[16]_i_1_n_8 ),
        .Q(nowpc[23]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[24] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[24]_i_1_n_15 ),
        .Q(nowpc[24]));
  (* OPT_MODIFIED = "SWEEP" *) 
  CARRY8 #(
    .CARRY_TYPE("SINGLE_CY8")) 
    \pc_reg[24]_i_1 
       (.CI(\pc_reg[16]_i_1_n_0 ),
        .CI_TOP(1'b0),
        .CO(\NLW_pc_reg[24]_i_1_CO_UNCONNECTED [7:0]),
        .DI({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .O({\pc_reg[24]_i_1_n_8 ,\pc_reg[24]_i_1_n_9 ,\pc_reg[24]_i_1_n_10 ,\pc_reg[24]_i_1_n_11 ,\pc_reg[24]_i_1_n_12 ,\pc_reg[24]_i_1_n_13 ,\pc_reg[24]_i_1_n_14 ,\pc_reg[24]_i_1_n_15 }),
        .S(nowpc[31:24]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[25] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[24]_i_1_n_14 ),
        .Q(nowpc[25]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[26] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[24]_i_1_n_13 ),
        .Q(nowpc[26]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[27] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[24]_i_1_n_12 ),
        .Q(nowpc[27]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[28] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[24]_i_1_n_11 ),
        .Q(nowpc[28]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[29] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[24]_i_1_n_10 ),
        .Q(nowpc[29]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[0]_i_1_n_13 ),
        .Q(nowpc[2]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[30] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[24]_i_1_n_9 ),
        .Q(nowpc[30]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[31] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[24]_i_1_n_8 ),
        .Q(nowpc[31]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[3] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[0]_i_1_n_12 ),
        .Q(nowpc[3]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[4] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[0]_i_1_n_11 ),
        .Q(nowpc[4]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[5] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[0]_i_1_n_10 ),
        .Q(nowpc[5]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[6] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[0]_i_1_n_9 ),
        .Q(nowpc[6]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[7] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[0]_i_1_n_8 ),
        .Q(nowpc[7]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[8] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[8]_i_1_n_15 ),
        .Q(nowpc[8]));
  (* OPT_MODIFIED = "SWEEP" *) 
  CARRY8 #(
    .CARRY_TYPE("SINGLE_CY8")) 
    \pc_reg[8]_i_1 
       (.CI(\pc_reg[0]_i_1_n_0 ),
        .CI_TOP(1'b0),
        .CO({\pc_reg[8]_i_1_n_0 ,\NLW_pc_reg[8]_i_1_CO_UNCONNECTED [6:0]}),
        .DI(nowpc[15:8]),
        .O({\pc_reg[8]_i_1_n_8 ,\pc_reg[8]_i_1_n_9 ,\pc_reg[8]_i_1_n_10 ,\pc_reg[8]_i_1_n_11 ,\pc_reg[8]_i_1_n_12 ,\pc_reg[8]_i_1_n_13 ,\pc_reg[8]_i_1_n_14 ,\pc_reg[8]_i_1_n_15 }),
        .S(nowpc[15:8]));
  FDCE #(
    .INIT(1'b0)) 
    \pc_reg[9] 
       (.C(clk),
        .CE(1'b1),
        .CLR(rst),
        .D(\pc_reg[8]_i_1_n_14 ),
        .Q(nowpc[9]));
endmodule

(* ECO_CHECKSUM = "e7c6b866" *) 
(* NotValidForBitStream *)
module top_cpu
   (clk,
    rst,
    data1,
    data2,
    data3,
    data4,
    data5,
    debug_reg1,
    debug_reg2,
    debug_reg3,
    nowpc,
    inst_d,
    res_d,
    imm_d,
    alu_src_d,
    RD22_d);
  input clk;
  input rst;
  output [31:0]data1;
  output [31:0]data2;
  output [31:0]data3;
  output [31:0]data4;
  output [31:0]data5;
  output [31:0]debug_reg1;
  output [31:0]debug_reg2;
  output [31:0]debug_reg3;
  output [31:0]nowpc;
  output [31:0]inst_d;
  output [31:0]res_d;
  output [31:0]imm_d;
  output alu_src_d;
  output [31:0]RD22_d;

  wire [31:0]RD22_d;
  wire alu_src_d;
  wire clk;
  wire clk_IBUF;
  wire clk_IBUF_BUFG;
  wire [31:0]data1;
  wire [31:0]data2;
  wire [31:0]data3;
  wire [31:0]data4;
  wire [31:0]data5;
  wire [31:0]debug_reg1;
  wire [31:0]debug_reg2;
  wire [31:0]debug_reg3;
  wire [31:0]imm_d;
  wire [31:0]inst_d;
  wire [31:0]nowpc;
  wire [31:0]nowpc_OBUF;
  wire [31:0]res_d;
  wire rst;
  wire rst_IBUF;

  OBUF \RD22_d_OBUF[0]_inst 
       (.I(1'b0),
        .O(RD22_d[0]));
  OBUF \RD22_d_OBUF[10]_inst 
       (.I(1'b0),
        .O(RD22_d[10]));
  OBUF \RD22_d_OBUF[11]_inst 
       (.I(1'b0),
        .O(RD22_d[11]));
  OBUF \RD22_d_OBUF[12]_inst 
       (.I(1'b0),
        .O(RD22_d[12]));
  OBUF \RD22_d_OBUF[13]_inst 
       (.I(1'b0),
        .O(RD22_d[13]));
  OBUF \RD22_d_OBUF[14]_inst 
       (.I(1'b0),
        .O(RD22_d[14]));
  OBUF \RD22_d_OBUF[15]_inst 
       (.I(1'b0),
        .O(RD22_d[15]));
  OBUF \RD22_d_OBUF[16]_inst 
       (.I(1'b0),
        .O(RD22_d[16]));
  OBUF \RD22_d_OBUF[17]_inst 
       (.I(1'b0),
        .O(RD22_d[17]));
  OBUF \RD22_d_OBUF[18]_inst 
       (.I(1'b0),
        .O(RD22_d[18]));
  OBUF \RD22_d_OBUF[19]_inst 
       (.I(1'b0),
        .O(RD22_d[19]));
  OBUF \RD22_d_OBUF[1]_inst 
       (.I(1'b0),
        .O(RD22_d[1]));
  OBUF \RD22_d_OBUF[20]_inst 
       (.I(1'b0),
        .O(RD22_d[20]));
  OBUF \RD22_d_OBUF[21]_inst 
       (.I(1'b0),
        .O(RD22_d[21]));
  OBUF \RD22_d_OBUF[22]_inst 
       (.I(1'b0),
        .O(RD22_d[22]));
  OBUF \RD22_d_OBUF[23]_inst 
       (.I(1'b0),
        .O(RD22_d[23]));
  OBUF \RD22_d_OBUF[24]_inst 
       (.I(1'b0),
        .O(RD22_d[24]));
  OBUF \RD22_d_OBUF[25]_inst 
       (.I(1'b0),
        .O(RD22_d[25]));
  OBUF \RD22_d_OBUF[26]_inst 
       (.I(1'b0),
        .O(RD22_d[26]));
  OBUF \RD22_d_OBUF[27]_inst 
       (.I(1'b0),
        .O(RD22_d[27]));
  OBUF \RD22_d_OBUF[28]_inst 
       (.I(1'b0),
        .O(RD22_d[28]));
  OBUF \RD22_d_OBUF[29]_inst 
       (.I(1'b0),
        .O(RD22_d[29]));
  OBUF \RD22_d_OBUF[2]_inst 
       (.I(1'b0),
        .O(RD22_d[2]));
  OBUF \RD22_d_OBUF[30]_inst 
       (.I(1'b0),
        .O(RD22_d[30]));
  OBUF \RD22_d_OBUF[31]_inst 
       (.I(1'b0),
        .O(RD22_d[31]));
  OBUF \RD22_d_OBUF[3]_inst 
       (.I(1'b0),
        .O(RD22_d[3]));
  OBUF \RD22_d_OBUF[4]_inst 
       (.I(1'b0),
        .O(RD22_d[4]));
  OBUF \RD22_d_OBUF[5]_inst 
       (.I(1'b0),
        .O(RD22_d[5]));
  OBUF \RD22_d_OBUF[6]_inst 
       (.I(1'b0),
        .O(RD22_d[6]));
  OBUF \RD22_d_OBUF[7]_inst 
       (.I(1'b0),
        .O(RD22_d[7]));
  OBUF \RD22_d_OBUF[8]_inst 
       (.I(1'b0),
        .O(RD22_d[8]));
  OBUF \RD22_d_OBUF[9]_inst 
       (.I(1'b0),
        .O(RD22_d[9]));
  OBUF alu_src_d_OBUF_inst
       (.I(1'b1),
        .O(alu_src_d));
  (* XILINX_LEGACY_PRIM = "BUFG" *) 
  BUFGCE #(
    .CE_TYPE("ASYNC"),
    .SIM_DEVICE("ULTRASCALE"),
    .STARTUP_SYNC("FALSE")) 
    clk_IBUF_BUFG_inst
       (.CE(1'b1),
        .I(clk_IBUF),
        .O(clk_IBUF_BUFG));
  IBUF clk_IBUF_inst
       (.I(clk),
        .O(clk_IBUF));
  OBUF \data1_OBUF[0]_inst 
       (.I(1'b0),
        .O(data1[0]));
  OBUF \data1_OBUF[10]_inst 
       (.I(1'b0),
        .O(data1[10]));
  OBUF \data1_OBUF[11]_inst 
       (.I(1'b0),
        .O(data1[11]));
  OBUF \data1_OBUF[12]_inst 
       (.I(1'b0),
        .O(data1[12]));
  OBUF \data1_OBUF[13]_inst 
       (.I(1'b0),
        .O(data1[13]));
  OBUF \data1_OBUF[14]_inst 
       (.I(1'b0),
        .O(data1[14]));
  OBUF \data1_OBUF[15]_inst 
       (.I(1'b0),
        .O(data1[15]));
  OBUF \data1_OBUF[16]_inst 
       (.I(1'b0),
        .O(data1[16]));
  OBUF \data1_OBUF[17]_inst 
       (.I(1'b0),
        .O(data1[17]));
  OBUF \data1_OBUF[18]_inst 
       (.I(1'b0),
        .O(data1[18]));
  OBUF \data1_OBUF[19]_inst 
       (.I(1'b0),
        .O(data1[19]));
  OBUF \data1_OBUF[1]_inst 
       (.I(1'b0),
        .O(data1[1]));
  OBUF \data1_OBUF[20]_inst 
       (.I(1'b0),
        .O(data1[20]));
  OBUF \data1_OBUF[21]_inst 
       (.I(1'b0),
        .O(data1[21]));
  OBUF \data1_OBUF[22]_inst 
       (.I(1'b0),
        .O(data1[22]));
  OBUF \data1_OBUF[23]_inst 
       (.I(1'b0),
        .O(data1[23]));
  OBUF \data1_OBUF[24]_inst 
       (.I(1'b0),
        .O(data1[24]));
  OBUF \data1_OBUF[25]_inst 
       (.I(1'b0),
        .O(data1[25]));
  OBUF \data1_OBUF[26]_inst 
       (.I(1'b0),
        .O(data1[26]));
  OBUF \data1_OBUF[27]_inst 
       (.I(1'b0),
        .O(data1[27]));
  OBUF \data1_OBUF[28]_inst 
       (.I(1'b0),
        .O(data1[28]));
  OBUF \data1_OBUF[29]_inst 
       (.I(1'b0),
        .O(data1[29]));
  OBUF \data1_OBUF[2]_inst 
       (.I(1'b0),
        .O(data1[2]));
  OBUF \data1_OBUF[30]_inst 
       (.I(1'b0),
        .O(data1[30]));
  OBUF \data1_OBUF[31]_inst 
       (.I(1'b0),
        .O(data1[31]));
  OBUF \data1_OBUF[3]_inst 
       (.I(1'b0),
        .O(data1[3]));
  OBUF \data1_OBUF[4]_inst 
       (.I(1'b0),
        .O(data1[4]));
  OBUF \data1_OBUF[5]_inst 
       (.I(1'b0),
        .O(data1[5]));
  OBUF \data1_OBUF[6]_inst 
       (.I(1'b0),
        .O(data1[6]));
  OBUF \data1_OBUF[7]_inst 
       (.I(1'b0),
        .O(data1[7]));
  OBUF \data1_OBUF[8]_inst 
       (.I(1'b0),
        .O(data1[8]));
  OBUF \data1_OBUF[9]_inst 
       (.I(1'b0),
        .O(data1[9]));
  OBUF \data2_OBUF[0]_inst 
       (.I(1'b0),
        .O(data2[0]));
  OBUF \data2_OBUF[10]_inst 
       (.I(1'b0),
        .O(data2[10]));
  OBUF \data2_OBUF[11]_inst 
       (.I(1'b0),
        .O(data2[11]));
  OBUF \data2_OBUF[12]_inst 
       (.I(1'b0),
        .O(data2[12]));
  OBUF \data2_OBUF[13]_inst 
       (.I(1'b0),
        .O(data2[13]));
  OBUF \data2_OBUF[14]_inst 
       (.I(1'b0),
        .O(data2[14]));
  OBUF \data2_OBUF[15]_inst 
       (.I(1'b0),
        .O(data2[15]));
  OBUF \data2_OBUF[16]_inst 
       (.I(1'b0),
        .O(data2[16]));
  OBUF \data2_OBUF[17]_inst 
       (.I(1'b0),
        .O(data2[17]));
  OBUF \data2_OBUF[18]_inst 
       (.I(1'b0),
        .O(data2[18]));
  OBUF \data2_OBUF[19]_inst 
       (.I(1'b0),
        .O(data2[19]));
  OBUF \data2_OBUF[1]_inst 
       (.I(1'b0),
        .O(data2[1]));
  OBUF \data2_OBUF[20]_inst 
       (.I(1'b0),
        .O(data2[20]));
  OBUF \data2_OBUF[21]_inst 
       (.I(1'b0),
        .O(data2[21]));
  OBUF \data2_OBUF[22]_inst 
       (.I(1'b0),
        .O(data2[22]));
  OBUF \data2_OBUF[23]_inst 
       (.I(1'b0),
        .O(data2[23]));
  OBUF \data2_OBUF[24]_inst 
       (.I(1'b0),
        .O(data2[24]));
  OBUF \data2_OBUF[25]_inst 
       (.I(1'b0),
        .O(data2[25]));
  OBUF \data2_OBUF[26]_inst 
       (.I(1'b0),
        .O(data2[26]));
  OBUF \data2_OBUF[27]_inst 
       (.I(1'b0),
        .O(data2[27]));
  OBUF \data2_OBUF[28]_inst 
       (.I(1'b0),
        .O(data2[28]));
  OBUF \data2_OBUF[29]_inst 
       (.I(1'b0),
        .O(data2[29]));
  OBUF \data2_OBUF[2]_inst 
       (.I(1'b0),
        .O(data2[2]));
  OBUF \data2_OBUF[30]_inst 
       (.I(1'b0),
        .O(data2[30]));
  OBUF \data2_OBUF[31]_inst 
       (.I(1'b0),
        .O(data2[31]));
  OBUF \data2_OBUF[3]_inst 
       (.I(1'b0),
        .O(data2[3]));
  OBUF \data2_OBUF[4]_inst 
       (.I(1'b0),
        .O(data2[4]));
  OBUF \data2_OBUF[5]_inst 
       (.I(1'b0),
        .O(data2[5]));
  OBUF \data2_OBUF[6]_inst 
       (.I(1'b0),
        .O(data2[6]));
  OBUF \data2_OBUF[7]_inst 
       (.I(1'b0),
        .O(data2[7]));
  OBUF \data2_OBUF[8]_inst 
       (.I(1'b0),
        .O(data2[8]));
  OBUF \data2_OBUF[9]_inst 
       (.I(1'b0),
        .O(data2[9]));
  OBUF \data3_OBUF[0]_inst 
       (.I(1'b0),
        .O(data3[0]));
  OBUF \data3_OBUF[10]_inst 
       (.I(1'b0),
        .O(data3[10]));
  OBUF \data3_OBUF[11]_inst 
       (.I(1'b0),
        .O(data3[11]));
  OBUF \data3_OBUF[12]_inst 
       (.I(1'b0),
        .O(data3[12]));
  OBUF \data3_OBUF[13]_inst 
       (.I(1'b0),
        .O(data3[13]));
  OBUF \data3_OBUF[14]_inst 
       (.I(1'b0),
        .O(data3[14]));
  OBUF \data3_OBUF[15]_inst 
       (.I(1'b0),
        .O(data3[15]));
  OBUF \data3_OBUF[16]_inst 
       (.I(1'b0),
        .O(data3[16]));
  OBUF \data3_OBUF[17]_inst 
       (.I(1'b0),
        .O(data3[17]));
  OBUF \data3_OBUF[18]_inst 
       (.I(1'b0),
        .O(data3[18]));
  OBUF \data3_OBUF[19]_inst 
       (.I(1'b0),
        .O(data3[19]));
  OBUF \data3_OBUF[1]_inst 
       (.I(1'b0),
        .O(data3[1]));
  OBUF \data3_OBUF[20]_inst 
       (.I(1'b0),
        .O(data3[20]));
  OBUF \data3_OBUF[21]_inst 
       (.I(1'b0),
        .O(data3[21]));
  OBUF \data3_OBUF[22]_inst 
       (.I(1'b0),
        .O(data3[22]));
  OBUF \data3_OBUF[23]_inst 
       (.I(1'b0),
        .O(data3[23]));
  OBUF \data3_OBUF[24]_inst 
       (.I(1'b0),
        .O(data3[24]));
  OBUF \data3_OBUF[25]_inst 
       (.I(1'b0),
        .O(data3[25]));
  OBUF \data3_OBUF[26]_inst 
       (.I(1'b0),
        .O(data3[26]));
  OBUF \data3_OBUF[27]_inst 
       (.I(1'b0),
        .O(data3[27]));
  OBUF \data3_OBUF[28]_inst 
       (.I(1'b0),
        .O(data3[28]));
  OBUF \data3_OBUF[29]_inst 
       (.I(1'b0),
        .O(data3[29]));
  OBUF \data3_OBUF[2]_inst 
       (.I(1'b0),
        .O(data3[2]));
  OBUF \data3_OBUF[30]_inst 
       (.I(1'b0),
        .O(data3[30]));
  OBUF \data3_OBUF[31]_inst 
       (.I(1'b0),
        .O(data3[31]));
  OBUF \data3_OBUF[3]_inst 
       (.I(1'b0),
        .O(data3[3]));
  OBUF \data3_OBUF[4]_inst 
       (.I(1'b0),
        .O(data3[4]));
  OBUF \data3_OBUF[5]_inst 
       (.I(1'b0),
        .O(data3[5]));
  OBUF \data3_OBUF[6]_inst 
       (.I(1'b0),
        .O(data3[6]));
  OBUF \data3_OBUF[7]_inst 
       (.I(1'b0),
        .O(data3[7]));
  OBUF \data3_OBUF[8]_inst 
       (.I(1'b0),
        .O(data3[8]));
  OBUF \data3_OBUF[9]_inst 
       (.I(1'b0),
        .O(data3[9]));
  OBUF \data4_OBUF[0]_inst 
       (.I(1'b0),
        .O(data4[0]));
  OBUF \data4_OBUF[10]_inst 
       (.I(1'b0),
        .O(data4[10]));
  OBUF \data4_OBUF[11]_inst 
       (.I(1'b0),
        .O(data4[11]));
  OBUF \data4_OBUF[12]_inst 
       (.I(1'b0),
        .O(data4[12]));
  OBUF \data4_OBUF[13]_inst 
       (.I(1'b0),
        .O(data4[13]));
  OBUF \data4_OBUF[14]_inst 
       (.I(1'b0),
        .O(data4[14]));
  OBUF \data4_OBUF[15]_inst 
       (.I(1'b0),
        .O(data4[15]));
  OBUF \data4_OBUF[16]_inst 
       (.I(1'b0),
        .O(data4[16]));
  OBUF \data4_OBUF[17]_inst 
       (.I(1'b0),
        .O(data4[17]));
  OBUF \data4_OBUF[18]_inst 
       (.I(1'b0),
        .O(data4[18]));
  OBUF \data4_OBUF[19]_inst 
       (.I(1'b0),
        .O(data4[19]));
  OBUF \data4_OBUF[1]_inst 
       (.I(1'b0),
        .O(data4[1]));
  OBUF \data4_OBUF[20]_inst 
       (.I(1'b0),
        .O(data4[20]));
  OBUF \data4_OBUF[21]_inst 
       (.I(1'b0),
        .O(data4[21]));
  OBUF \data4_OBUF[22]_inst 
       (.I(1'b0),
        .O(data4[22]));
  OBUF \data4_OBUF[23]_inst 
       (.I(1'b0),
        .O(data4[23]));
  OBUF \data4_OBUF[24]_inst 
       (.I(1'b0),
        .O(data4[24]));
  OBUF \data4_OBUF[25]_inst 
       (.I(1'b0),
        .O(data4[25]));
  OBUF \data4_OBUF[26]_inst 
       (.I(1'b0),
        .O(data4[26]));
  OBUF \data4_OBUF[27]_inst 
       (.I(1'b0),
        .O(data4[27]));
  OBUF \data4_OBUF[28]_inst 
       (.I(1'b0),
        .O(data4[28]));
  OBUF \data4_OBUF[29]_inst 
       (.I(1'b0),
        .O(data4[29]));
  OBUF \data4_OBUF[2]_inst 
       (.I(1'b0),
        .O(data4[2]));
  OBUF \data4_OBUF[30]_inst 
       (.I(1'b0),
        .O(data4[30]));
  OBUF \data4_OBUF[31]_inst 
       (.I(1'b0),
        .O(data4[31]));
  OBUF \data4_OBUF[3]_inst 
       (.I(1'b0),
        .O(data4[3]));
  OBUF \data4_OBUF[4]_inst 
       (.I(1'b0),
        .O(data4[4]));
  OBUF \data4_OBUF[5]_inst 
       (.I(1'b0),
        .O(data4[5]));
  OBUF \data4_OBUF[6]_inst 
       (.I(1'b0),
        .O(data4[6]));
  OBUF \data4_OBUF[7]_inst 
       (.I(1'b0),
        .O(data4[7]));
  OBUF \data4_OBUF[8]_inst 
       (.I(1'b0),
        .O(data4[8]));
  OBUF \data4_OBUF[9]_inst 
       (.I(1'b0),
        .O(data4[9]));
  OBUF \data5_OBUF[0]_inst 
       (.I(1'b0),
        .O(data5[0]));
  OBUF \data5_OBUF[10]_inst 
       (.I(1'b0),
        .O(data5[10]));
  OBUF \data5_OBUF[11]_inst 
       (.I(1'b0),
        .O(data5[11]));
  OBUF \data5_OBUF[12]_inst 
       (.I(1'b0),
        .O(data5[12]));
  OBUF \data5_OBUF[13]_inst 
       (.I(1'b0),
        .O(data5[13]));
  OBUF \data5_OBUF[14]_inst 
       (.I(1'b0),
        .O(data5[14]));
  OBUF \data5_OBUF[15]_inst 
       (.I(1'b0),
        .O(data5[15]));
  OBUF \data5_OBUF[16]_inst 
       (.I(1'b0),
        .O(data5[16]));
  OBUF \data5_OBUF[17]_inst 
       (.I(1'b0),
        .O(data5[17]));
  OBUF \data5_OBUF[18]_inst 
       (.I(1'b0),
        .O(data5[18]));
  OBUF \data5_OBUF[19]_inst 
       (.I(1'b0),
        .O(data5[19]));
  OBUF \data5_OBUF[1]_inst 
       (.I(1'b0),
        .O(data5[1]));
  OBUF \data5_OBUF[20]_inst 
       (.I(1'b0),
        .O(data5[20]));
  OBUF \data5_OBUF[21]_inst 
       (.I(1'b0),
        .O(data5[21]));
  OBUF \data5_OBUF[22]_inst 
       (.I(1'b0),
        .O(data5[22]));
  OBUF \data5_OBUF[23]_inst 
       (.I(1'b0),
        .O(data5[23]));
  OBUF \data5_OBUF[24]_inst 
       (.I(1'b0),
        .O(data5[24]));
  OBUF \data5_OBUF[25]_inst 
       (.I(1'b0),
        .O(data5[25]));
  OBUF \data5_OBUF[26]_inst 
       (.I(1'b0),
        .O(data5[26]));
  OBUF \data5_OBUF[27]_inst 
       (.I(1'b0),
        .O(data5[27]));
  OBUF \data5_OBUF[28]_inst 
       (.I(1'b0),
        .O(data5[28]));
  OBUF \data5_OBUF[29]_inst 
       (.I(1'b0),
        .O(data5[29]));
  OBUF \data5_OBUF[2]_inst 
       (.I(1'b0),
        .O(data5[2]));
  OBUF \data5_OBUF[30]_inst 
       (.I(1'b0),
        .O(data5[30]));
  OBUF \data5_OBUF[31]_inst 
       (.I(1'b0),
        .O(data5[31]));
  OBUF \data5_OBUF[3]_inst 
       (.I(1'b0),
        .O(data5[3]));
  OBUF \data5_OBUF[4]_inst 
       (.I(1'b0),
        .O(data5[4]));
  OBUF \data5_OBUF[5]_inst 
       (.I(1'b0),
        .O(data5[5]));
  OBUF \data5_OBUF[6]_inst 
       (.I(1'b0),
        .O(data5[6]));
  OBUF \data5_OBUF[7]_inst 
       (.I(1'b0),
        .O(data5[7]));
  OBUF \data5_OBUF[8]_inst 
       (.I(1'b0),
        .O(data5[8]));
  OBUF \data5_OBUF[9]_inst 
       (.I(1'b0),
        .O(data5[9]));
  OBUF \debug_reg1_OBUF[0]_inst 
       (.I(1'b0),
        .O(debug_reg1[0]));
  OBUF \debug_reg1_OBUF[10]_inst 
       (.I(1'b0),
        .O(debug_reg1[10]));
  OBUF \debug_reg1_OBUF[11]_inst 
       (.I(1'b0),
        .O(debug_reg1[11]));
  OBUF \debug_reg1_OBUF[12]_inst 
       (.I(1'b0),
        .O(debug_reg1[12]));
  OBUF \debug_reg1_OBUF[13]_inst 
       (.I(1'b0),
        .O(debug_reg1[13]));
  OBUF \debug_reg1_OBUF[14]_inst 
       (.I(1'b0),
        .O(debug_reg1[14]));
  OBUF \debug_reg1_OBUF[15]_inst 
       (.I(1'b0),
        .O(debug_reg1[15]));
  OBUF \debug_reg1_OBUF[16]_inst 
       (.I(1'b0),
        .O(debug_reg1[16]));
  OBUF \debug_reg1_OBUF[17]_inst 
       (.I(1'b0),
        .O(debug_reg1[17]));
  OBUF \debug_reg1_OBUF[18]_inst 
       (.I(1'b0),
        .O(debug_reg1[18]));
  OBUF \debug_reg1_OBUF[19]_inst 
       (.I(1'b0),
        .O(debug_reg1[19]));
  OBUF \debug_reg1_OBUF[1]_inst 
       (.I(1'b0),
        .O(debug_reg1[1]));
  OBUF \debug_reg1_OBUF[20]_inst 
       (.I(1'b0),
        .O(debug_reg1[20]));
  OBUF \debug_reg1_OBUF[21]_inst 
       (.I(1'b0),
        .O(debug_reg1[21]));
  OBUF \debug_reg1_OBUF[22]_inst 
       (.I(1'b0),
        .O(debug_reg1[22]));
  OBUF \debug_reg1_OBUF[23]_inst 
       (.I(1'b0),
        .O(debug_reg1[23]));
  OBUF \debug_reg1_OBUF[24]_inst 
       (.I(1'b0),
        .O(debug_reg1[24]));
  OBUF \debug_reg1_OBUF[25]_inst 
       (.I(1'b0),
        .O(debug_reg1[25]));
  OBUF \debug_reg1_OBUF[26]_inst 
       (.I(1'b0),
        .O(debug_reg1[26]));
  OBUF \debug_reg1_OBUF[27]_inst 
       (.I(1'b0),
        .O(debug_reg1[27]));
  OBUF \debug_reg1_OBUF[28]_inst 
       (.I(1'b0),
        .O(debug_reg1[28]));
  OBUF \debug_reg1_OBUF[29]_inst 
       (.I(1'b0),
        .O(debug_reg1[29]));
  OBUF \debug_reg1_OBUF[2]_inst 
       (.I(1'b0),
        .O(debug_reg1[2]));
  OBUF \debug_reg1_OBUF[30]_inst 
       (.I(1'b0),
        .O(debug_reg1[30]));
  OBUF \debug_reg1_OBUF[31]_inst 
       (.I(1'b0),
        .O(debug_reg1[31]));
  OBUF \debug_reg1_OBUF[3]_inst 
       (.I(1'b0),
        .O(debug_reg1[3]));
  OBUF \debug_reg1_OBUF[4]_inst 
       (.I(1'b0),
        .O(debug_reg1[4]));
  OBUF \debug_reg1_OBUF[5]_inst 
       (.I(1'b0),
        .O(debug_reg1[5]));
  OBUF \debug_reg1_OBUF[6]_inst 
       (.I(1'b0),
        .O(debug_reg1[6]));
  OBUF \debug_reg1_OBUF[7]_inst 
       (.I(1'b0),
        .O(debug_reg1[7]));
  OBUF \debug_reg1_OBUF[8]_inst 
       (.I(1'b0),
        .O(debug_reg1[8]));
  OBUF \debug_reg1_OBUF[9]_inst 
       (.I(1'b0),
        .O(debug_reg1[9]));
  OBUF \debug_reg2_OBUF[0]_inst 
       (.I(1'b0),
        .O(debug_reg2[0]));
  OBUF \debug_reg2_OBUF[10]_inst 
       (.I(1'b0),
        .O(debug_reg2[10]));
  OBUF \debug_reg2_OBUF[11]_inst 
       (.I(1'b0),
        .O(debug_reg2[11]));
  OBUF \debug_reg2_OBUF[12]_inst 
       (.I(1'b0),
        .O(debug_reg2[12]));
  OBUF \debug_reg2_OBUF[13]_inst 
       (.I(1'b0),
        .O(debug_reg2[13]));
  OBUF \debug_reg2_OBUF[14]_inst 
       (.I(1'b0),
        .O(debug_reg2[14]));
  OBUF \debug_reg2_OBUF[15]_inst 
       (.I(1'b0),
        .O(debug_reg2[15]));
  OBUF \debug_reg2_OBUF[16]_inst 
       (.I(1'b0),
        .O(debug_reg2[16]));
  OBUF \debug_reg2_OBUF[17]_inst 
       (.I(1'b0),
        .O(debug_reg2[17]));
  OBUF \debug_reg2_OBUF[18]_inst 
       (.I(1'b0),
        .O(debug_reg2[18]));
  OBUF \debug_reg2_OBUF[19]_inst 
       (.I(1'b0),
        .O(debug_reg2[19]));
  OBUF \debug_reg2_OBUF[1]_inst 
       (.I(1'b0),
        .O(debug_reg2[1]));
  OBUF \debug_reg2_OBUF[20]_inst 
       (.I(1'b0),
        .O(debug_reg2[20]));
  OBUF \debug_reg2_OBUF[21]_inst 
       (.I(1'b0),
        .O(debug_reg2[21]));
  OBUF \debug_reg2_OBUF[22]_inst 
       (.I(1'b0),
        .O(debug_reg2[22]));
  OBUF \debug_reg2_OBUF[23]_inst 
       (.I(1'b0),
        .O(debug_reg2[23]));
  OBUF \debug_reg2_OBUF[24]_inst 
       (.I(1'b0),
        .O(debug_reg2[24]));
  OBUF \debug_reg2_OBUF[25]_inst 
       (.I(1'b0),
        .O(debug_reg2[25]));
  OBUF \debug_reg2_OBUF[26]_inst 
       (.I(1'b0),
        .O(debug_reg2[26]));
  OBUF \debug_reg2_OBUF[27]_inst 
       (.I(1'b0),
        .O(debug_reg2[27]));
  OBUF \debug_reg2_OBUF[28]_inst 
       (.I(1'b0),
        .O(debug_reg2[28]));
  OBUF \debug_reg2_OBUF[29]_inst 
       (.I(1'b0),
        .O(debug_reg2[29]));
  OBUF \debug_reg2_OBUF[2]_inst 
       (.I(1'b0),
        .O(debug_reg2[2]));
  OBUF \debug_reg2_OBUF[30]_inst 
       (.I(1'b0),
        .O(debug_reg2[30]));
  OBUF \debug_reg2_OBUF[31]_inst 
       (.I(1'b0),
        .O(debug_reg2[31]));
  OBUF \debug_reg2_OBUF[3]_inst 
       (.I(1'b0),
        .O(debug_reg2[3]));
  OBUF \debug_reg2_OBUF[4]_inst 
       (.I(1'b0),
        .O(debug_reg2[4]));
  OBUF \debug_reg2_OBUF[5]_inst 
       (.I(1'b0),
        .O(debug_reg2[5]));
  OBUF \debug_reg2_OBUF[6]_inst 
       (.I(1'b0),
        .O(debug_reg2[6]));
  OBUF \debug_reg2_OBUF[7]_inst 
       (.I(1'b0),
        .O(debug_reg2[7]));
  OBUF \debug_reg2_OBUF[8]_inst 
       (.I(1'b0),
        .O(debug_reg2[8]));
  OBUF \debug_reg2_OBUF[9]_inst 
       (.I(1'b0),
        .O(debug_reg2[9]));
  OBUF \debug_reg3_OBUF[0]_inst 
       (.I(1'b0),
        .O(debug_reg3[0]));
  OBUF \debug_reg3_OBUF[10]_inst 
       (.I(1'b0),
        .O(debug_reg3[10]));
  OBUF \debug_reg3_OBUF[11]_inst 
       (.I(1'b0),
        .O(debug_reg3[11]));
  OBUF \debug_reg3_OBUF[12]_inst 
       (.I(1'b0),
        .O(debug_reg3[12]));
  OBUF \debug_reg3_OBUF[13]_inst 
       (.I(1'b0),
        .O(debug_reg3[13]));
  OBUF \debug_reg3_OBUF[14]_inst 
       (.I(1'b0),
        .O(debug_reg3[14]));
  OBUF \debug_reg3_OBUF[15]_inst 
       (.I(1'b0),
        .O(debug_reg3[15]));
  OBUF \debug_reg3_OBUF[16]_inst 
       (.I(1'b0),
        .O(debug_reg3[16]));
  OBUF \debug_reg3_OBUF[17]_inst 
       (.I(1'b0),
        .O(debug_reg3[17]));
  OBUF \debug_reg3_OBUF[18]_inst 
       (.I(1'b0),
        .O(debug_reg3[18]));
  OBUF \debug_reg3_OBUF[19]_inst 
       (.I(1'b0),
        .O(debug_reg3[19]));
  OBUF \debug_reg3_OBUF[1]_inst 
       (.I(1'b0),
        .O(debug_reg3[1]));
  OBUF \debug_reg3_OBUF[20]_inst 
       (.I(1'b0),
        .O(debug_reg3[20]));
  OBUF \debug_reg3_OBUF[21]_inst 
       (.I(1'b0),
        .O(debug_reg3[21]));
  OBUF \debug_reg3_OBUF[22]_inst 
       (.I(1'b0),
        .O(debug_reg3[22]));
  OBUF \debug_reg3_OBUF[23]_inst 
       (.I(1'b0),
        .O(debug_reg3[23]));
  OBUF \debug_reg3_OBUF[24]_inst 
       (.I(1'b0),
        .O(debug_reg3[24]));
  OBUF \debug_reg3_OBUF[25]_inst 
       (.I(1'b0),
        .O(debug_reg3[25]));
  OBUF \debug_reg3_OBUF[26]_inst 
       (.I(1'b0),
        .O(debug_reg3[26]));
  OBUF \debug_reg3_OBUF[27]_inst 
       (.I(1'b0),
        .O(debug_reg3[27]));
  OBUF \debug_reg3_OBUF[28]_inst 
       (.I(1'b0),
        .O(debug_reg3[28]));
  OBUF \debug_reg3_OBUF[29]_inst 
       (.I(1'b0),
        .O(debug_reg3[29]));
  OBUF \debug_reg3_OBUF[2]_inst 
       (.I(1'b0),
        .O(debug_reg3[2]));
  OBUF \debug_reg3_OBUF[30]_inst 
       (.I(1'b0),
        .O(debug_reg3[30]));
  OBUF \debug_reg3_OBUF[31]_inst 
       (.I(1'b0),
        .O(debug_reg3[31]));
  OBUF \debug_reg3_OBUF[3]_inst 
       (.I(1'b0),
        .O(debug_reg3[3]));
  OBUF \debug_reg3_OBUF[4]_inst 
       (.I(1'b0),
        .O(debug_reg3[4]));
  OBUF \debug_reg3_OBUF[5]_inst 
       (.I(1'b0),
        .O(debug_reg3[5]));
  OBUF \debug_reg3_OBUF[6]_inst 
       (.I(1'b0),
        .O(debug_reg3[6]));
  OBUF \debug_reg3_OBUF[7]_inst 
       (.I(1'b0),
        .O(debug_reg3[7]));
  OBUF \debug_reg3_OBUF[8]_inst 
       (.I(1'b0),
        .O(debug_reg3[8]));
  OBUF \debug_reg3_OBUF[9]_inst 
       (.I(1'b0),
        .O(debug_reg3[9]));
  OBUF \imm_d_OBUF[0]_inst 
       (.I(1'b0),
        .O(imm_d[0]));
  OBUF \imm_d_OBUF[10]_inst 
       (.I(1'b0),
        .O(imm_d[10]));
  OBUF \imm_d_OBUF[11]_inst 
       (.I(1'b0),
        .O(imm_d[11]));
  OBUF \imm_d_OBUF[12]_inst 
       (.I(1'b0),
        .O(imm_d[12]));
  OBUF \imm_d_OBUF[13]_inst 
       (.I(1'b0),
        .O(imm_d[13]));
  OBUF \imm_d_OBUF[14]_inst 
       (.I(1'b0),
        .O(imm_d[14]));
  OBUF \imm_d_OBUF[15]_inst 
       (.I(1'b0),
        .O(imm_d[15]));
  OBUF \imm_d_OBUF[16]_inst 
       (.I(1'b0),
        .O(imm_d[16]));
  OBUF \imm_d_OBUF[17]_inst 
       (.I(1'b0),
        .O(imm_d[17]));
  OBUF \imm_d_OBUF[18]_inst 
       (.I(1'b0),
        .O(imm_d[18]));
  OBUF \imm_d_OBUF[19]_inst 
       (.I(1'b0),
        .O(imm_d[19]));
  OBUF \imm_d_OBUF[1]_inst 
       (.I(1'b0),
        .O(imm_d[1]));
  OBUF \imm_d_OBUF[20]_inst 
       (.I(1'b0),
        .O(imm_d[20]));
  OBUF \imm_d_OBUF[21]_inst 
       (.I(1'b0),
        .O(imm_d[21]));
  OBUF \imm_d_OBUF[22]_inst 
       (.I(1'b0),
        .O(imm_d[22]));
  OBUF \imm_d_OBUF[23]_inst 
       (.I(1'b0),
        .O(imm_d[23]));
  OBUF \imm_d_OBUF[24]_inst 
       (.I(1'b0),
        .O(imm_d[24]));
  OBUF \imm_d_OBUF[25]_inst 
       (.I(1'b0),
        .O(imm_d[25]));
  OBUF \imm_d_OBUF[26]_inst 
       (.I(1'b0),
        .O(imm_d[26]));
  OBUF \imm_d_OBUF[27]_inst 
       (.I(1'b0),
        .O(imm_d[27]));
  OBUF \imm_d_OBUF[28]_inst 
       (.I(1'b0),
        .O(imm_d[28]));
  OBUF \imm_d_OBUF[29]_inst 
       (.I(1'b0),
        .O(imm_d[29]));
  OBUF \imm_d_OBUF[2]_inst 
       (.I(1'b0),
        .O(imm_d[2]));
  OBUF \imm_d_OBUF[30]_inst 
       (.I(1'b0),
        .O(imm_d[30]));
  OBUF \imm_d_OBUF[31]_inst 
       (.I(1'b0),
        .O(imm_d[31]));
  OBUF \imm_d_OBUF[3]_inst 
       (.I(1'b0),
        .O(imm_d[3]));
  OBUF \imm_d_OBUF[4]_inst 
       (.I(1'b0),
        .O(imm_d[4]));
  OBUF \imm_d_OBUF[5]_inst 
       (.I(1'b0),
        .O(imm_d[5]));
  OBUF \imm_d_OBUF[6]_inst 
       (.I(1'b0),
        .O(imm_d[6]));
  OBUF \imm_d_OBUF[7]_inst 
       (.I(1'b0),
        .O(imm_d[7]));
  OBUF \imm_d_OBUF[8]_inst 
       (.I(1'b0),
        .O(imm_d[8]));
  OBUF \imm_d_OBUF[9]_inst 
       (.I(1'b0),
        .O(imm_d[9]));
  OBUF \inst_d_OBUF[0]_inst 
       (.I(1'b0),
        .O(inst_d[0]));
  OBUF \inst_d_OBUF[10]_inst 
       (.I(1'b0),
        .O(inst_d[10]));
  OBUF \inst_d_OBUF[11]_inst 
       (.I(1'b0),
        .O(inst_d[11]));
  OBUF \inst_d_OBUF[12]_inst 
       (.I(1'b0),
        .O(inst_d[12]));
  OBUF \inst_d_OBUF[13]_inst 
       (.I(1'b0),
        .O(inst_d[13]));
  OBUF \inst_d_OBUF[14]_inst 
       (.I(1'b0),
        .O(inst_d[14]));
  OBUF \inst_d_OBUF[15]_inst 
       (.I(1'b0),
        .O(inst_d[15]));
  OBUF \inst_d_OBUF[16]_inst 
       (.I(1'b0),
        .O(inst_d[16]));
  OBUF \inst_d_OBUF[17]_inst 
       (.I(1'b0),
        .O(inst_d[17]));
  OBUF \inst_d_OBUF[18]_inst 
       (.I(1'b0),
        .O(inst_d[18]));
  OBUF \inst_d_OBUF[19]_inst 
       (.I(1'b0),
        .O(inst_d[19]));
  OBUF \inst_d_OBUF[1]_inst 
       (.I(1'b0),
        .O(inst_d[1]));
  OBUF \inst_d_OBUF[20]_inst 
       (.I(1'b0),
        .O(inst_d[20]));
  OBUF \inst_d_OBUF[21]_inst 
       (.I(1'b0),
        .O(inst_d[21]));
  OBUF \inst_d_OBUF[22]_inst 
       (.I(1'b0),
        .O(inst_d[22]));
  OBUF \inst_d_OBUF[23]_inst 
       (.I(1'b0),
        .O(inst_d[23]));
  OBUF \inst_d_OBUF[24]_inst 
       (.I(1'b0),
        .O(inst_d[24]));
  OBUF \inst_d_OBUF[25]_inst 
       (.I(1'b0),
        .O(inst_d[25]));
  OBUF \inst_d_OBUF[26]_inst 
       (.I(1'b0),
        .O(inst_d[26]));
  OBUF \inst_d_OBUF[27]_inst 
       (.I(1'b0),
        .O(inst_d[27]));
  OBUF \inst_d_OBUF[28]_inst 
       (.I(1'b0),
        .O(inst_d[28]));
  OBUF \inst_d_OBUF[29]_inst 
       (.I(1'b0),
        .O(inst_d[29]));
  OBUF \inst_d_OBUF[2]_inst 
       (.I(1'b0),
        .O(inst_d[2]));
  OBUF \inst_d_OBUF[30]_inst 
       (.I(1'b0),
        .O(inst_d[30]));
  OBUF \inst_d_OBUF[31]_inst 
       (.I(1'b0),
        .O(inst_d[31]));
  OBUF \inst_d_OBUF[3]_inst 
       (.I(1'b0),
        .O(inst_d[3]));
  OBUF \inst_d_OBUF[4]_inst 
       (.I(1'b0),
        .O(inst_d[4]));
  OBUF \inst_d_OBUF[5]_inst 
       (.I(1'b0),
        .O(inst_d[5]));
  OBUF \inst_d_OBUF[6]_inst 
       (.I(1'b0),
        .O(inst_d[6]));
  OBUF \inst_d_OBUF[7]_inst 
       (.I(1'b0),
        .O(inst_d[7]));
  OBUF \inst_d_OBUF[8]_inst 
       (.I(1'b0),
        .O(inst_d[8]));
  OBUF \inst_d_OBUF[9]_inst 
       (.I(1'b0),
        .O(inst_d[9]));
  pc my_pc
       (.clk(clk_IBUF_BUFG),
        .nowpc(nowpc_OBUF),
        .rst(rst_IBUF));
  OBUF \nowpc_OBUF[0]_inst 
       (.I(nowpc_OBUF[0]),
        .O(nowpc[0]));
  OBUF \nowpc_OBUF[10]_inst 
       (.I(nowpc_OBUF[10]),
        .O(nowpc[10]));
  OBUF \nowpc_OBUF[11]_inst 
       (.I(nowpc_OBUF[11]),
        .O(nowpc[11]));
  OBUF \nowpc_OBUF[12]_inst 
       (.I(nowpc_OBUF[12]),
        .O(nowpc[12]));
  OBUF \nowpc_OBUF[13]_inst 
       (.I(nowpc_OBUF[13]),
        .O(nowpc[13]));
  OBUF \nowpc_OBUF[14]_inst 
       (.I(nowpc_OBUF[14]),
        .O(nowpc[14]));
  OBUF \nowpc_OBUF[15]_inst 
       (.I(nowpc_OBUF[15]),
        .O(nowpc[15]));
  OBUF \nowpc_OBUF[16]_inst 
       (.I(nowpc_OBUF[16]),
        .O(nowpc[16]));
  OBUF \nowpc_OBUF[17]_inst 
       (.I(nowpc_OBUF[17]),
        .O(nowpc[17]));
  OBUF \nowpc_OBUF[18]_inst 
       (.I(nowpc_OBUF[18]),
        .O(nowpc[18]));
  OBUF \nowpc_OBUF[19]_inst 
       (.I(nowpc_OBUF[19]),
        .O(nowpc[19]));
  OBUF \nowpc_OBUF[1]_inst 
       (.I(nowpc_OBUF[1]),
        .O(nowpc[1]));
  OBUF \nowpc_OBUF[20]_inst 
       (.I(nowpc_OBUF[20]),
        .O(nowpc[20]));
  OBUF \nowpc_OBUF[21]_inst 
       (.I(nowpc_OBUF[21]),
        .O(nowpc[21]));
  OBUF \nowpc_OBUF[22]_inst 
       (.I(nowpc_OBUF[22]),
        .O(nowpc[22]));
  OBUF \nowpc_OBUF[23]_inst 
       (.I(nowpc_OBUF[23]),
        .O(nowpc[23]));
  OBUF \nowpc_OBUF[24]_inst 
       (.I(nowpc_OBUF[24]),
        .O(nowpc[24]));
  OBUF \nowpc_OBUF[25]_inst 
       (.I(nowpc_OBUF[25]),
        .O(nowpc[25]));
  OBUF \nowpc_OBUF[26]_inst 
       (.I(nowpc_OBUF[26]),
        .O(nowpc[26]));
  OBUF \nowpc_OBUF[27]_inst 
       (.I(nowpc_OBUF[27]),
        .O(nowpc[27]));
  OBUF \nowpc_OBUF[28]_inst 
       (.I(nowpc_OBUF[28]),
        .O(nowpc[28]));
  OBUF \nowpc_OBUF[29]_inst 
       (.I(nowpc_OBUF[29]),
        .O(nowpc[29]));
  OBUF \nowpc_OBUF[2]_inst 
       (.I(nowpc_OBUF[2]),
        .O(nowpc[2]));
  OBUF \nowpc_OBUF[30]_inst 
       (.I(nowpc_OBUF[30]),
        .O(nowpc[30]));
  OBUF \nowpc_OBUF[31]_inst 
       (.I(nowpc_OBUF[31]),
        .O(nowpc[31]));
  OBUF \nowpc_OBUF[3]_inst 
       (.I(nowpc_OBUF[3]),
        .O(nowpc[3]));
  OBUF \nowpc_OBUF[4]_inst 
       (.I(nowpc_OBUF[4]),
        .O(nowpc[4]));
  OBUF \nowpc_OBUF[5]_inst 
       (.I(nowpc_OBUF[5]),
        .O(nowpc[5]));
  OBUF \nowpc_OBUF[6]_inst 
       (.I(nowpc_OBUF[6]),
        .O(nowpc[6]));
  OBUF \nowpc_OBUF[7]_inst 
       (.I(nowpc_OBUF[7]),
        .O(nowpc[7]));
  OBUF \nowpc_OBUF[8]_inst 
       (.I(nowpc_OBUF[8]),
        .O(nowpc[8]));
  OBUF \nowpc_OBUF[9]_inst 
       (.I(nowpc_OBUF[9]),
        .O(nowpc[9]));
  OBUF \res_d_OBUF[0]_inst 
       (.I(1'b0),
        .O(res_d[0]));
  OBUF \res_d_OBUF[10]_inst 
       (.I(1'b0),
        .O(res_d[10]));
  OBUF \res_d_OBUF[11]_inst 
       (.I(1'b0),
        .O(res_d[11]));
  OBUF \res_d_OBUF[12]_inst 
       (.I(1'b0),
        .O(res_d[12]));
  OBUF \res_d_OBUF[13]_inst 
       (.I(1'b0),
        .O(res_d[13]));
  OBUF \res_d_OBUF[14]_inst 
       (.I(1'b0),
        .O(res_d[14]));
  OBUF \res_d_OBUF[15]_inst 
       (.I(1'b0),
        .O(res_d[15]));
  OBUF \res_d_OBUF[16]_inst 
       (.I(1'b0),
        .O(res_d[16]));
  OBUF \res_d_OBUF[17]_inst 
       (.I(1'b0),
        .O(res_d[17]));
  OBUF \res_d_OBUF[18]_inst 
       (.I(1'b0),
        .O(res_d[18]));
  OBUF \res_d_OBUF[19]_inst 
       (.I(1'b0),
        .O(res_d[19]));
  OBUF \res_d_OBUF[1]_inst 
       (.I(1'b0),
        .O(res_d[1]));
  OBUF \res_d_OBUF[20]_inst 
       (.I(1'b0),
        .O(res_d[20]));
  OBUF \res_d_OBUF[21]_inst 
       (.I(1'b0),
        .O(res_d[21]));
  OBUF \res_d_OBUF[22]_inst 
       (.I(1'b0),
        .O(res_d[22]));
  OBUF \res_d_OBUF[23]_inst 
       (.I(1'b0),
        .O(res_d[23]));
  OBUF \res_d_OBUF[24]_inst 
       (.I(1'b0),
        .O(res_d[24]));
  OBUF \res_d_OBUF[25]_inst 
       (.I(1'b0),
        .O(res_d[25]));
  OBUF \res_d_OBUF[26]_inst 
       (.I(1'b0),
        .O(res_d[26]));
  OBUF \res_d_OBUF[27]_inst 
       (.I(1'b0),
        .O(res_d[27]));
  OBUF \res_d_OBUF[28]_inst 
       (.I(1'b0),
        .O(res_d[28]));
  OBUF \res_d_OBUF[29]_inst 
       (.I(1'b0),
        .O(res_d[29]));
  OBUF \res_d_OBUF[2]_inst 
       (.I(1'b0),
        .O(res_d[2]));
  OBUF \res_d_OBUF[30]_inst 
       (.I(1'b0),
        .O(res_d[30]));
  OBUF \res_d_OBUF[31]_inst 
       (.I(1'b0),
        .O(res_d[31]));
  OBUF \res_d_OBUF[3]_inst 
       (.I(1'b0),
        .O(res_d[3]));
  OBUF \res_d_OBUF[4]_inst 
       (.I(1'b0),
        .O(res_d[4]));
  OBUF \res_d_OBUF[5]_inst 
       (.I(1'b0),
        .O(res_d[5]));
  OBUF \res_d_OBUF[6]_inst 
       (.I(1'b0),
        .O(res_d[6]));
  OBUF \res_d_OBUF[7]_inst 
       (.I(1'b0),
        .O(res_d[7]));
  OBUF \res_d_OBUF[8]_inst 
       (.I(1'b0),
        .O(res_d[8]));
  OBUF \res_d_OBUF[9]_inst 
       (.I(1'b0),
        .O(res_d[9]));
  IBUF rst_IBUF_inst
       (.I(rst),
        .O(rst_IBUF));
endmodule
`ifndef GLBL
`define GLBL
`timescale  1 ps / 1 ps

module glbl ();

    parameter ROC_WIDTH = 100000;
    parameter TOC_WIDTH = 0;

//--------   STARTUP Globals --------------
    wire GSR;
    wire GTS;
    wire GWE;
    wire PRLD;
    tri1 p_up_tmp;
    tri (weak1, strong0) PLL_LOCKG = p_up_tmp;

    wire PROGB_GLBL;
    wire CCLKO_GLBL;
    wire FCSBO_GLBL;
    wire [3:0] DO_GLBL;
    wire [3:0] DI_GLBL;
   
    reg GSR_int;
    reg GTS_int;
    reg PRLD_int;

//--------   JTAG Globals --------------
    wire JTAG_TDO_GLBL;
    wire JTAG_TCK_GLBL;
    wire JTAG_TDI_GLBL;
    wire JTAG_TMS_GLBL;
    wire JTAG_TRST_GLBL;

    reg JTAG_CAPTURE_GLBL;
    reg JTAG_RESET_GLBL;
    reg JTAG_SHIFT_GLBL;
    reg JTAG_UPDATE_GLBL;
    reg JTAG_RUNTEST_GLBL;

    reg JTAG_SEL1_GLBL = 0;
    reg JTAG_SEL2_GLBL = 0 ;
    reg JTAG_SEL3_GLBL = 0;
    reg JTAG_SEL4_GLBL = 0;

    reg JTAG_USER_TDO1_GLBL = 1'bz;
    reg JTAG_USER_TDO2_GLBL = 1'bz;
    reg JTAG_USER_TDO3_GLBL = 1'bz;
    reg JTAG_USER_TDO4_GLBL = 1'bz;

    assign (strong1, weak0) GSR = GSR_int;
    assign (strong1, weak0) GTS = GTS_int;
    assign (weak1, weak0) PRLD = PRLD_int;

    initial begin
	GSR_int = 1'b1;
	PRLD_int = 1'b1;
	#(ROC_WIDTH)
	GSR_int = 1'b0;
	PRLD_int = 1'b0;
    end

    initial begin
	GTS_int = 1'b1;
	#(TOC_WIDTH)
	GTS_int = 1'b0;
    end

endmodule
`endif
