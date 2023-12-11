`include "macro.vh"

module Reg_File_tb;

    reg clk;
    reg rst;
    reg we;
    reg [4:0] RA1;
    reg [4:0] RA2;
    reg [4:0] WA;
    reg [31:0] WD;
    wire [31:0] RD1;
    wire [31:0] RD2;
    wire [31:0] debug_reg1;
    wire [31:0] debug_reg2;
    wire [31:0] debug_reg3;

    Reg_File dut (
        .clk(clk),
        .rst(rst),
        .we(we),
        .RA1(RA1),
        .RA2(RA2),
        .WA(WA),
        .WD(WD),
        .RD1(RD1),
        .RD2(RD2),
        .debug_reg1(debug_reg1),
        .debug_reg2(debug_reg2),
        .debug_reg3(debug_reg3)
    );

    initial begin
        // Test case 1: Write data and read data to register 1
        clk = 0;
        rst = 1;
        we = 0;
        RA1 = 0;
        RA2 = 0;
        WA = 1;
        WD = 123;
        #10;
        clk = 1;
        rst = 0;
        we = 1;
        #10;
        we = 0;
        RA1 = 1;
        #10;
        $display("Value in register 1: %d", RD1);
        
        $finish;
    end

endmodule