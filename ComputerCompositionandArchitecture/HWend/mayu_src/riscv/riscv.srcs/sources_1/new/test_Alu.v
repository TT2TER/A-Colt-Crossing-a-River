`include "macro.vh"

module Alu_tb;

    reg [1:0] op;
    reg [31:0] in1;
    reg [31:0] in2;
    wire ZF;
    wire SF;
    wire [31:0] res;

    Alu dut (
        .op(op),
        .in1(in1),
        .in2(in2),
        .ZF(ZF),
        .SF(SF),
        .res(res)
    );

    initial begin
        // Test case 1: Addition
        op = 0;
        in1 = 10;
        in2 = 5;
        #10;
        $display("Result of addition: %d", res);
        
        // Test case 2: Subtraction
        op = 1;
        in1 = 10;
        in2 = 5;
        #10;
        $display("Result of subtraction: %d", res);
        
        // Test case 3: Bitwise OR
        op = 2;
        in1 = 10;
        in2 = 5;
        #10;
        $display("Result of bitwise OR: %d", res);
        
        // Add more test cases here
        
        $finish;
    end

endmodule