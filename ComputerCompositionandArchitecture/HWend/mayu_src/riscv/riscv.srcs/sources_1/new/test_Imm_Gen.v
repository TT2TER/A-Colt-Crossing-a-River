`include "macro.vh"

module Imm_Gen_tb;

    reg [31:0] inst;
    wire [31:0] imm;

    Imm_Gen dut (
        .inst(inst),
        .imm(imm)
    );

    initial begin
        inst = 32'h00108093; // addi x1,x1,1
        #10;
        $display("Immediate value: %d", imm);
        
        
        $finish;
    end

endmodule