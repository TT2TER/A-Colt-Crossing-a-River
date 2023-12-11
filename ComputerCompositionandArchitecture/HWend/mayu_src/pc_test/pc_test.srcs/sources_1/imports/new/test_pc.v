`include "macro.vh"

module pc_tb;

    reg clk;
    reg rst;
    reg [31:0] npc;
    wire [31:0] pc;

    pc dut (
        .clk(clk),
        .rst(rst),
        .npc(npc),
        .pc(pc)
    );

    initial begin
        // Test case 1: Reset
        clk = 0;
        rst = 1;
        npc = 0;
        #10;
        clk = 1;
        #10;
        rst = 0;
        clk=0;
        #10;
        $display("PC value after reset: %d", pc);

        // Test case 2: Update NPC

        npc = 123;
        rst = 0;
        clk = 1;
        #10;
        $display("PC value after updating NPC: %d", pc);

        $finish;
    end

endmodule