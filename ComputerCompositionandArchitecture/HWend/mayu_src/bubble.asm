.data
list:
 .word 5,1,3,4,2 
.text
 addi a0, x0,0 
 addi a1, x0,20 
loop1_start:
 sub t0,t0,t0
 addi t1,x0,4 
loop2_start:
 blt t1,a1,loop2_body
 jal x1,loop1_end
loop2_body:
 lw t3,-4(t1) 
 lw t4,0(t1)
 blt t3,t4,loop2_end
 sw t3,0(t1)
 sw t4,-4(t1)
 addi t0,x0,1
loop2_end:
 addi t1,t1,4
 jal x1,loop2_start
loop1_end:
 beq t0,x0,stop
 jal x1,loop1_start
stop:
 jal stop