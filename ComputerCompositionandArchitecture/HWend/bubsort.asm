.data
list:
	.word 4,2,8,5,7,1
	
.text
bubsort:
	la a0, list 
	li a1, 6  #a1=size

loop1_start:
	li t0,0
	li t1,1
	
loop2_start:
	bge t1,a1,loop1_end
	slli t3,t1,2
	add t3,a0,t3
	lw t4, -4(t3)
	lw t5, 0(t3)
	ble t4,t5,loop2_end
	li t0,1
	sw t4,0(t3)
	sw t5,-4(t3)
	
loop2_end:
	addi t1,t1,1
	jal loop2_start
	
loop1_end:
	bnez t0,loop1_start
	
stop:
	jal stop
	