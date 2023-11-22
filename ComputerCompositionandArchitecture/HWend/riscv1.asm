.data
	message:.ascii "hello,world\n"
	
.text
main:
	li a7, 4 #Load Immediate
	la a0, message #Load Address
	ecall
	
exit:
	li a7,10
	ecall
	