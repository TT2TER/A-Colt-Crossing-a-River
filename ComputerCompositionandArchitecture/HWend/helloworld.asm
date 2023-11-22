.data
	message:.ascii "hello,world\n"
	
.text
main:
	la a0, message #Load Address
	jal print
	
exit:#Ã»Õý³£exit
	li a7,10
	ecall
	
print:
	li a7,4
	ecall
	