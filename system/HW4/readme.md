# 操作系统实验4:复制文件夹
马煜 1120210526

# 运行方式
```bash
$ cd <your path>/HW4
$ g++ myDirCpy.cpp -o myDirCpy
$ ./myDirCpy <source dir> <target dir> > log.txt
$ cat log.txt # 查看复制过程
```
以下是一次运行的结果
```bash
❯ cd system/HW4/
❯ g++ myDirCpy.cpp -o myDirCpy
❯ ./myDirCpy input_dir output_dir > log.txt
Copy finished!
```

log内容如下：
```txt
Creating target directory...
Target directory output_dir created!
Copying input_dir ...
----Copying hello.py ...
----File input_dir/hello.py copied to output_dir/hello.py
----Copying subdir ...
--------Copying main.o ...
--------File input_dir/subdir/main.o copied to output_dir/subdir/main.o
----Directory input_dir/subdir copied to output_dir/subdir
----Copying hello_link.py ...
----Link input_dir/hello_link.py copied to output_dir/hello_link.py
----Copying demo.exe ...
----File input_dir/demo.exe copied to output_dir/demo.exe
Directory input_dir copied to output_dir
Copy finished!
```
