# 操作系统实验4:复制文件夹
马煜 1120210526

# 运行和结果展示
## 如何运行
```bash
$ cd <your path>/HW4
$ g++ myDirCpy.cpp -o myDirCpy
$ ./myDirCpy <source dir> <target dir> > log.txt
$ cat log.txt # 查看复制过程
```
## 运行结果
测试的源文件夹结构
```txt
.
├── demo.exe
├── hello_link.py -> hello.py
├── hello.py
└── subdir
    └── main.o

1 directory, 4 files

```
以下是一次运行的结果
```bash
❯ cd system/HW4/
❯ g++ myDirCpy.cpp -o myDirCpy
❯ ./myDirCpy input_dir output_dir > log.txt
Copy finished!
```

输出的log内容如下：
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

针对一些特殊情况，
我也做了一些处理，针对目标文件和文件夹已经存在的情况，我做了如下提示处理，该提示会输出到终端：

```bash
Target directory already exists!
Still continue? (y/n)
$ y
File output_dir/hello.py already exists!
Still continue? (y/n)
$ n
File output_dir/subdir/main.o already exists!
Still continue? (y/n)
$ y
Copy finished!
```
测试复制之后的可执行文件和连接文件均可以正常运行和打开

# 设计文档
## 代码结构（伪代码形式呈现）
```cpp
// 定义函数 cpyFile，用于复制文件
void cpyFile(char *source, char *target, int depth)
{
    // 判断 target 是否存在，如果存在，询问是否覆盖
    // 如果不存在，直接复制
    // 读取 source 文件内容，写入 target 文件
    // 输出复制成功的信息
}

// 定义函数 cpyDir，用于复制目录
void cpyDir(char *source, char *target, int depth)
{
    // 打开 source 目录
    // 遍历 source 目录下的所有文件和子目录
    // 如果是子目录，递归调用 cpyDir 函数
    // 如果是文件，调用 cpyFile 函数
    // 如果是链接，调用 symlink
    // 如果是其他类型，写了处理逻辑了，但没有测试
    // 输出复制成功的信息
}

// 主函数
int main(int argc, char *argv[])
{
    // 判断参数是否正确
    // 判断源目录和目标目录是否相同
    // 判断源目录是否存在
    // 如果目标目录不存在，创建目标目录
    // 如果目标目录已存在，询问是否覆盖
    // 调用 cpyDir 函数，开始拷贝
    // 输出拷贝完成的信息
    return 0;
}
```

## 库函数使用和理解
主要查阅man手册在中的定义和例子

參考的链接如下:

[write(3p)](https://man7.org/linux/man-pages/man3/write.3p.html)

[read(3p)](https://man7.org/linux/man-pages/man3/read.3p.html)

[open(3p)](https://man7.org/linux/man-pages/man3/open.3p.html)

[readdir(3)](https://man7.org/linux/man-pages/man3/readdir.3.html)

……

我认为最复杂的部分就是文件复制部分

下面详细介绍一下我是如何实现的

在最一开始我想当然的以为复制的只有文本文件，所以在使用了fopen来打开文件

但是在测试的时候发现，复制的文件中有一些是二进制文件，所以我就改用了open函数

从定义中可以看到，open函数的返回值是一个int,用来判断文件是否成功打开以及标识该文件的文件描述符
     Otherwise, these functions shall return -1 and set
       errno to indicate the error. If -1 is returned, no files shall be
       created or modified.

```cpp
int open(const char *path, int oflag, ...); 
```
在文档中，很清楚地说明了oflag的含义，打开文件的时候用的是O_RDONLY

而在后面创建文件的时候用的是和源文件一样的权限，所以用的是statbuf.st_mode

在复制的时候新建了一个缓冲区从文件中读取数据，然后写入到目标文件中

```cpp
while ((flag = read(in, buf, 1024)) > 0)//flag用来判断是否读取成功.是读入数据的大小
    {
        if (write(out, buf, flag) != flag)//用来验证是否完全写入成功
        {
            perror("write");
            exit(1);
        }
    }
```
