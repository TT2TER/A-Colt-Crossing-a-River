#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <fcntl.h>
#define MAX_PATH 260

void CopyFileDIY(char *source, char *target)
{
    int in, out, flag;
    // 获取文件信息
    struct stat statbuf;
    stat(source, &statbuf);
    // 建立并初始化缓冲区
    char *buffer = NULL;
    buffer = new char[statbuf.st_size];
    memset(buffer, '0', statbuf.st_size);   
    in = open(source, O_RDONLY, S_IRUSR); // 打开文件
    out = creat(target, statbuf.st_mode); // 创建文件
    // 读->缓冲区->写
    while ((flag = read(in, buffer, statbuf.st_size)) > 0)
    {
        write(out, buffer, flag);
    }
    // 释放缓冲区
    delete[] buffer;
    buffer = NULL;
    // 关闭文件
    close(in);
    close(out);
}

void CopyDirectoryDIY(char *source, char *target)
{
    struct stat statbuf;
    struct dirent *direntbuf;
    DIR *dir;
    char src[MAX_PATH]; // 源路径
    char tgt[MAX_PATH]; // 目标路径
    // 打开目录
    dir = opendir(source);
    while ((direntbuf = readdir(dir)) != NULL)
    {
        // 重新初始化两个路径，因为被修改过
        memset(src, '0', sizeof(src));
        memset(tgt, '0', sizeof(tgt));
        strcpy(src, source);
        strcat(src, "/");
        strcpy(tgt, target);
        strcat(tgt, "/");
        // 目录 or 文件 or 符号链接
        if (direntbuf->d_type == DT_DIR)
        {
            // 目录，排除 . 和 ..
            if (strcmp(direntbuf->d_name, ".") != 0 && strcmp(direntbuf->d_name, "..") != 0)
            {
                // 路径 + 名称
                strcat(src, direntbuf->d_name);
                strcat(tgt, direntbuf->d_name);
                // 获取目录信息
                stat(src, &statbuf);
                // 创建新目录
                mkdir(tgt, statbuf.st_mode);
                // 复制目录中文件，递归调用自己
                CopyDirectoryDIY(src, tgt);
            }
        }
        else if (direntbuf->d_type == DT_REG)
        {
            // 文件，路径 + 名称
            strcat(src, direntbuf->d_name);
            strcat(tgt, direntbuf->d_name);
            // 直接复制
            CopyFileDIY(src, tgt);
        }
        else if (direntbuf->d_type == DT_LNK)
        {
            // 符号链接，路径 + 名称
            strcat(src, direntbuf->d_name);
            strcat(tgt, direntbuf->d_name);
            char buffer[MAX_PATH];
            memset(buffer, '\0', sizeof(buffer));
            // 读取源符号链接
            readlink(src, buffer, sizeof(buffer) - 1);
            // 创建新符号链接
            symlink(buffer, tgt);
        }
        else
        {
            // 其他就跳过
            continue;
        }
    }
}

int main(int argc, char *argv[])
{
    struct stat statbuf;
    DIR *dir;
    if (argc != 3)
    {
        printf("参数错误\n");
        exit(1);
    }
    else if ((dir = opendir(argv[1])) == NULL)
    {
        printf("源目录不存在\n");
        exit(1);
    }
    else if ((dir = opendir(argv[2])) == NULL)
    {
        // 获取源目录信息
        stat(argv[1], &statbuf);
        // 创建目标目录
        mkdir(argv[2], statbuf.st_mode);
    }
    else
    {
        printf("目标目录已存在\n");
        exit(1);
    }
    CopyDirectoryDIY(argv[1], argv[2]);
    printf("复制完成，按回车键结束");
    getchar();
    return 0;
}
//https://duck1998.github.io/2019/07/03/OS-homework-4.html