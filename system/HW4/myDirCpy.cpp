#include <iostream>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#define MAX_PATH 256
void cpyFile(char *source, char *target, int depth)
{
    // //读取source文件内容，如果target存在，询问是否覆盖
    // //如果不存在，直接复制
    // struct stat statbuf;
    // if(stat(target,&statbuf)==0)
    // {
    //     std::cout<<"File "<<target<<" already exists!"<<std::endl;
    //     std::cout<<"Still continue? (y/n)"<<std::endl;
    //     char c;
    //     std::cin>>c;
    //     if(c!='y')
    //         return;
    // }
    // FILE *fp1=fopen(source,"r");
    // FILE *fp2=fopen(target,"w");
    // char c;
    // while((c=fgetc(fp1))!=EOF)
    //     fputc(c,fp2);
    // fclose(fp1);
    // fclose(fp2);

    // 以上只考虑了文本文件的复制
    int in, out, flag;
    char buf[1024];
    struct stat statbuf;
    if (stat(target, &statbuf) == 0)
    {
        std::cerr << "File " << target << " already exists!" << std::endl;
        std::cerr << "Still continue? (y/n)" << std::endl;
        char c;
        std::cin >> c;
        if (c != 'y')
            return;
    }
    stat(source, &statbuf);
    in = open(source, O_RDONLY);
    if (in == -1)
    {
        perror("open");
        exit(1);
    }
    out = creat(target, statbuf.st_mode);
    if (out == -1)
    {
        perror("creat");
        exit(1);
    }
    while ((flag = read(in, buf, 1024)) > 0)
    {
        if (write(out, buf, flag) != flag)
        {
            perror("write");
            exit(1);
        }
    }
    if (flag == -1)
    {
        perror("read");
        exit(1);
    }
    for (int i = 0; i < depth; i++)
        std::cout << "----";
    std::cout << "File " << source << " copied to " << target << std::endl;
    close(in);
    close(out);
}
void cpyDir(char *source, char *target, int depth)
{
    struct stat statbuf;
    struct dirent *direntbuf;
    DIR *dir;
    char source_name[MAX_PATH];
    char target_name[MAX_PATH];
    dir = opendir(source);
    while ((direntbuf = readdir(dir)) != NULL)
    {
        memset(source_name, 0, sizeof(source_name));
        memset(target_name, 0, sizeof(target_name));
        // strcpy(source_name, "/home/user/source");
        // strcat(source_name, "/"); // 此时 source_name 为 "/home/user/source/"
        strcpy(source_name, source);
        strcat(source_name, "/");
        strcpy(target_name, target);
        strcat(target_name, "/");

        if (direntbuf->d_type == DT_DIR)
        {

            if (strcmp(direntbuf->d_name, ".") != 0 && strcmp(direntbuf->d_name, "..") != 0)
            {
                for (int i = 0; i < depth; i++)
                    std::cout << "----";
                std::cout << "Copying " << direntbuf->d_name << " ..." << std::endl;
                strcat(source_name, direntbuf->d_name);
                strcat(target_name, direntbuf->d_name);
                stat(source_name, &statbuf);
                mkdir(target_name, statbuf.st_mode);
                cpyDir(source_name, target_name, depth + 1);
            }
        }
        else if (direntbuf->d_type == DT_REG)
        {
            for (int i = 0; i < depth; i++)
                std::cout << "----";
            std::cout << "Copying " << direntbuf->d_name << " ..." << std::endl;
            strcat(source_name, direntbuf->d_name);
            strcat(target_name, direntbuf->d_name);
            cpyFile(source_name, target_name, depth);
        }
        else if (direntbuf->d_type == DT_LNK)
        {
            for (int i = 0; i < depth; i++)
                std::cout << "----";
            std::cout << "Copying " << direntbuf->d_name << " ..." << std::endl;
            strcat(source_name, direntbuf->d_name);
            strcat(target_name, direntbuf->d_name);
            // 判斷連接是否已經存在
            if (stat(target_name, &statbuf) == 0)
            {
                std::cerr << "Link " << target_name << " already exists!" << std::endl;
                std::cerr << "Still continue? (y/n)" << std::endl;
                char c;
                std::cin >> c;
                if (c != 'y')
                    return;
            }
            char buffer[MAX_PATH];
            memset(buffer, '\0', sizeof(buffer));
            readlink(source_name, buffer, sizeof(buffer));
            symlink(buffer, target_name);
            for (int i = 0; i < depth; i++)
                std::cout << "----";
            std::cout << "Link " << source_name << " copied to " << target_name << std::endl;
        }
        else if (direntbuf->d_type == DT_FIFO)
        {
            strcat(source_name, direntbuf->d_name);
            strcat(target_name, direntbuf->d_name);
            mkfifo(target_name, 0666);
        }
        else if (direntbuf->d_type == DT_BLK)
        {
            strcat(source_name, direntbuf->d_name);
            strcat(target_name, direntbuf->d_name);
            mknod(target_name, S_IFBLK, 0);
        }
        else if (direntbuf->d_type == DT_CHR)
        {
            strcat(source_name, direntbuf->d_name);
            strcat(target_name, direntbuf->d_name);
            mknod(target_name, S_IFCHR, 0);
        }
        else if (direntbuf->d_type == DT_SOCK)
        {
            strcat(source_name, direntbuf->d_name);
            strcat(target_name, direntbuf->d_name);
            mknod(target_name, S_IFSOCK, 0);
        }
        else if (direntbuf->d_type == DT_UNKNOWN)
        {
            strcat(source_name, direntbuf->d_name);
            strcat(target_name, direntbuf->d_name);
            mknod(target_name, S_IFREG, 0);
        }
        else
        {
            for (int i = 0; i < depth; i++)
                std::cout << "----";
            std::cout << "Unknown file type!" << std::endl;
            continue;
        }
    }
    for (int i = 0; i < depth-1; i++)
        std::cout << "----";
    std::cout << "Directory " << source << " copied to " << target << std::endl;
}

int main(int argc, char *argv[])
{

    // argv[1] is the source directory
    // argv[2] is the target directory
    if (argc != 3)
    {
        std::cerr << "Wrong number of arguments!" << std::endl;
        return 0;
    }
    if (argv[1] == argv[2])
    {
        std::cerr << "Source and target directories are the same!" << std::endl;
        return 0;
    }
    if (opendir(argv[1]) == NULL)
    {
        std::cerr << "No such source directory!" << std::endl;
        return 0;
    }
    if (opendir(argv[2]) == NULL)
    {
        struct stat statbuf;
        stat(argv[1], &statbuf);
        std::cout << "Creating target directory..." << std::endl;
        mkdir(argv[2], statbuf.st_mode); // st_mode中包含文件权限信息
        std::cout << "Target directory " << argv[2] << " created!" << std::endl;
    }
    else
    {
        std::cerr << "Target directory already exists!" << std::endl;
        std::cerr << "Still continue? (y/n)" << std::endl;
        char c;
        std::cin >> c;
        if (c != 'y')
            return 0;
    }
    // 开始拷贝
    int depth = 1;
    std::cout << "Copying " << argv[1] << " ..." << std::endl;
    cpyDir(argv[1], argv[2], depth);
    std::cout << "Copy finished!" << std::endl;
    std::cerr << "Copy finished!" << std::endl;
    return 0;
}
