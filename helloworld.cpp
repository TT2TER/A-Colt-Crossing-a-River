// VS Code C/C++ 测试代码 "Hello World"
// 由 VSCodeConfigHelper v4.0.8 生成

// 您可以在当前文件夹（工作文件夹）下新建 .cpp 源文件编写代码。

// 按下 Ctrl + F5 编译运行。
// 按下 F5 编译调试。
// 按下 Ctrl + Shift + B 编译。

#include <iostream>
#include <fstream>
#include <string>

int main()
{
    // 在标准输出中打印 "Hello, world!"
    std::cout << "Hello, world!" << std::endl;
    // 文件路径
    std::string filePath = "./example.txt";

    // 尝试打开文件
    std::ifstream inputFile(filePath);

    // 检查文件是否成功打开
    if (!inputFile.is_open())
    {
        std::cerr << "无法打开文件: " << filePath << std::endl;
        return 1; // 返回错误代码
    }

    // 读取文件内容并打印字符串
    std::string line;
    while (std::getline(inputFile, line))
    {
        std::cout << line << std::endl;
    }

    // 关闭文件
    inputFile.close();
    printf("中文测试");
}

// 此文件编译运行将输出 "Hello, world!"。
// 按下 Ctrl + F5 后，你将在弹出的终端窗口中看到这一行字。
// !! 重要提示：请您在编写代码前，确认文件名不含中文或特殊字符。 !!
// 若遇到问题，请联系开发者 guyutongxue@163.com。