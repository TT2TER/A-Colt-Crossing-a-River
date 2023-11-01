#include <iostream>
#include <cmath>
#include <string>
#include <cstring>
using namespace std;
int OperatorRank(char a)
{
    switch (a)
    {
    case '(':
        return 0;
    case '+':
        return 1;
    case '-':
        return 2;
    case '&':
        return 3;
    case '|':
        return 4;
    case '!':
        return 5;
    case ')':
        return 6;
    }
}
bool is_in(char a, string x)
{
    for (unsigned int i = 0; i < x.size(); i++)
    {
        if (x[i] == a)
            return true;
    }
    return false;
}
string evalRPN(string &input)
{
    string stack, output; // stack是运算符栈，output是输出字符串
    int b = 0;
    for (unsigned int i = 0; i < input.size(); i++)
    {
        if (is_in(input[i], "()+-&|!"))
        { // 如果不是字母
            if (input[i] == ')')
            { // 是后括号
                while (stack[stack.size() - 1] != '(')
                { // 将栈中的运算符弹出
                    output += stack[stack.size() - 1];
                    stack.erase(stack.begin() + stack.size() - 1);
                }
                stack.erase(stack.begin() + stack.size() - 1); // 弹出前括号
            }
            else
            { // 不是后括号
                while ((stack.size() != 0) && OperatorRank((stack[stack.size() - 1])) >= OperatorRank(input[i]) && input[i] != '(')
                { // 按照优先级弹出栈中的运算符
                    output += stack[stack.size() - 1];
                    stack.erase(stack.begin() + stack.size() - 1);
                }
                stack += input[i]; // 将当前运算符压入栈中
            }
        }
        else
            output += input[i]; // 如果是字母，直接输出
    }
    while (stack.size() != 0)
    { // 将栈中剩余的运算符弹出
        output += stack[stack.size() - 1];
        stack.erase(stack.begin() + stack.size() - 1);
    }
    // cout<<output<<endl;
    return output;
}
string GetElements(string a)
{
    string b = "";
    for (unsigned int i = 0; i < a.size(); i++)
    {
        if (a[i] >= 'a' && a[i] <= 'z' && !is_in(a[i], b))
        {
            b += a[i];
        }
    }
    for (unsigned int x = 0; x < b.size(); x++)
    {
        for (unsigned int y = 0; y < b.size() - x - 1; y++)
        {
            if (a[y] > a[y + 1])
            {
                char c = a[y];
                a[y] = a[y + 1];
                a[y + 1] = c;
            }
        }
    }
    // cout<<b<<endl;
    return b;
}
int cal(int a, int b, char c)
{
    if (c == '+')
    {
        if (a == b)
            return 1;
        else
            return 0;
    }
    else if (c == '-')
    {
        if (a == 1 && b == 0)
            return 0;
        else
            return 1;
    }
    else if (c == '|')
    {
        if (a + b >= 1)
            return 1;
        else
            return 0;
    }
    else if (c == '&')
    {
        return a * b;
    }
    else if (c == '!')
    {
        if (a == 1)
            return 0;
        else
            return 1;
    }
}
int Calculate(string a)
{ // 计算该二进制的真值
    while (a.size() > 1)
    {
        for (unsigned int i = 0; i < a.size(); i++)
        {
            if (is_in(a[i], "+|&-!"))
            {
                if (a[i] != '!')
                {
                    a[i - 2] = char(cal(int(a[i - 2] - '0'), int(a[i - 1] - '0'), a[i]) + '0');
                    a.erase(a.begin() + i - 1);
                    a.erase(a.begin() + i - 1);
                    break;
                }
                else
                {
                    a[i - 1] = char(cal(int(a[i - 1] - '0'), 0, a[i]) + '0');
                    a.erase(a.begin() + i);
                    break;
                }
            }
        }
    }
    // cout<<int(a[0] - '0')<<endl;
    return int(a[0] - '0');
}
bool is_in2(int a, int *x, int d)
{
    for (int i = 0; i < d; i++)
    {
        if (x[i] == a)
            return true;
    }
    return false;
}
bool is_in3(int n, int i0, int *x, int d)
{
    for (int i = i0 + 1; i < n; i++)
    {
        if (!is_in2(i, x, d))
            return true;
    }
    return false;
}
void PRINT(int *n, string binary, int d)
{
    if (d == 0)
    {
        cout << "0"
             << " ; ";
    }
    for (int i = 0; i < d; i++)
    {
        if ((i != d - 1) && is_in2(n[i], n, d))
        {
            cout << "m" << n[i] << " ∨ ";
        }
        else if (is_in2(n[i], n, d))
        {
            cout << "m" << n[i] << " ; ";
        }
    }
    for (int i = 0; i < pow(2, binary.size()); i++)
    {
        if (!is_in3(pow(2, binary.size()), -1, n, d))
        {
            cout << "1" << endl;
            break;
        }
        else
        {
            if (!is_in2(i, n, d) && is_in3(pow(2, binary.size()), i, n, d))
            {
                cout << "M" << i << " ∧ ";
            }
            else if (!is_in2(i, n, d))
            {
                cout << "M" << i << endl;
            }
        }
    }
}
int find(string variables, char a)
{
    for (int i = 0; i < variables.size(); i++)
    {
        if (variables[i] == a)
            return i;
    }
}
void assign(string variables, string rpn)
{
    string binary = variables; // binary是规格化二进制字符串
    int res[500], true_cont = 0;
    for (int i = 0; i < pow(2, binary.size()); i++)
    {
        int temp = i;
        int length = variables.size();
        for (int j = length - 1; j >= 0; j--)
        {                              // 生成二进制字符串binary
            binary[j] = (temp & 1) + '0'; // 将最低位加到字符 '0'
            temp >>= 1;                   // 右移一位，处理下一个位
        }
        
    //cout<<binary<<endl;
        string num_binary = rpn;
        for (unsigned int i = 0; i < num_binary.size(); i++)
        {
            if (num_binary[i] >= 'a' && num_binary[i] <= 'z')
            {
                num_binary[i] = binary[find(variables, num_binary[i])];
            }
        }

        if (Calculate(num_binary) == 1)
        {
            res[true_cont++] = i;
        }
    }
    PRINT(res, binary, true_cont);
}
int main(void)
{
    string input;
    cin >> input;
    string rpn = evalRPN(input); // 得到逆波兰表达式

    assign(GetElements(rpn), rpn); // 对每个二进制进行赋值
}