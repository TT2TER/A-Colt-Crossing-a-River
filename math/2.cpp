#include <stdio.h>

#include <cstring>
#include <iostream>
#include <string>
using namespace std;

#define N 1010
int q[N][30];
int sum0, sum1, sum2;

void init(string input)
{
    memset(q, 0, sizeof(q));
    sum0 = sum1 = -1;
    sum2 = 0;
    int len = input.size();
    for (int i = 0; i <= len; i++)
    {
        if (input[i] >= 'a' && input[i] <= 'z')
        {
            q[sum2][input[i] - 'a'] = 1;
        }
        else if (input[i] == '&')
        {
            sum2++;
        }
        else if (input[i] == '!')
        {
            q[sum2][input[++i] - 'a'] = 2;
        }
    }
}
bool same(int *a, int *b) // 判断两简单析取式是否相同
{
    for (int i = 0; i < 26; i++)
        if (a[i] != b[i])
            return false;
    return true;
}
bool not_in(int *n) // 检查S1，S2中是否有重复
{
    for (int i = 0; i <= sum1; i++)
        if (same(q[i], n))
            return false;
    return true;
}
bool resolve(int *a, int *b)
{
    int num_single = 0;
    int num_double = 0;
    for (int i = 0; i < 26; i++)
    {
        if (!a[i] && !b[i])
        {
            continue; // 没有该变项，跳过
        }
        else if (a[i] + b[i] == 3)
        {
            num_double++;
        }
        else
        {
            num_single++;
        }
    }

    if (num_double != 1)
        return true; // 不能消解or多项可消解

    if (num_single == 0)
        return false; // 空子句

    // 消解
    int c[30];
    memset(c, 0, sizeof(c));
    for (int i = 0; i < 26; i++)
    {
        if ((!a[i] && !b[i]) || (a[i] + b[i] == 3))
            c[i] = 0;
        else if (a[i] == 1 || b[i] == 1)
            c[i] = 1;
        else
            c[i] = 2;
    }
    if (not_in(c))
    {
        sum2++;
        for (int i; i < 26; i++)
        {
            q[sum2][i] = c[i];
        }
    }
    return true;
}
int main()
{
    string input;
    cin >> input;

    init(input);

    do
    {
        sum0 = sum1;
        sum1 = sum2;

        for (int i = 0; i <= sum0; i++)
        {
            for (int j = sum0 + 1; j <= sum1; j++)
            {
                if (!resolve(q[i], q[j]))
                {
                    cout << "NO" << endl;
                    return 0;
                }
            }
        }
        for (int i = sum0 + 1; i <= sum1; i++)
        // s1中每一对自己内部比较
        {
            for (int j = i + 1; j <= sum1; j++)
            {
                if (!resolve(q[i], q[j]))
                {
                    cout << "NO" << endl;
                    return 0;
                }
            }
        }
    } while (sum2 > sum1);
    cout << "YES" << endl;
    return 0;
}