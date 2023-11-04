// 求偏序集中的极大元与极小元
#include <iostream>
#include <string>

using namespace std;
#define N 27

int main()
{
    string a, b, ans = "";
    cin >> a >> b;
    int abc[N]={0}, min[N]={0}, max[N] = {0},can_be_max[N],can_be_min[N];
    for(int i=0;i<26;i++)
    {
        can_be_min[i]=1;
        can_be_max[i]=1;
    }
    
    for (int i = 0; i < (int)a.size(); i++)
    {
        if ('a' <= a[i] &&a[i] <= 'z')
            abc[a[i] - 'a'] = 1;
    }
    
    // for(int i=0;i<26;i++)
    // {
    //     std::cout<<abc[i]<<" ";
    // }

    for (int i = 0; i < int(b.size()); i++)
    {
        if (b[i] == '<')
        {
            if (abc[b[i + 1] - 'a']==1)
            {
                if (max[b[i + 1] - 'a'] == 1)
                {
                    can_be_max[b[i + 1] - 'a'] = 0;
                    max[b[i + 1] - 'a'] = 0;
                }
                if (min[b[i + 1] - 'a'] == 0&&can_be_min[b[i + 1] - 'a']==1)
                {
                    min[b[i + 1] - 'a'] = 1;
                    can_be_max[b[i + 1] - 'a'] = 0;
                }
            }
            if (abc[b[i + 3] - 'a']==1)
            {
                if (max[b[i + 3] - 'a'] == 0&&can_be_max[b[i + 3] - 'a']==1)
                {
                    max[b[i + 3] - 'a'] = 1;
                    can_be_min[b[i + 3] - 'a'] = 0;
                }
                if (min[b[i + 3] - 'a'] == 1)
                {
                    can_be_min[b[i + 3] - 'a'] = 0;
                    min[b[i + 3] - 'a'] = 0;
                }
            }
        }
    }
    for(int i=0;i<26;i++)
    {
        if(abc[i]==1&&can_be_max[i]==1&&can_be_min[i]==1)
        {
            max[i]=1;
            min[i]=1;
        }
            
    }
    for (int i = 0; i < 26; i++)
    {
        if (min[i] == 1)
        {
            // std::cout<<i<<endl;
            ans += (char)(i + 'a');
            ans +=",";
        }
    }
    ans.pop_back();
    std::cout << ans ;
    std::cout<<endl;
    ans = "";
    for (int i = 0; i < 26; i++)
    {
        if (max[i] == 1)
        {
            ans += (char)(i + 'a');
            ans +=",";
        }
    }
    ans.pop_back();
    std::cout << ans << endl;

    return 0;
}