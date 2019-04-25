#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

# 样本转换程序

#原始编码
code = ["1111110","0110000","1101101","1101101", \
        "0110011","1011011","1011111","1110000", \
        "1111111","1111011","1110111","0011111", \
        "1001110","0111101","1001111","1000111"]


#转换成数组
#bitcode = [list(x) for x in code]
bitcode = [list(map(lambda x:int(x),list(x))) for x in code]

#按列输出
out = map(list,zip(*bitcode))
with open('./bitout.txt','w',encoding='utf-8') as f:
    for x in out :
        print(x)
        f.write(str(x))
        f.write('\n')


if __name__ == '__main__':
    pass

