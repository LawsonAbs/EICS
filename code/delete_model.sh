#!/bin/bash
i=1 # 赋初始值
while [ $i -le 100 ] #判断i是否符合条件， -le 表示 Less and Equal
do
echo the current i is :$i # 输出验证当前的i
rm "BERT_base_$i.pt" #删除指定的文件
i=$((i+1)) #对变量i进行加1操作
done #结束while循环
