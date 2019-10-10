import re
import string
from collections import Counter
import collections
import pandas as pd
import numpy as np


m= 'My name is Janavi'


# def Sorting(lst):
#     lst= lst.split()
#     lst2 = sorted(lst, key=len)
#     return lst2
#

# Driver code

# print(Sorting(lst))

# def order(text):
#     text= text.split()
#     k=[]
#     for i,word in enumerate(text):
#         print(i)
#         k.append(len(i))
#         sorted(k)
#
#
#     #sorted(text, key =len)
#     #text.sort(lambda x, y: len(x) < len(y))
#     text.sort(key=lambda s: len(s))
#     print(text)
#
# l = order(m)

# def detectpair(arr,k):
#     m=[]
#     for i in arr:
#         m.append(k-i)
#     x= Counter(m)
#     y= Counter(arr)
#     z= x & y
#     l= []
#     #print(z)
#     print(z.keys())
#     for i in z.elements():
#         l.append(i)
#     #print(z.items())
#     return l
#
#
# arr= [1,4,14,6,10,-8]
# k=16
# e= detectpair(arr,k)
# print(e)
# def dup(string):
#     m= collections.defaultdict(string)
#     n= m
#     print(m)
#     index=[]
#     l= list(enumerate(string))
#     print(l)
#     # for pos,words in enumerate(string):
#     #     print(words)
#     #     if m-n <=1 :
#     #         index.append(words)
#     #         n=n-1
#     # return index
#
# string= "tree traversal"
# k=dup(string)
# print(k)




# def nextHigher(num):
#     strNum=str(num)
#     length=len(strNum)
#     for i in range(length-2, -1, -1):
#         print(i)
#         current=strNum[i]
#         temp = sorted(strNum[i:])
#         print(temp)
#         print(temp.index(current))
#         next = temp[temp.index(current) + 1]
#         temp.remove(next)
#         print(temp)
#         temp = ''.join(temp)
#         print(temp)
#         print("=========================================")
#         return int(strNum[:i] + next + temp)
#
#
# return num
# right=strNum[i+1]
#         if current<right:
#             print(strNum[i:])
#
# print(nextHigher(12543))
#
def reverseWords1(text):
     #text=text.lower()
     text= text.split()
     return text
print(reverseWords1("MY name is Janavi"))
# #
# def anagrams(string1,string2):
#     str1= reverseWords1(string1)
#     str2= reverseWords1(string2)
#     s1= Counter(str1)
#     s2= Counter(str2)
#     m= s1 & s2
#     print(m)
#     if len(m)== len(str1):
#         print("they are anagrams")
#     else:
#         print("they arent anagrams")
#
# string1= "Eleven plus two"
# string2 ="Twelve plus onl"
# anagrams(string1,string2)
# #
# def reverseWords2(text):
#     print("".join(text.split()[::-1]))
# #
# # # def reverseWords4(text):
# # #     words=text[::-1].split()
# # #     print(" ".join([word[::-1] for word in words]))
# #
# num= 123
# def getLeftHalf(num):
#     return str(num)[:len(str(num))/2]
# m= getLeftHalf(num)
# print(m)

# import functools
# def cnteven(arr):
#     result=0
#     counts= Counter(arr)
#     list1= counts.tolist()
#     for num in list1+ arr :
#         result^=num
#     return result
#     # for i in counts:
#     #     print(i)
#     #     if counts[i]%2 == 0:
#     #         return i
#     #     else:
#     #         continue
#     # return 0
#
# def getEven2(arr):
#     return functools.reduce(lambda x, y: x ^ y, arr + list(set(arr)))
#
# arr=[1,3,3,3, 3,2,5,5,5,5,5]
# m= getEven2(arr)
# print(m)

# def oddele(arr):
#     result=0
#     for num in arr:
#         result^=num
#     return result
#
# arr=[1,1,3,3,3,4,4]
# m= oddele(arr)
# # print(m)
#
# def wordPosition( text, word):
#     #m= Counter(text)
#     #m= text.split()
#     m=text
#     index = collections.defaultdict(list)
#     print(index)
#     count_index=[]
#     for i in text:
#         if m[i]==word:
#             count_index.append(i)
#         else:
#             continue
#     return count_index
#
# text= "My name is Janavi. I am a girl. is fj is you"
# word="is"
# print(wordPosition(text,word))
#
# def getWords(text):
#     return re.sub(r'[^a-z0-9]',' ',text.lower()).split()
#
# def createIndex1(text):
#     index=collections.defaultdict(list)
#     words=getWords(text)
#     for pos, word in enumerate(words):
#         index[word].append(pos)
#     return index
#
# def queryIndex1(index, word):
#     if word in index:
#         return index[word]
#     else:
#         return []
#
# text= "My name is Janavi. I am a girl. is fj is you"
# word="is"
# k= createIndex1(text)
# print(k)
# print(queryIndex1(k,word))