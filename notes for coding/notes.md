<center><font size = 5>Notes</font></center>
<p align = 'right'>by Nemo</p>
<p align = 'right'>2023.8.25</p>

### tensor.view()
改变矩阵维度  
```python
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

a = a.view(1,6)

print(a.shape)
# torch.Size([1, 6])
```
需要保证改变维度后元素个数不变，不然会报错  
可以用tensor.view(-1)展到一维  

### tensor.shape()
返回矩阵维度，返回值也是张量  
一维tensor的shape就是它的大小

### tensor.sum()
```python
a = torch.ones((2, 3))
a1 = torch.sum(a)
a2 = torch.sum(a, dim=0)
a3 = torch.sum(a, dim=1)
a4 = torch.sum(a, dim=0, keepdim=True)

print(a1)
print(a2)
print(a3)
print(a4)
print(a4.shape)

'''
tensor(6.)
tensor([2., 2., 2.])
tensor([3., 3.])
tensor([[2., 2., 2.]])
torch.Size([1, 3])
'''
```
dim = k就是沿第k维的方向压缩

### tensor运算
https://blog.csdn.net/lijiaming_99/article/details/114642093

### reshape()、transpose()、unsqueeze()
https://blog.csdn.net/weixin_43253464/article/details/121984905?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-121984905-blog-127615586.235%5Ev38%5Epc_relevant_anti_vip_base&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-121984905-blog-127615586.235%5Ev38%5Epc_relevant_anti_vip_base&utm_relevant_index=2

### torch.max()和torch.topk()
https://zhuanlan.zhihu.com/p/488535846  
max()获取值最大的键：https://blog.csdn.net/qq_43657442/article/details/109075004

### python推导式生成列表
```python
[x*y for x in range(1,5) if x > 2 for y in range(1,4) if y < 3]
```
k = 1 got accuracies: [26.3, 25.7, 26.4, 27.8, 26.6]
k = 3 got accuracies: [23.9, 24.9, 24.0, 26.6, 25.4]
k = 5 got accuracies: [24.8, 26.6, 28.0, 29.2, 28.0]
k = 8 got accuracies: [26.2, 28.2, 27.3, 29.0, 27.3]
k = 10 got accuracies: [26.5, 29.6, 27.6, 28.4, 28.0]
k = 12 got accuracies: [26.0, 29.5, 27.9, 28.3, 28.0]
k = 15 got accuracies: [25.2, 28.9, 27.8, 28.2, 27.4]
k = 20 got accuracies: [27.0, 27.9, 27.9, 28.2, 28.5]
k = 50 got accuracies: [27.1, 28.8, 27.8, 26.9, 26.6]
k = 100 got accuracies: [25.6, 27.0, 26.3, 25.6, 26.3]