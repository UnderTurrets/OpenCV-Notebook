

@[TOC](文章目录)


---

#  1.基本操作
##  1.1 tensor的dtype类型
| 代码 |含义  |
|--|--|
|float32|32位float|
|float|floa|
|float64|64位float|
|double|double|
|float16|16位float|
|bfloat16|比float范围大但精度低
|int8|8位int|
|int16|16位int|
|short|short|
|int32|32位int|
|int|int|
|int64|64位int|
|long|long|
|complex32|32位complex|
|complex64|64位complex|
|cfloat|complex float|
|complex128|128位complex float|
|cdouble|complex double|
##  1.2 创建tensor（建议写出参数名字）
###  1.2.1 空tensor（无用数据填充）
####  API
```python
@overload
def empty(size: Sequence[Union[_int, SymInt]], *, memory_format: Optional[memory_format]=None, out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def empty(*size: _int, memory_format: Optional[memory_format]=None, out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def empty(size: _size, *, names: Optional[Sequence[Union[str, ellipsis, None]]], memory_format: Optional[memory_format]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def empty(*size: _int, names: Optional[Sequence[Union[str, ellipsis, None]]], memory_format: Optional[memory_format]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
```
size：[行数,列数]

dtype(deepth type)：数据类型

device：选择运算设备

requires_grad：是否进行自动求导，默认为False
####  示例

```python
       gpu=torch.device("cuda")
       empty_tensor=torch.empty(size=[3,4],device=gpu,requires_grad=True)
       print(empty_tensor)
```
输出

```python
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]], device='cuda:0', requires_grad=True)
```
###  1.2.2 全一tensor
```python
@overload
def ones(size: _size, *, names: Optional[Sequence[Union[str, ellipsis, None]]], dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def ones(*size: _int, names: Optional[Sequence[Union[str, ellipsis, None]]], dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def ones(size: Sequence[Union[_int, SymInt]], *, out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def ones(*size: _int, out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
```
size：[行数,列数]

dtype(deepth type)：数据类型

device：选择运算设备

requires_grad：是否进行自动求导，默认为False

###  1.2.3 全零tensor

```python
@overload
def zeros(size: _size, *, names: Optional[Sequence[Union[str, ellipsis, None]]], dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def zeros(*size: _int, names: Optional[Sequence[Union[str, ellipsis, None]]], dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def zeros(size: Sequence[Union[_int, SymInt]], *, out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def zeros(*size: _int, out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
```
###  1.2.4 随机值[0,1)的tensor

```python
@overload
def rand(size: _size, *, generator: Optional[Generator], names: Optional[Sequence[Union[str, ellipsis, None]]], dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def rand(*size: _int, generator: Optional[Generator], names: Optional[Sequence[Union[str, ellipsis, None]]], dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def rand(size: _size, *, generator: Optional[Generator], out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def rand(*size: _int, generator: Optional[Generator], out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def rand(size: _size, *, out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def rand(*size: _int, out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def rand(size: _size, *, names: Optional[Sequence[Union[str, ellipsis, None]]], dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def rand(*size: _int, names: Optional[Sequence[Union[str, ellipsis, None]]], dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...

```
###  1.2.5 随机值为整数且规定上下限的tensor
####  API
```python
@overload
def randint(low: _int, high: _int, size: _size, *, generator: Optional[Generator]=None, dtype: Optional[_dtype]=None, device: Device=None, requires_grad: _bool=False) -> Tensor: ...
@overload
def randint(high: _int, size: _size, *, generator: Optional[Generator]=None, dtype: Optional[_dtype]=None, device: Device=None, requires_grad: _bool=False) -> Tensor: ...
@overload
def randint(high: _int, size: _size, *, generator: Optional[Generator], out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def randint(low: _int, high: _int, size: _size, *, generator: Optional[Generator], out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def randint(high: _int, size: _size, *, out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def randint(low: _int, high: _int, size: _size, *, out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
```
#### 示例

```python
   int_tensor=torch.randint(low=0,high=20,size=[5,6],device=gpu)
   print(int_tensor)
```
输出

```python
tensor([[18,  0, 14,  7, 18, 14],
        [17,  0,  2,  0,  0,  3],
        [16, 17,  5, 15,  1, 14],
        [ 7, 12,  8,  6,  4, 11],
        [12,  4,  7,  5,  3,  3]], device='cuda:0')
```
###  1.2.6 随机值均值0方差1的tensor

```python
@overload
def randn(size: _size, *, generator: Optional[Generator], names: Optional[Sequence[Union[str, ellipsis, None]]], dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def randn(*size: _int, generator: Optional[Generator], names: Optional[Sequence[Union[str, ellipsis, None]]], dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def randn(size: _size, *, generator: Optional[Generator], out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def randn(*size: _int, generator: Optional[Generator], out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def randn(size: _size, *, out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def randn(*size: _int, out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def randn(size: _size, *, names: Optional[Sequence[Union[str, ellipsis, None]]], dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def randn(*size: _int, names: Optional[Sequence[Union[str, ellipsis, None]]], dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
```
###  1.2.7 列表或numpy数组转为tensor
####  API
```python
def tensor(data: Any, dtype: Optional[_dtype]=None, device: Device=None, requires_grad: _bool=False) -> Tensor: ...
```
####  示例

```python
   sequence_tensor1=torch.tensor(np.array([[[1,2,3],
                                            [4,5,6]],

                                           [[9,8,7],
                                            [6,5,4]]]),
                                 dtype=torch.float,device=gpu,requires_grad=True)
   print(sequence_tensor1)
   sequence_tensor2=torch.tensor([[[1,2,3],
                                   [4,5,6]],

                                  [[9,8,7],
                                   [6,5,4]]],
                                 dtype=torch.float,device=gpu,requires_grad=True)
   print(sequence_tensor2)
```
输出

```python
tensor([[[1., 2., 3.],
         [4., 5., 6.]],

        [[9., 8., 7.],
         [6., 5., 4.]]], device='cuda:0', requires_grad=True)
tensor([[[1., 2., 3.],
         [4., 5., 6.]],

        [[9., 8., 7.],
         [6., 5., 4.]]], device='cuda:0', requires_grad=True)
```
##  1.3 tensor常用成员函数和成员变量
###  1.3.1 item函数

```python
    def item(self): # real signature unknown; restored from __doc__
        """
        item() -> number
        
        Returns the value of this tensor as a standard Python number. This only works
        for tensors with one element. For other cases, see :meth:`~Tensor.tolist`.
        
        This operation is not differentiable.
        
        Example::
        
            >>> x = torch.tensor([1.0])
            >>> x.item()
            1.0
        """
        return 0
```

 - 如果tensor只有一个元素，就返回它的值
 - 如果tensor有多个元素，抛出ValueError
###  1.3.2 转为numpy数组

```python
    def numpy(self, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
        """
        numpy(*, force=False) -> numpy.ndarray
        
        Returns the tensor as a NumPy :class:`ndarray`.
        
        If :attr:`force` is ``False`` (the default), the conversion
        is performed only if the tensor is on the CPU, does not require grad,
        does not have its conjugate bit set, and is a dtype and layout that
        NumPy supports. The returned ndarray and the tensor will share their
        storage, so changes to the tensor will be reflected in the ndarray
        and vice versa.
        
        If :attr:`force` is ``True`` this is equivalent to
        calling ``t.detach().cpu().resolve_conj().resolve_neg().numpy()``.
        If the tensor isn't on the CPU or the conjugate or negative bit is set,
        the tensor won't share its storage with the returned ndarray.
        Setting :attr:`force` to ``True`` can be a useful shorthand.
        
        Args:
            force (bool): if ``True``, the ndarray may be a copy of the tensor
                       instead of always sharing memory, defaults to ``False``.
        """
        pass

```

 - 只有在CPU上运算的tensor才可以转为numpy数组

###  1.3.3 获取形状

```python
    def size(self, dim=None): # real signature unknown; restored from __doc__
        """
        size(dim=None) -> torch.Size or int
        
        Returns the size of the :attr:`self` tensor. If ``dim`` is not specified,
        the returned value is a :class:`torch.Size`, a subclass of :class:`tuple`.
        If ``dim`` is specified, returns an int holding the size of that dimension.
        
        Args:
          dim (int, optional): The dimension for which to retrieve the size.
        
        Example::
        
            >>> t = torch.empty(3, 4, 5)
            >>> t.size()
            torch.Size([3, 4, 5])
            >>> t.size(dim=1)
            4
        """
        pass
```
###  1.3.4 改变形状（非transpose）

```python
    def view(self, *shape): # real signature unknown; restored from __doc__
        """
        Example::
        
            >>> x = torch.randn(4, 4)
            >>> x.size()
            torch.Size([4, 4])
            >>> y = x.view(16)
            >>> y.size()
            torch.Size([16])
            >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
            >>> z.size()
            torch.Size([2, 8])
        
            >>> a = torch.randn(1, 2, 3, 4)
            >>> a.size()
            torch.Size([1, 2, 3, 4])
            >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
            >>> b.size()
            torch.Size([1, 3, 2, 4])
            >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
            >>> c.size()
            torch.Size([1, 3, 2, 4])
            >>> torch.equal(b, c)
            False 
        """
        return _te.Tensor(*(), **{})
```

 - 先把数据变成一维数组，然后再转换成指定形状，**前后的乘积必须相等**
###  1.3.5 transpose转置
####  矩阵转置

```python
    def t(self): # real signature unknown; restored from __doc__
        """
        t() -> Tensor
        
        See :func:`torch.t`
        """
        return _te.Tensor(*(), **{})
```
####  tensor转置

```python
    def transpose(self, dim0, dim1): # real signature unknown; restored from __doc__
        """
        transpose(dim0, dim1) -> Tensor
        
        See :func:`torch.transpose`
        """
        return _te.Tensor(*(), **{})
```

 - 把维度0和维度1进行转置
####  tensor多维度同时转置

```python
    def permute(self, *dims): # real signature unknown; restored from __doc__
        """
        permute(*dims) -> Tensor
        
        See :func:`torch.permute`
        """
        return _te.Tensor(*(), **{})
```

 - 把要转置的维度放到对应位置上，比如对于三维tensor，x、y、z分别对应0、1、2，如果想要转置x轴和y轴，则输入2、1、0即可
####  示例

```python
   sequence_tensor=torch.tensor(np.array([[[1,2,3],
                                            [4,5,6]],

                                           [[9,8,7],
                                            [6,5,4]]]),
                                 dtype=torch.float,device=gpu,requires_grad=True)
   print(sequence_tensor)
   sequence_tensor_permute=sequence_tensor.permute(2,1,0)
   print(sequence_tensor_permute)
   sequence_tensor_transpose=sequence_tensor.transpose(0,2)
   print(sequence_tensor_transpose)
```
输出

```python
tensor([[[1., 2., 3.],
         [4., 5., 6.]],

        [[9., 8., 7.],
         [6., 5., 4.]]], device='cuda:0', requires_grad=True)
tensor([[[1., 9.],
         [4., 6.]],

        [[2., 8.],
         [5., 5.]],

        [[3., 7.],
         [6., 4.]]], device='cuda:0', grad_fn=<PermuteBackward0>)
tensor([[[1., 9.],
         [4., 6.]],

        [[2., 8.],
         [5., 5.]],

        [[3., 7.],
         [6., 4.]]], device='cuda:0', grad_fn=<TransposeBackward0>)
```
可以看到两者的效果是一样的
###  1.3.6 获取维度个数

```python
    def dim(self): # real signature unknown; restored from __doc__
        return 0
```

 - 返回一个int表示维度个数
###  1.3.7 浅拷贝、深拷贝

####  detach函数浅拷贝
假设有模型A和模型B，我们需要将A的输出作为B的输入，但训练时我们只训练模型B. 那么可以这样做：

```python
input_B = output_A.detach()
```

它可以使两个计算图的梯度传递断开，从而实现我们所需的功能。

返回一个新的tensor，新的tensor和原来的tensor共享数据内存，但不涉及梯度计算，即requires_grad=False。修改其中一个tensor的值，另一个也会改变，因为是共享同一块内存。
####  detach函数+转numpy数组深拷贝

```python
   sequence_tensor=torch.tensor(np.array([[[1,2,3],
                                            [4,5,6]],

                                           [[9,8,7],
                                            [6,5,4]]]),
                                 dtype=torch.float,requires_grad=True,device=gpu)
   sequence_tensor_deepCp=torch.tensor(sequence_tensor.to(cpu).detach().numpy())
   sequence_tensor_deepCp+=1
   print(sequence_tensor)
   print(sequence_tensor_deepCp)
```
输出

```python
tensor([[[1., 2., 3.],
         [4., 5., 6.]],

        [[9., 8., 7.],
         [6., 5., 4.]]], device='cuda:0', requires_grad=True)
tensor([[[ 2.,  3.,  4.],
         [ 5.,  6.,  7.]],

        [[10.,  9.,  8.],
         [ 7.,  6.,  5.]]])
```

###  1.3.8 数学运算

```python
    def mean(self, dim=None, keepdim=False, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
        """
        mean(dim=None, keepdim=False, *, dtype=None) -> Tensor
        
        See :func:`torch.mean`
        """
        pass
    def sum(self, dim=None, keepdim=False, dtype=None): # real signature unknown; restored from __doc__
        """
        sum(dim=None, keepdim=False, dtype=None) -> Tensor
        
        See :func:`torch.sum`
        """
        return _te.Tensor(*(), **{})
    def median(self, dim=None, keepdim=False): # real signature unknown; restored from __doc__
        """
        median(dim=None, keepdim=False) -> (Tensor, LongTensor)
        
        See :func:`torch.median`
        """
        pass
    def mode(self, dim=None, keepdim=False): # real signature unknown; restored from __doc__
        """
        mode(dim=None, keepdim=False) -> (Tensor, LongTensor)
        
        See :func:`torch.mode`
        """
        pass
    def norm(self, p="fro", dim=None, keepdim=False, dtype=None):
        r"""See :func:`torch.norm`"""
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.norm, (self,), self, p=p, dim=dim, keepdim=keepdim, dtype=dtype
            )
        return torch.norm(self, p, dim, keepdim, dtype=dtype)
    def dist(self, other, p=2): # real signature unknown; restored from __doc__
        """
        dist(other, p=2) -> Tensor
        
        See :func:`torch.dist`
        """
        return _te.Tensor(*(), **{})
    def std(self, dim, unbiased=True, keepdim=False): # real signature unknown; restored from __doc__
        """
        std(dim, unbiased=True, keepdim=False) -> Tensor
        
        See :func:`torch.std`
        
        .. function:: std(unbiased=True) -> Tensor
           :noindex:
        
        See :func:`torch.std`
        """
        return _te.Tensor(*(), **{})
    def var(self, dim, unbiased=True, keepdim=False): # real signature unknown; restored from __doc__
        """
        var(dim, unbiased=True, keepdim=False) -> Tensor
        
        See :func:`torch.var`
        
        .. function:: var(unbiased=True) -> Tensor
           :noindex:
        
        See :func:`torch.var`
        """
        return _te.Tensor(*(), **{})
    def cumsum(self, dim, dtype=None): # real signature unknown; restored from __doc__
        """
        cumsum(dim, dtype=None) -> Tensor
        
        See :func:`torch.cumsum`
        """
        return _te.Tensor(*(), **{})
    def cumprod(self, dim, dtype=None): # real signature unknown; restored from __doc__
        """
        cumprod(dim, dtype=None) -> Tensor
        
        See :func:`torch.cumprod`
        """
        return _te.Tensor(*(), **{})
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/a787ea7238b54a6789e878517372c9c7.jpeg#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/71d9ef6d81b445bda4dbf59331220f23.jpeg#pic_center)


