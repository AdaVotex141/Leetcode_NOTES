# Summary
思路是：https://blog.algomooc.com/LeetCode/%E7%AC%AC%E4%B8%80%E5%91%A8/%E7%AC%AC%E4%BA%8C%E5%A4%A9/LeetCode%20485%E3%80%81%E6%9C%80%E5%A4%A7%E8%BF%9E%E7%BB%AD%201%20%E7%9A%84%E4%B8%AA%E6%95%B0.html#%E4%B8%80%E3%80%81%E9%A2%98%E7%9B%AE%E6%8F%8F%E8%BF%B0
跟着这个先做 大概一共200题，然后再跟着leetcode 101刷第二遍= =+
先用C吧现在对C熟一点
---
主要是一个```fast-index```一个```slow-index```的对array的操作
## 485、最大连续 1 的个数
没什么技巧，直接出思路
学到了```fmax()```
## 283、移动零
普通交换= =+
就是如果不是0就不停地往前交换
```c
void moveZeroes(int* nums, int numsSize) {
    int i=0;
    int temp=0;
    for(i=0;i<numsSize;i++)
        if(nums[i]!=0){
            int num_temp=nums[i];
            nums[i]=nums[temp];
            nums[temp]=num_temp;
            temp++;
        }
    }

```
```python
def moveZeros(self, nums:List[int]):
    slow=0
    fast=0
    for fast in range(lens(nums)):
        if nums[fast]!=0:
            nums[slow]=nums[fast]
            slow++
#设置两个指针，一个slow一个fast，然后如果fast不等于0，就交换（画个图就行了
    for i in range(slow,lens(nums)):
        nums[i]=0
```


## 26、删除有序数组中的重复项
```c
int removeDuplicates(int* nums, int numsSize) {
    int i=0;
    int j=0;//Pointing at the place to be 
    for(i=0;i<numsSize;i++){
       if(i==0||nums[i]!=nums[i-1]){
           nums[j]=nums[i];
           j++;
       } 
    }
    return j;
}
```
```python
def removeDuplicates(nums:List[int]):
    fast=0
    slow=0
    for fast in range(lens(nums)):
        if i==0 || nums[fast]!=nums[fast-1]:
            nums[slow]=nums[fast]
            slow++
    return slow
```
其实这个画个图就理解了，大概就是```j```指向被赋值的地方，如果```nums[i]```和前一个不同（跳出了duplicates），就替换什么的
## 27、移除元素
```c
int removeElement(int* nums, int numsSize, int val) {
    int i=0;
    int j=0;
    for(i=0;i<numsSize;i++){
        if(nums[i]!=val){
            nums[j]=nums[i];
            j++;
        }
    }
    return j;
}
```

```python

```
跟上一题很相似，其实这个自己试着历遍一次就理解了，大概就是```j```指向被赋值的地方，一个快一个慢
# Summary
链表只想用java，C老是要分配释放内存很烦的（
1.dummy Node的概念，很多是可以用虚拟头节点来规避一些条件判断
2.recursion，链表题画图会清晰很多的！
3.试了下两道easy题，主要考的就是普通的ADT
## 19、删除链表的倒数第 N 个结点
链表还是java比较熟（
```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        //Create a dummy at the start
        ListNode dummy=new ListNode(0);
        dummy.next=head;
        ListNode fast = head;
        ListNode former = head;
        ListNode latter=dummy;
        for(int i=0;i<n;i++){
            fast=fast.next;
        }
        while(fast!=null){
            fast=fast.next;
            former=former.next;
            latter=latter.next;
        }
        latter.next=former.next;
        return dummy.next;
    }
}
```

```C

/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     struct ListNode *next;
 * };
 */

struct ListNode* removeNthFromEnd(struct ListNode* head, int n) {
    struct ListNode* dummy = (struct ListNode*)malloc(sizeof(struct ListNode));
    dummy->next = head;
    struct ListNode* fast = dummy;
    struct ListNode* slow = dummy;

    // Move the fast pointer n+1 steps ahead
    for (int i = 0; i <= n; i++) {
        fast = fast->next;
    }

    // Move both pointers until the fast pointer reaches the end
    while (fast != NULL) {
        fast = fast->next;
        slow = slow->next;
    }

    // Remove the nth node from the end
    struct ListNode* toRemove = slow->next;
    slow->next = slow->next->next;
    free(toRemove);

    struct ListNode* result = dummy->next;
    free(dummy);

    return result;
}
```
你看C的处理就很麻烦，老是需要分配内存什么的
思路：
1.创建虚拟节点的概念，这样不需要对操作的节点是否为头节点进行判断
2.如果要删的是N节点，那么操作指针要指向前面的那个节点
![](img/2023-11-27-23-33-24.png)
删除的话直接指向该节点.next即可

## 24、两两交换链表中的节点
两两交换的意思就是，比如一共5个，12交换，34交换 5后面是null所以维持原样
考虑用递归做
递归条件是：```head=null```或者```head.next=null```
每次递归从```head.next.next```开始
ListNode temp=head.next;
ListNode head=head.next;
ListNode head.next=temp;
```java
class Solution {
    public ListNode swapPairs(ListNode head) {
       // ListNode temp=new ListNode();
//用递归啊,如果本身就是
        if(head==null || head.next==null){
            return head;
        }
        //递归
        ListNode subhead=swapPairs(head.next.next);
        //创建一个Node指向head.next
        ListNode headnext=head.next;
        //现在headnext的下一个实际上是新的head（链表嘛）
        headnext.next=head;
        //head的下一个应该是subhead（即下一次两两交换的头）
        head.next=subhead;
        //实际上因为递归的是head.next.next
        return headnext;    
    }
}
```
![](img/2023-11-28-11-36-15.png)

```java
/*另一方面的思路：还是虚拟头节点，但总体来说还是
如果要交换两个链表势必要把第三个node搞过来
上一个思路是用的是将下一个node搞过来，这边是将前一个作为中间节点（
思路：https://www.bilibili.com/video/BV1YT411g7br/?spm_id_from=333.337.search-card.all.click&vd_source=dd2dd80e2ed658ce4d2f8ef286463856
*/
ListNode dummy=ListNode();
dummy.next=head;
ListNode cur=dummy;
//这个判断条件也不一样呢，因为用的是前一个节点作为中间节点来进行
while(cur.next!=null && cur.next.next!=null){
    //2.保存住节点1（temp）和节点3（temp3）
    ListNode temp=cur.next;
    ListNode temp1=cur.next.next.next
    //1.cur的下个指向2，但是节点1的指针没有了，所以先在上面加一个
    cur.next=cur.next.next;
    //3.节点2指向节点1
    cur.next.next=temp;
    temp.next=temp1;
    //4.cur节点往前
    cur=cur.next.next.next;
}
return dummy.next;
```
![](img/2023-11-28-13-51-29.png)
dummy->2->1->下一个
## 160、相交链表
一道easy题...好饿（
![](img/2023-11-28-14-07-42.png)
这个是一个规律判断：
让A和B分别以步长为1遍历，如果指到null就A->B接着历遍，B->A接着历遍，如果没有相交的则两个都会指向null
如果有那么会指向该公共元素，所以最后返回pointA就好
果然linked list熟一点（你
有动画理解一点
```python
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        pointA=headA
        pointB=headB
        while pointA!=pointB:
            if pointA==None:
                pointA=headB
            else:
                pointA=pointA.next
            
            if pointB==None:
                pointB=headA
            else:
                pointB=pointB.next
        return pointA

```
## 599、Minimum index Sum of two lists
索引和，一开始没有理解index sum这个东西（
暴力解法就是这样的
```python
class Solution(object):
    def findRestaurant(self, list1, list2):
        #代表一个无限大的数字
        min_mum = float("inf")
        min_name = []
        for i in range(len(list1)):
            n = list1[i]
            #如果在list 2里面，则储存索引和
            if n in list2:
                t = list2.index(n) + i
                #对索引和进行判断
                if t < min_mum:
                    min_mum = t
                    min_name = [n]
                elif t == min_mum:
                    min_name.append(n)
        return min_name
        """
        :type list1: List[str]
        :type list2: List[str]
        :rtype: List[str]
        """
```
当然还可以用harshmap做,创建两个dictionary
如果有类似的就将dictionary对应的索引相加并且进行比较
```python
class Solution:
    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        #idctionary就是harshmap
        dic = {s : i for i, s in enumerate(list1)}
        ans = []
        mid = inf
        for i, s in enumerate(list2):
            #如果s在
            if s in dic:
                j = dic[s]
                if i + j < mid:
                    mid = i + j
                    ans = [s]
                elif i + j == mid:
                    ans.append(s)
        return ans
'''enumerate()：enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)
组合为一个索引序列，同时列出数据和数据下标
一般用在 for 循环当中
'''
```
![](img/2023-11-30-04-07-37.png)
大概复习了下har
## 488、Find All Numbers Disappeared in an Array
做上瘾了！试试
```python
class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        setA=set()
        #最大数字 一个是8 一个是1
        n=(len(nums))
        for i in range(n):
            setA.add(i+1)
        setB=set(nums)
        return setA.difference(setB)
```
十分明显，用的set
学到了```set.difference```和```set.add```两个，python真好用.jpg

## 203、移除链表元素
这个也很好理解了,通过设置头节点来规避如果开头就是val或者一开始就是[]的情况
```python
class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        if head==None:
            return None
        #设置一个虚拟头节点，如果头节点就是val，那么相当于直接删除
        dummy=ListNode(-1)
        dummy.next=head
        pre=dummy
        current=head
        while current!=None:
            if current.val==val:
                pre.next=current.next
            else:
                pre=pre.next
            current=current.next
        return dummy.next

```
# Summary


## 21、合并两个有序链表
```python
class Solution(object):
    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        dummy=ListNode(-1)
        pre=dummy
        #不断比较list1和list2对应值的大小，直到历遍完成
        while list1 and list2:
            if list1.val<=list2.val:
                pre.next=list1
                list1=list1.next
            else:
                pre.next=list2
                list2=list2.next
            pre=pre.next
        pre.next=list1 or list2
        return dummy.next
```
因为不确定最后到底是list1先走完还是list2先走完，所以用的是 or

## 92 、(?????)反转链表 II

```python
class Solution(object):
    def reverseBetween(self, head, left, right):
        """
        :type head: ListNode
        :type left: int
        :type right: int
        :rtype: ListNode
        """
        dummy=ListNode(-1)
        dummy.next=head
        cur=head
        pre=dummy
        # 一般仅仅用于循环n次，不用设置变量
        for _ in range (left-1):
            cur=cur.next
            pre=pre.next
        #实际上是判断整个linked list的终止条件，这道题无法使用指针的方式作为结束条件while（cur！=None这种）
        #所以是不可能直接操作，而是每一次都使用循环来翻转（）
        for _ in range (right-left):
            #i=0:cur=2,pre=1,temp=3
            temp=cur.next
            #1->3->2->4->5
            #2->4
            cur.next=temp.next
            #3->2,所以，这行代码让 temp 的下一节点不是 4 ，而是 2
            temp.next=pre.next
            #1->3 所以，这行代码让 pre 的下一节点为 3
            pre.next=temp

        return dummy.next
```
这个的理解是：先两两交换？
然后：1->3->4->2->5

## 237、删除链表的节点
```python
def deleteNode(node):
    node.val = node.next.val
    node.next = node.next.next
```
画个图就行了x实际上并不是删除，而是直接跳过x

## 328、奇偶链表
```python
class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        #边界情况
        if head==None or head.next==None:
            return head
        odd=head
        even=head.next
        evenHead=even
        while odd.next!=None and even.next!=None:
            odd.next=even.next
            odd=odd.next
            even.next=odd.next
            even=even.next
        odd.next=evenHead
        return head
```
实际上就相当于搞了两条链表出来，一条叫odd，一条叫even，然后最后将odd和even连起来即可
## 876、链表的中间结点
```python
class Solution(object):
    def middleNode(self, head):
        slow = head
        fast = head
        while fast and fast.next:
            fast=fast.next.next
            slow=slow.next
        return slow
```
但是这么写主要是因为给的两个case是1->5和1->6
所以一开始我还在思考怎么知道总数来着
后来问了chagpt才发现：
![](img/2023-12-10-22-59-03.png)
实际上如果链表长度不一样的话还是不同的
但是思路是一样的：快慢指针
# Summary

前两个比较难，目前反转链表II还是有点不太理解
其他更多是画图，下一篇开始新的ADT类型了

## 20、有效的括号

![](img/2023-12-10-23-05-47.png)
```python
class Solution(object):
    def isValid(self, s):
        stack=[]
        mapping = {')': '(', '}': '{', ']': '['}
        for c in s:
            if c in mapping:
                top_element=stack.pop() if stack else '#'
                if mapping[c]!=top_element:
                    return False
            else:
                stack.append(c)
        return not stack
```
使用了dictionary 和 stack，实际上比所谓答案更加简洁
逻辑是：
1.如果是右边的，就加入stack
2.如果是左边的，就弹出stack，然后和dictionary的作对比

这边也是因为嵌套关系用到的
因为是嵌套所以用stack
## 71、简化路径
```python
class Solution(object):
    def simplifyPath(self, path):
        stack=[]
        names = path.split("/")
        for name in names:
            if name=='' or name=='.'  :
                continue
            elif name=='..':
                if stack:
                    stack.pop()
            else:
                stack.append(name)
        return '/'+'/'.join(stack)
```
对string的操作（都说C过时了！）
含“/”的切割
![](img/2023-12-11-13-36-24.png)

## 150、(?????)逆波兰表达式求值
```python
class Solution(object):
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        stack=[]
        for string in tokens:
            if string.isdigit():
                stack.append(int(string))
            else:
                top_element=stack.pop()
                second_element=stack.pop()
                if string== "+":
                    stack.append(top_element+second_element)
                elif string == '-':
                    stack.append(second_element-top_element)
                elif string=="/":
                    stack.append(second_element/top_element)
                elif string=="*":
                    stack.append(second_element*top_elemen)
        return stack[0]
```
一直会出现：
    IndexError: pop from empty list
    second_element=stack.pop()
    的问题
    迷茫了我觉得我的思路是对的，这道题虽然是medium但是本质是理解题意（
    就是debug很痛苦）
    改成了```token not in operators```和```'+', '-', '*', '/'```
就没出现过pop from empty list的问题了为什么（！！！！！！！！！
但是另一个是确实可以直接用这个判断（但是为什么isdigit()反而会出错）
另一个是用了这个方法之后case3 一直error：
![](img/2023-12-11-20-37-25.png)
但是抄了一篇旁边哥们的solutions也是这个error，不是我的问题= =

## 155、最小栈
![](img/2023-12-11-20-49-41.png)
对时间复杂度有要求（所以不能遍历,考虑做一个辅助的minstack
1.push
正常push
如果最小栈是空的，或者当前值小于min_stack最后的值，则加载min_stack的后面
2.pop
正常pop
如果pop的值就是min_stack的最后一个，则min_stack的最后一个也会相应地pop出来

```python
class MinStack(object):

    def __init__(self):
        self.stack=[]
        self.min_stack=[]

    def push(self, val):
        """
        :type val: int
        :rtype: None
        """
        self.stack.append(val)
        #如果 self.min_stack 是空的，或者当前值小于minstack的值，则在stack上加上
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        """
        :rtype: None
        """
        val=self.stack.pop()
        if val==self.min_stack[-1]:
            self.min_stack.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1]
        

    def getMin(self):
        """
        :rtype: int
        """
        return self.min_stack[-1]
        


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```
关于```stack[0]```和```stack[-1]```:
![](2023-12-11-21-04-49.png)

## 1614、括号的最大嵌套深度
```python
class Solution(object):
    def maxDepth(self, s):
        """
        :type s: str
        :rtype: int
        """
        stack=[]
        size=0
        dep=0
        for c in s:
            if c == '(':
                stack.append(c)
                size+=1
                dep=max(size,dep)
            elif c == ')':
                stack.pop()
                size-=1
        return dep
```
dep实际上代表s中(的最大数量
size就是栈中数量
# Summary
主要是stack，后面有一点点linked-list 内容，虽然基本都是直接看淡了（
stack主要是，成对，嵌套的内容基本都是用栈
* 后面两个linkedlist一开始想复杂了
* **字符串解码**这道题主要是自己画图过一遍，然后因为是prev+current*num，所以是xx边界条件，入栈
* **基本计算器**主要是用sign来替代+ -，对于括号是用堆栈，这道题和逆波兰表达式的区别就在于一个是等号在最后面一个是在中间，所以用的是sign来替代+和-
* **最长有效括号**是用索引来进行计算，虽然知道什么意思但是不知道怎么想出来的（最终栈中存储的是未匹配的右括号的索引。遍历栈，计算每个未匹配右括号之间的距离，其中最大的距离即为最长有效括号的长度？？）直到这个解法但是不知道为什么要这么做
* **删除重复元素**，依然是虚拟头结点，如果有一个是重复的，那就往下知道不是这个重复值为止
* **删除重复元素（但是保留一个）**就是是否相同，相同就下一个（
* **两数相加**主要是carry（进位）的问题，然后理解题意和边界条件（l1 l2 和carry任意一个不为0都要继续下去）
* 
## 394、字符串解码
```
输入：s = "3[a]2[bc]"
输出："aaabcbc"

输入：s = "3[a2[c]]"
输出："accaccacc"
```
实例2：
![](2024-02-04-14-58-04.png)
本质是在'[' ']'的时候才进行入栈操作
1. num部分处理
   整数部分是1~300
   使用digit储存
2. 字母部分处理
   字母使用res=“”进行储存
3. '['部分处理
    入栈，储存起来
    stack.append(num,current)
    清空
4. ']'部分处理
   出栈一次，处理完毕
   num,current=stack.pop()
   current=prev+current * num

```py
class Solution(object):
    def decodeString(self,s):
        stack = []
        current_num = 0
        current_str = ""

        for char in s:
            if char.isdigit():
                current_num = current_num * 10 + int(char)
            elif char == '[':
                stack.append((current_num, current_str))
                current_num = 0
                current_str = ""
            elif char == ']':
                num, prev_str = stack.pop()
                current_str = prev_str + current_str * num
            else:
                current_str += char

        return current_str
```
![](img/2024-02-04-16-55-50.png)
对着写一遍就理解了

## 224、基本计算器
括号这种成对出现的肯定是堆栈）
首先不是逆波兰表达式，所以普通的计算不需要用stack，而是使用sign来表示+和-
依然也是用stack来处理内部的
```py
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        stack=[]
        result=0
        sign=1 #track the sign
        i=0
        while i<len(s):
            char=s[i]
            if char.isdigit():
                num=0
                while i<len(s) and s[i].isdigit():
                    num=num*10+int(s[i])
                    i+=1
                result+=sign*num
            elif char =='+':
                sign=1
                i+=1
            elif char=='-':
                sign=-1
                i+=1
            elif char=='(':
                stack.append((result,sign))
                result=0
                sign=1
                i+=1
            elif char==')':
                if stack:
                    result2,sign=stack.pop()
                    result=result+result2*sign
                else:
                    i+=1   
            else:
               i+=1
        return result
```
但是这个pass不了这个case：
![](img/2024-02-04-17-31-22.png)

```py
class Solution:
    def calculate(self, s) :
        stack=[]
        i=0
        sign=1
        result=0
        while i<len(s):
            char=s[i]
            if char==' ':
                i+=1
            elif char.isdigit():
                num=int(s[i])
                while i+1<len(s) and s[i + 1].isdigit():
                    i+=1
                    num=num*10+int(s[i])
                result=result+num*sign
                i+=1
            elif char=='+':
                sign=1
                i+=1
            elif char=='-':
                sign=-1
                i+=1
            elif char=="(":
                stack.append((result,sign))
                result=0
                sign=1
                i+=1
            elif char==')':
                formerresult,former_sign=stack.pop()
                result=formerresult+former_sign*result
                i+=1
        return result

```
改了```result = result2 + sign * result```这一行对了

## LeetCode 32、最长有效括号
括号依然是用stack，但是这里主要是对索引进行操作
![](img/2024-02-04-20-36-00.png)
有几种情况
1.如果是'('则入栈
2.如果是')'  
    出栈
    a. 仅剩一个，就压入
    b.最长长度就是max（max，当前-栈顶的数字）

注意这里如果一开始就是')'可能会导致错误，所以初始化的时候就是stack = [-1]

```py
class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        stack = [-1]  # Initialize the stack with -1
        max_length = 0

        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    max_length = max(max_length, i - stack[-1])

        return max_length
```
https://www.bilibili.com/video/BV1KJ411G7U7/?spm_id_from=333.337.search-card.all.click&vd_source=dd2dd80e2ed658ce4d2f8ef286463856

## LeetCode 82、删除排序链表中的重复元素 II（迭代版）
```py
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head
        current = dummy

        while current.next and current.next.next:
            if current.next.val == current.next.next.val:
                # Skip duplicate nodes
                duplicate_val = current.next.val
                while current.next and current.next.val == duplicate_val:
                    current.next = current.next.next
            else:
                current = current.next

        return dummy.next
```
实际上是

## LeetCode 83、删除排序链表中的重复元素
上一题的另一种形式，但是保留一个
```py
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        current=head
        while current and current.next :
            if current.next.val==current.val:
                current.next=current.next.next

            else:
                current=current.next
        return head
```
比上一题季丹丹
## LeetCode 2、两数相加
主要是进位的问题，然后理解题意是倒着来的

```py
    dummy=ListNode(-1)
    current=dummy
    carry=0

    while l1 and l2 and carry:
        value1=l1.value if l1 else 0
        value2=l2.value if l2 else 0
        total=value1+value2+total
        current.next=ListNode(total%10)
        carry=total//10

        l1=l1.next if l1 else None
        l2=l2.next if l2 else None
    return 
```
![](img/2024-02-04-22-05-46.png)
只要l1，l2，carry中有一个没到0，就是能继续

# Summay

# NEETCODE（150）
![](img/2024-02-07-16-48-23.png)

## Arrays&Hashing
学到了hashmap这个东西
甚至最后两道medium题能自己写出来，主要是痛苦debug，至少至少地有思路了是嘛！
### 217.Contains Duplicate
```py
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        for i in range(len(nums)-1):
            for j in range(i+1,len(nums)):
                if nums[i]==nums[j]:
                    return True  
        return False
``` 
第一反应能写出来，但是Time Limit Exceeded
第二种是sort()然后两两表
```py
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        hashset=set()
        for n in nums:
            if n in hashset:
                return True
            hashset.add(n)
        return False
```
使用hashset来实现，本身就是一个set
### 242.Valid Anagram
一招顺解：
```py
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        return sorted(s)==sorted(t)
```
按顺序排好，确实完全相等的话就是对的
```py
#hashmap
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(t)!=len(s):
            return False
        countS={}
        countT={}
        for i in range(len(s)):
            countS[s[i]]=1+countS.get(s[i],0)
            countT[t[i]]=1+countT.get(t[i],0)
        return countS == countT
```
首先这个东西在python应该叫dictionary，在java里确实是hashmap
![](img/2024-02-06-18-27-59.png)
![](img/2024-02-06-18-29-17.png)
### 1.Two Sum
```py
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        output=[0,0]
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                if nums[i]+nums[j]==target:
                    output[0]=i
                    output[1]=j
        return output
```
最简单的思路
hashmap：
```py
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap={}
        for i in range(len(nums)):
            complement=target-nums[i]
            if complement in hashmap:
                return [i,hashmap[complement]]
            hashmap[nums[i]]=i
```
意思就是hashmap的hashmap[key]获取的就是对应的value
### 49.Group Anagrams
```py
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        grouped_anagrams = {}
        for words in strs:
            sorted_word=''.join(sorted(words))
            if sorted_word in grouped_anagrams:
                grouped_anagrams[sorted_word].append(words)
            else:
                grouped_anagrams[sorted_word]=[words]
        result = list(grouped_anagrams.values())
        return result
```
hashmap真有用啊）
```sorted_word=''.join(sorted(words))```意思是，sorted，然后重新变成字符串（不然就是一个list）
![](img/2024-02-06-19-27-37.png)
如果sorted在hashmap里面，那么他对应的值里面，加上当前words
如果不在，那么就创建一个新的![](img/2024-02-06-19-29-55.png)
但是hashmap太有用了（？？？
最后，将 grouped_anagrams 字典中的所有值（value）提取出来，并将它们组成一个列表
### 347.Top K Frequent Elements
依然用了hashmap
但是一开始是：
```py
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        hashmap={}
        for i in nums:
                hashmap[i]=1+hashmap.get(i,0)
        frequent=sorted(hashmap.values()，reverse=True)
        #[3,2,1]
        result = []
        for j in range(k):
            value=hashmap[frequent[j]]
            result.append(value)
        return result
```
这里的sorted里面的reverse=True是降序排列的意思
但是会报错

根据chatgpt debug之后是这样的：
```py
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        hashmap={}
        for i in nums:
                hashmap[i]=1+hashmap.get(i,0)
        frequent = sorted(hashmap.items(), key=lambda x: x[1], reverse=True)
        #[3,2,1]
        result = []
        for j in range(k):
            result.append(frequent[j][0])
        return result
```
![](img/2024-02-06-19-54-53.png)
```‘1’:3```
然后实际上x[1]就是3，按照key值排序
(![](img/2024-02-06-19-57-00.png)
因为sorted会对其本身进行操作

### 238.Product of Array Except Self
难点在于算法的时间复杂度是O(n)，所以都是一次遍历

```py
class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        n=len(nums)
        prefix=[1]*n
        suffix=[1]*n
        left=1
        for i in range(n):
            prefix[i]=left
            left*=nums[i]
        right=1
        for j in range(n-1,-1,-1):
            suffix[j]=right
            right*=nums[j]
        result=[1]*n
        for k in range(n):
            result[k]=prefix[k]*suffix[k]
        return result
```
这个题目的意思就是，前缀乘积x后缀乘积
前缀乘积就是nums[i]左侧所有的乘积
后缀乘积就是nums[i]右侧所有的乘积
还有比较特别的是倒序for j in range(n-1,-1,-1):
![](img/2024-02-07-15-39-53.png)

### 36.Valid Sudoku

我的方法很...繁琐？一遍对row进行检测，一遍对col进行检测
还有一遍对3x3进行检测

```py
class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        #checkROW
        for row in board:
            hashmapROW={}
            for cell in row:
                if cell.isdigit():
                    hashmapROW[cell]=1+hashmapROW.get(cell,0)
            value=hashmapROW.values()
            for i in value:
                if i>1:
                    return False
        #checkCol
        for col in board:
            hashmapCOL={}
            for cell in row:
                if cell.isdigit():
                    hashmapCOL[cell]=1+hashmapCOL.get(cell,0)
            for i in hashmapCOL.values():
                if i>1:
                    return False
        #check 3x3
        for start_row in range(0, len(board), 3):
            for start_col in range(0, len(board[0]), 3):
                hashmapTHREE = {}
                for col in range(start_col, start_col + 3):
                    for row in range(start_row, start_row + 3):
                        if board[col][row].isdigit():
                            cell = board[col][row]
                            hashmapTHREE[cell] = 1 + hashmapTHREE.get(cell, 0)
                for count in hashmapTHREE.values():
                    if count > 1:
                        return False

        return True
```
但是提交的时候反而：
![](img/2024-02-07-16-02-43.png)
让chatGPT改成了这样就能过了：
```py
class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        # Check each row
        for row in board:
            hashmapROW = {}
            for cell in row:
                if cell.isdigit():
                    hashmapROW[cell] = 1 + hashmapROW.get(cell, 0)
                    if hashmapROW[cell] > 1:
                        return False

        # Check each column
        for col in range(9):
            hashmapCOL = {}
            for row in range(9):
                if board[row][col].isdigit():
                    hashmapCOL[board[row][col]] = 1 + hashmapCOL.get(board[row][col], 0)
                    if hashmapCOL[board[row][col]] > 1:
                        return False

        # Check each 3x3 subgrid
        for start_row in range(0, 9, 3):
            for start_col in range(0, 9, 3):
                hashmapTHREE = {}
                for col in range(start_col, start_col + 3):
                    for row in range(start_row, start_row + 3):
                        if board[col][row].isdigit():
                            cell = board[col][row]
                            hashmapTHREE[cell] = 1 + hashmapTHREE.get(cell, 0)
                            if hashmapTHREE[cell] > 1:
                                return False

        return True
```
源代码中列检查和3x3检查有错误
所以改成了
```py
        for col in range(9):
            hashmapCOL = {}
            for row in range(9):
```

### 128.Longest Consecutive Sequence
首先这个是他给的test case 能过：
```py
class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums=sorted(nums)
        longestSeq=1
        if len(nums)==0:
            return 0
        for i in range(len(nums)-1):
            if nums[i]==nums[i+1]-1:
                longestSeq+=1
        return longestSeq
```
但是提交的时候的这个testcase没过
![](img/2024-02-07-16-28-27.png)
后来发现是因为 这个序列是：
-1 0 1  | 3 4 5 6 7 8 9
但是我反而一直在往上加，所以一到不等于的情况，想到用一个set来储存当前数值
```py
class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums=sorted(nums)
        longestSeq=1
        seqSET=set()
        if len(nums)==0:
            return 0
        
        for i in range(len(nums)-1):
            if nums[i]==nums[i+1]-1:
                longestSeq+=1
                seqSET.add(longestSeq)
            else:
                longestSeq=0
        seqSET.add(longestSeq)   
        longestSeq=max(seqSET)    
        return longestSeq
```
看了下答案呢确实用了set
```py
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        numSet = set(nums)
        longest = 0

        for n in numSet:
            # check if its the start of a sequence
            if (n - 1) not in numSet:
                length = 1
                while (n + length) in numSet:
                    length += 1
                longest = max(length, longest)
        return longest
```
但是我的版本也可以修改为：
```py
class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0

        nums = sorted(nums)
        longestSeq = 1
        currentSeq = 1

        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                currentSeq += 1
            elif nums[i] != nums[i-1]:
                currentSeq = 1
            longestSeq = max(longestSeq, currentSeq)

        return longestSeq
```


## Two Pointers
### 125.Valid Palindrome
我自己的方法，很快给做出来了，但是看下他给的方法？
```py
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        #turn to string without uppercase and diploma''
        s_r=''
        for i in s:
            if i.isalpha():
                s_r+=i.lower()
            elif i.isdigit():
                s_r+=i
        #if it is a palindrome
        length=len(s_r)
        if length==0:
            return True
        #even or odd
        max_num=length//2
        i=0
        if length%2==0:
            while(i<max_num):
                if s_r[i]!=s_r[length-1-i]:
                    return False
                i+=1
            return True
        elif (length%2!=0):
            while(i<max_num):
                if s_r[i]!=s_r[length-1-i]:
                    return False
                i+=1
            return True
```
草，他也是差不多额额，但是后面那个一大堆就很快（）
```py
class Solution:
    def isPalindrome(self, s: str) -> bool:
        new = ''
        for a in s:
            if a.isalpha() or a.isdigit():
                new += a.lower()
        return (new == new[::-1])

```
![](img/2024-02-12-19-47-41.png)

### 167.Two Sum II - Input Array Is Sorted
我自己的初次尝试，能跑过test case但是提交的时候有错误
```py
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        fast=1
        slow=0
        for i in range(len(numbers)):
            sum=numbers[slow]+numbers[fast]
            if sum<target:
                slow+=1
                fast+=1
            elif sum>target:
                slow-=1
            else:
               return [slow+1,fast+1] 
```
看了眼chatgpt，从头尾开始放置指针
```py
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        fast=len(numbers)-1
        slow=0
        while fast>slow:
            sum=numbers[slow]+numbers[fast]
            if sum<target:
                slow+=1
            elif sum>target:
                fast-=1
            else:
               return [slow+1,fast+1] 
        return []
```

### 15.3Sum
看了眼提示，是fix一个number，然后剩下的变成2sum
有一些是根据chatGPT写的：
```py
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        sorted_nums = sorted(nums)
        result = []
        
        for i in range(len(sorted_nums)-2):
            slow, fast = i + 1, len(sorted_nums) - 1
            #跳过重复的元素
            if i > 0 and sorted_nums[i] == sorted_nums[i-1]:
                continue
            while slow < fast:
                current_sum = sorted_nums[i] + sorted_nums[slow] + sorted_nums[fast]
                if current_sum > 0:
                    fast -= 1
                elif current_sum < 0:
                    slow += 1
                else:
                    figure = [sorted_nums[i], sorted_nums[slow], sorted_nums[fast]]
                    result.append(figure)
                    #找到结果之后， 用于跳过与当前找到的组合中第二个元素（即 figure[1]）、第三个元素（即 figure[2]）相同的元素。这是因为在已排序的数组中，相同的元素可能会组成相同的组合。
                    while slow < fast and sorted_nums[slow] == figure[1]:
                        slow += 1
                    while slow < fast and sorted_nums[fast] == figure[2]:
                        fast -= 1
        return result
```
主要是边界的问题比如说
看了眼neetcode差不多就是这个思路？
### 11.Container With Most Water
```py
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        max_water=0
        left=0
        right=len(height)-1
        while (right>left):
            distance=right-left
            side_len=min(height[right],height[left])
            max_water=max(max_water,distance*side_len)
            if height[right]>height[left]:
                left+=1
            else:
                right-=1
        return max_water
        
```
看了眼提示很快就做出来了
### 42.Trapping Rain Water
臭名昭著接雨水
```py
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        left=0
        right=len(height)-1
        max_left=height[left]
        max_right=height[right]
        res=0
        while left<right:
            if max_left<max_right:
                if max_left-height[left]>0:
                    res+=max_left-height[left]
                left+=1
                max_left = max(max_left, height[left])
            else:
                if max_right-height[right]>0:
                    res+=max_right-height[right]
                right-=1
                max_right = max(max_right, height[right])
        return res
```
本质是1.理解为什么 雨水的量是min[left,right]-height[i]
左侧做大的数字-目前高度
2.为什么左右pointer可以规避掉min[left，right]，因为
本质就是哪边小就往那边走


## Binary Search Tree
### 704.Binary Search
很简单，但是！时间太慢了
```py
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        for i in range(len(nums)):
            if nums[i]==target:
                return i
        return -1
``` 
如何使用BST呢
```py
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        #for i in range(len(nums)):
        #    if nums[i]==target:
        #        return i
        #return -1
        left=0
        right=len(nums)-1
        while left<=right:
            m=(left+right)//2
            if nums[m]>target:
                right=m-1
            elif nums[m]<target:
                left=m+1
            else:
                return m
        return -1

```
其实就是二分查找，不是树状结构，还在思考怎么和树状搞到一起
### 74.Search a 2D Matrix
编程矩阵，对时间复杂度有要求![](img/2024-02-14-19-15-09.png)
```py
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        narROW=0
        m=len(matrix)
        if m==0:
            return False
        n=len(matrix[0])
        if n==0:
            return False
        for i in range(m):
            if matrix[i][0]<=target<=matrix[i][n-1]:
                narROW=i
                break
            elif matrix[i][0]>target:
                break
        
        left=0
        right=n-1
        while left<=right:
            middle=(left+right)//2
            if matrix[narROW][middle]<target:
                left=middle+1
            elif matrix[narROW][middle]>target:
                right=middle-1
            else:
                return True
        return False
```
基本思路：找到对应的行然后对行使用二分查找
neetcode的方法本质还是对上面找行的也使用了二分查找
### 875.koko eat banana
```py
class Solution(object):
    def minEatingSpeed(self, piles, h):
        """
        :type piles: List[int]
        :type h: int
        :rtype: int
        """
        k=0
        piles=sorted(piles)
        pace=piles[0]
        time=0
        while(pace<piles[-1] and time<=h):
            time=0
            for i in range(len(piles)):
                if pace>=piles[i]:
                    time+=1
                elif pace<piles[i]:
                    temp=(piles[i]%pace)
                    if temp!=0:
                        time+=piles[i]//pace+1
                    else:
                        time+=piles[i]//pace
            pace+=1
        return pace
```
但是过不了案例）
实际上用二分查找的话，意思是：
left和right刚好是速度最小值和最大值,比如说最大是11，最小是1，
那么就从3，4，5，6，...,11里面慢慢找

middle就是(right+left)//2
```py
class Solution(object):
    def minEatingSpeed(self, piles, h):
        """
        :type piles: List[int]
        :type h: int
        :rtype: int
        """
        piles=sorted(piles)
        left,right=1,piles[-1]
        while (left<=right):
            time=0
            middle=(left+right)//2
            for pile in piles:
                time+=math.ceil(pile/midde)
            if time>h:
                left=middle+1
            else:
                right=middle-1
        return left
```
```math.ceil()```向上取整
```math.floor()```向下取整

### 153.Find Minimum in Rotated Sorted Array

莫名生草
```py
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        original=sorted(nums)
        return original[0]
```
```py
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left=0
        right=len(nums)-1
        count=0
        while(left<right):
            mid=(left+right)//2
            if nums[mid]>nums[right]:
                left=mid+1
            else:
                right=mid
        return nums[left]
```
```if nums[mid]>nums[right]```
因为是 比如34512，如果中间值大于右边，就说明中间值在递增那个的左半边（大概理解一下）
![](img/2024-02-21-02-46-14.png)

### 33.Search in Rotated Sorted Array
```py
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left=0
        right=len(nums)-1
        if len(nums)==1 and target==nums[0]:
            return 0
        while(left<=right):
            mid=(left+right)//2
            if nums[mid]==target:
                return mid
            if nums[left]<=nums[mid]:
                if nums[left]<=target<=nums[mid]:
                    right=mid-1
                else:
                    left=mid+1
            else:
                if nums[mid]<=target<=nums[right]:
                    left=mid+1
                else:
                    right=mid-1
            # if nums[mid]==target:
            #     return mid
            # elif abs(target)-abs(nums[left])<abs(target)-abs(nums[right]):
            #         left=mid+1
            # else:
            #         right=mid
        return -1
```
是之前那道题的变体，主要是要找到变体是在哪一边
一开始是过了下面的test case
然后很多都是条件判断了，大概自己想明白了虽然还是靠着chatgpt做的第一次
本质还是左右寻找target在哪边，然后左右都是逐步递增/递减的
### 981.Time Based Key-Value Store







### 4.
```py
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        nums = nums1 + nums2
        nums = sorted(nums)
        mid = 0

        if len(nums) % 2 == 0:
            right = len(nums) // 2
            left = right - 1
            return (nums[right] + nums[left]) / 2.0
        elif len(nums) % 2 != 0:
            mid = len(nums)//2
            return float(nums[mid])
```
草绷不住了，但是sorted在面试中不准用的，来看下常规解法，这是一道HARD题，看了眼评论很多人都做不出来的


## Sliding Window



## Stack
其实大部分发现都做过的，但是再做一次试试
![](../2024-02-22-211212.png)