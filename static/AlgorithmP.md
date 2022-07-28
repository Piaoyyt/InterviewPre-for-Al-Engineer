# 解题算法的记录和总结

>本文旨在总结自己平时在刷题时所碰到的数据结构题目以及比较重要和典型的解题思路。
## 主要内容

- [数据结构](#Data_struture)
- [解题思路](#Solve_ideas)

## <a id = 'Data_struture'></a> 1.数据结构篇
### 1.1数组
> 通常是以一维数组或者二维矩阵的形式来考察，典型的类型如下。
- 搜索问题
  > 从给定的数组里面搜索满足条件的值，条件可能是恰好等、第一个大于等于、最后一个小于等_,通常是利用二分的思想去做，二分里面重要的是就是确定两个指针，即哪一个指针指向明确的点，哪一个
  > 指向不明确的，循环直至两指针重合即可；对于有些搜索可能不是二分，但是本质上也是不断缩小搜索的范围
  > 边界，直到边界重合。
  - [在排序数组中查找数字 I](https://leetcode.cn/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)
  - [旋转数组的最小数字](https://leetcode.cn/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)
  - 寻找两个正序数组的中位数
  - [二维数组中的查找](https://leetcode.cn/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)
- 排序问题
  > 对于给的数组进行各类排序，或者隐含的利用排序的思想去做。
  - [数组中的逆序对](https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)
  - [计算右侧小于当前元素的个数](https://leetcode.cn/problems/count-of-smaller-numbers-after-self/)
    - >利用的是归并排序在并的时候，判断两个子数组各个值的本质上为判断逆序对。
  - [数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/)
    - >既可以利用堆排序的思想，用一个大小为K的堆保持数组里面最大的K个数，这样总的时间复杂度为nlogK；另一种思路
      利用快速排序的思想，快速排序每一次选定一个基准，然后将基准左边全变成比它小的，右边变成比它大的，我们可以再每一次
      排完之后增加一个判断，如果基准所在的位置为倒数第K个，那么就是我们要求的值，如果在右边，那么就从左子区间继续查找，
      否则从右子区间查找。
      ```python3
      class Solution:
             def findKthLargest(self, nums: List[int], k: int) -> int:
        
                def qucik_sort(left, right):
            
                    pviot = nums[left]
                    l = left
                    r = right
                    while l < r:
                    #先从右往左找第一个不符合大于等于的元素，即小于它的数
                    #如果先从左往右，就需要改变写法，因为先左到右，会导致l一开始的指向变了，指向的是
                    #第一个大于的值，假如是5，6，那么此时l指向6跳出循环，因为此时l并不是指向最后一个小于等于基准的值，所以最后交换基准值的时候就会导致出错，将5和6交换出错，所以这样写的话就需要改成另一种形式。
                        while l < r and nums[r] >= pviot:
                            r -= 1
                        while l < r and nums[l] <= pviot:
                            l += 1
                    #l可能等于r：因为l永远指向的是小于等于基准的值，所以这时候重合意味着l和r指向的数为最后一个小于等于基准的值，l仍然是我们最后要交换的地方；l小于r：l的值和r的值交换之后，l的值仍然是小于等于基准的值，意义不变
                        nums[r], nums[l] = nums[l], nums[r]
                    nums[l], nums[left] = nums[left], nums[l]
                    if l == len(nums) - k:
                        return nums[l]
                    if l > len(nums) - k: return qucik_sort(left, l-1)
                    else: return qucik_sort(l + 1, right)
                return qucik_sort(0, len(nums) - 1)
      
       def quick_search(left, right):
            '''
            以数组的left位置为基准，找该基准应该处于排序的第几大位置，第1大即最后一个位置，也就是数组的末尾
            从左到右先遍历的情况，此时r为最后一个小于基准的位置，和上面的有区别
            '''
            pivot = nums[left]
            l, r = left + 1, right
            while l <= r:#注意，如果仅仅有俩个数[3,4]，这时候如果是l < r就会出错，因为l是默认从left+1开始判断，会默认交换3和4，但实际4>3不应该交换
                #从左往右找第一个大于基准的位置
                while l <= r and nums[l] <= pivot:
                    l += 1
                #从右往左找第一个小于基准的位置
                while l <= r and nums[r] >= pivot:
                    r -= 1
                if l >= r:#说明此时r落在了最后一个小于基准的位置，这里的等号是防止l=r，比如[3,2]，这时候l=1，r=1，如果不break就会无限循环
                    break
                #交换左大右小的
                nums[l], nums[r] = nums[r], nums[l]
            #r所落的位置即不大于基准的最后一个元素上，r的右边全是大于基准的数
            nums[r], nums[left] = nums[left], nums[r]
            return r #返回基准所在的位置
      ```
  - [把数组排成最小的数](https://leetcode.cn/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)
    - > 巧妙的利用到了快速排序的思想，在原始的快速排序里面，l定位大于基准的位置，r定位小于基准的位置，然后不断交换，
      定位的准则仅是根据值的大小，而这一题里面定位的准则变了一下，变成了 l + pivot < r + pivot，本质上，还是快速
      排序的思想。
- 子数组、子序列问题
  > 一般是求给的数组里面满足某一条件的子数组或者子序列的个数，常常需要结合dp去做，即需要用额外的空间来记录之前的
  > 状态，有时候用来记录状态的结构可以是数组，也可以是哈希表（常在前缀后缀里用到），同时状态的定义不一定就直接是题目
  > 所要求解的目标，巧妙设计可以使得算法的效率更高。
  - [连续子数组](#ContinuousSubA)
  - [子序列](#SubSeq)

### 1.2链表
> 链表是一种插入和删除元素比较高效的数据结构，链表的类型可以分为很多类，刷题里面常见的几种有：环形链表、回文链表，相交链表、
> 双向链表等，不同链表会涉及到不同的判断技巧。
- 翻转链表
  - > 只需要每一次翻转当前结点，记录下一结点，然后依次后移即可，一般需要创建一个哨兵结点让第一个结点指向它。
  - [反转链表](https://leetcode.cn/problems/fan-zhuan-lian-biao-lcof/)
  - [k个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/)
- 合并链表
  - [合并k个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)
- 环形链表
  - > 对于一个环形链表，最典型的做法就是设置快慢指针，如果在两个指针都不指向None的情况下慢指针再一次和快指针重合，说明链表
    中存在环。
  - [环形链表](https://leetcode.cn/problems/linked-list-cycle/)
- 相交链表
  - > 判断两个链表是否有相交的部分，可以利用跳转的思想，当两个指向两链表的指针指向链表的末尾的时候，让指针跳转到另一个链表上，
    因为两个链表的长度和是一个定值，如果相交的话，那么一定会在这个过程中出现两指针指向的结点是同一个，否则两指针最后都会同时
    到达链表末尾。
  - [相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/)
- 回文链表
  - > 回文链表的关键是定位链表的中心，我们可以利用快慢指针去做，也可以统计出链表长度再指针右移去做，定位好之后，就是判断两边
    > 是否对称即可，可以用列表存储元素然后逐个判断即可，也可以先将某一部分反转，然后迭代挨个位置判断即可。
  - [回文链表](https://leetcode.cn/problems/palindrome-linked-list/)
### 1.3字符串
### 1.4哈希表
### 1.5二叉树
### 1.6图
### 1.7堆
### 1.8并查集
### 1.9线段树
### 1.10单调栈


## <a id = 'Solve_ideas'></a> 2.解题思路篇

### 2.1双指针
### 2.2回溯
### 2.3分治
### 2.4DFS和BFS
### 2.5贪心
### 2.6动态规划
> 动态规划的核心在于**通过之前的状态来得到当前的状态，关键在于状态的定义以及状态转移方程的设计**
> ，不同于贪心里面局部最优可以过渡到全局最优，动态规划需要对于每一种状态都记录，有的是利用前
> 一个状态，有的是利用前面所有时刻的状态，即根据场景的不同而转移的方式不同。
####<a id = 'ContinuousSubA'></a> 1.连续子数组问题
>对于连续子数组的问题，属于应用DP解题的典例，核心在于原问题等效于求所有子数组的和或者乘积等的最大
> 或最小问题，而所有子数组的可能情况☞求以每一个位置结尾的所有子数组的可能情况☞以当前位置结尾的子
> 数组情况又和**以前面一个位置**结尾的子数组有关☞构成了转移的方向，而转移的状态即当前位置结尾对应的子数组
> 目标值【根据题目不同目标会不一样】

- [连续子数组最大和](https://leetcode.cn/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)
- [乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/)
- [环形子数组的最大和](https://leetcode.cn/problems/maximum-sum-circular-subarray/)
- [子数组是否存在满足某个条件的](https://leetcode.cn/problems/continuous-subarray-sum/)
- [元素和为目标值的子矩阵数量](https://leetcode.cn/problems/number-of-submatrices-that-sum-to-target/)
- [统计全为1的正方形子矩阵](https://leetcode.cn/problems/count-square-submatrices-with-all-ones/)
- [最大正方形](https://leetcode.cn/problems/maximal-square/)  
####<a id = 'SubSeq'></a> 2.子序列问题
>对于子序列问题，相比于子数组少了一个连续的约束条件，因此当前状态可以由更多可能的状态转移而来，即可以不连续的
> 取，所以一般子序列的问题都是求满足某一个条件下的最优子序列求解问题，解决这类问题常规的思路和连续子数组的解
> 题思路一致，唯一的不同就是需要遍历当前位置之前的所有状态的值，即不管是时间还是空间上都是开销比较大的，因此
> 这种情况我们常常结合贪心的思路来去做，这样可以大大减少时间和空间开销。
- [最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)
- [最长递增子序列个数](https://leetcode.cn/problems/number-of-longest-increasing-subsequence/)
- [俄罗斯套娃信封问题](https://leetcode.cn/problems/russian-doll-envelopes/)
- [堆箱子](https://leetcode.cn/problems/pile-box-lcci/)
- [无矛盾的最佳球队](https://leetcode.cn/problems/best-team-with-no-conflicts/)
#### 3.字符串问题
> 字符串相关考察的问题主要是回文相关问题，比如典型的回文子序列和回文子串问题。

- [最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)
- [最长回文子序列](https://leetcode.cn/problems/longest-palindromic-subsequence/)
- [统计不同回文子序列](https://leetcode.cn/problems/count-different-palindromic-subsequences/)
- [段式回文](https://leetcode.cn/problems/longest-chunked-palindrome-decomposition/)
- [让字符串成为回文串的最少插入次数](https://leetcode.cn/problems/minimum-insertion-steps-to-make-a-string-palindrome/)

#### 4.背包问题
>所谓的背包问题，即在给定的背包限制容量下(通常是某一个约束，比如总重量、总体积等)，
> 去从给定的物品里面选取最优的组合值，用dp解答的思路即是逐渐扩大限制条件，逐渐扩大
> 可拿取物体的范围，最后得到全局下的最优解。
- 01问题
  - [分割等和子集:01](https://leetcode.cn/problems/partition-equal-subset-sum/)
  - [一和零](https://leetcode.cn/problems/ones-and-zeroes/)
  - [最后一块石头的重量](https://leetcode.cn/problems/last-stone-weight-ii/)
  - [盈利计划](https://leetcode.cn/problems/profitable-schemes/)
- 完全问题
  - [零钱兑换](https://leetcode.cn/problems/coin-change/)
  - [零钱兑换II](https://leetcode.cn/problems/coin-change-2/)
  
#### 5.状态压缩问题
> 对于题目中给的状态有时候不好表示或者表示起来十分复杂的时候，可以考虑用状态压缩的方式来表示，
> 所谓的状态压缩即将不同的状态用一串二进制数来进行表示，每一个位上的二进制数即代表状态的具体
> 构成，为什么表示成二进制数呢？因为在更新状态和判断的时候我们可以利用二进制中的位运算来巧妙
> 的表示，典型的就是合并状态用或操作，判断用与操作等

- [安卓手势解锁](https://leetcode.cn/problems/android-unlock-patterns/)
- [我能赢吗](https://leetcode.cn/problems/can-i-win/)
- [Nim游戏II](https://leetcode.cn/problems/game-of-nim/)
#### 6.数位dp问题
> 数位dp问题即在求解关于数的不同位值所构成的整体状态的统计问题，典型的比如给定一个数值大小的约束，
> 并给定能取的数字范围，求限制下所能构成的有效数字的个数。

**数位问题的关键在于：**  
1. 根据所给的限制对每一位上的范围做一个计算；  
2. 分析影响当前位置上数字取值的范围的因素【前导0、前值是否达到上界等】；
3. 设计dfs函数，可以用dfs的原因在于高位的状态会影响下低位的值选取情况，
所以从最高位层层往下遍历最后就可以得到想要的结果，因此传入dfs的参数一定
是对当前位值选取范围有决定意义的参数量【位次、前导0、前值是否上界等】，通 
过这些参数来决定当前位的取值范围，然后又根据当前位的选取给下低位传递新的
参数【即递归的思想】，直到碰到递归的终止条件【一般是递归完最低位就可以结束】；
- [最大位N的数字组合](https://leetcode.cn/problems/numbers-at-most-n-given-digit-set/)
- [中心对称数](https://leetcode.cn/problems/strobogrammatic-number-iii/)
- [统计各位数字都不同的数字个数](https://leetcode.cn/problems/count-numbers-with-unique-digits/)
#### 7.概率dp问题
> 运用概率论的知识对问题进行分析来设定状态以及状态转移方程。
- [锦标赛](https://www.notion.so/b3637c19e8f34c8bb32ecce5b5bd10c5#4f251640025944d0ad24dedebd412608)
#### 8.博弈dp问题