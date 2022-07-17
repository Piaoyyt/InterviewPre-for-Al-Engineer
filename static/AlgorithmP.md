# Algorithm problems recoding and summary

>本文旨在总结自己平时在刷题时所碰到的数据结构题目以及比较重要和典型的解题思路。
## Main Content

- [数据结构](#Data_struture)
- [解题思路](#Solve_ideas)

## <a id = 'Data_struture'></a> 1.数据结构篇

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
#### 1.连续子数组问题
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
#### 2.子序列问题
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
> 可拿取物体的范围，最后得到全局下的最优解
#### 5.状态压缩问题

#### 6.数位dp问题

#### 7.概率dp问题

#### 8.博弈dp问题