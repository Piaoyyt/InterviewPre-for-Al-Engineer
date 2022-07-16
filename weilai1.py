import collections

n, k, p = map(int,input().strip().split(' '))
colors = []
consumes = []

for i in range(n):
    color, consume = map(int, input().strip().split(' '))

    colors.append(color)
    consumes.append(consume)
#sum_color = collections.defaultdict(int)
#每个位置左边和右边不同颜色的个数
left = [[0 for i in range(k)] for j in range(n + 1)]
right = [[0 for i in range(k)] for j in range(n + 1)]
left[1][colors[0]] = 1
right[n][colors[n-1]] = 1
ans = 0
for i in range(1, n):
    cur_color = colors[i]
    cur_color2 = colors[n-i-1]
    for j in range(k):
        if j == cur_color:
            left[i+1][cur_color] = left[i][cur_color] + 1
        if j == cur_color2:
            right[n-i][cur_color2] = right[n-i+1][cur_color2] + 1
for i in range(2, n):#从第二个数开始选
    if consumes[i-1] <= p:
        for j in range(k):
            ans +=(left[i-1][j] * right[i+1][j])
print(ans)

