# def generateParenthesis(N):
#     ans = []
#     def backtrack(S = '', left = 0, right = 0):
#         print(S, ans)
#         if len(S) == 2 * N:
#             ans.append(S)
#             print('appended')
#             return
#         if left < N:
#             print(left)
#             backtrack(S+'(', left+1, right)
#         if right < left:
#             print(right)
#             backtrack(S+')', left, right+1)
#
#     backtrack()
#     return ans
# generateParenthesis(3)

def longestParenthesis(s):
    print('in here')
    max_cnt = 0
    for i in range(len(s)):
        # cnt_even = helper(s, i, i)
        cnt_odd = helper(s, i, i + 1)
        print(cnt_even, cnt_odd)


# max_cnt = max(max_cnt, cnt_even)
#     return max_cnt



def helper(s, i, j):
    max_cnt = 0
    left = 0
    right = 0

    print("i,j", i, j)
    while i >= 0 and j < len(s):
        if s[i] == '(':
            left += 1
        elif s[i] == ')':
            right += 1
        if s[j] == ')':
            right += 1
        elif s[j] == '(':
            left += 1
        i -= 1
        j += 1
    if left == right:
        max_cnt = max(max_cnt, left + right)
    # return left+right
    #     return 0

    print(right, left, left + right)


longestParenthesis('()()()(')