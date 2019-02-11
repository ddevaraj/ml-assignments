def moveZeroes(nums: 'List[int]') -> 'None':
    """
    Do not return anything, modify nums in-place instead.
    """
    a = set()
    for i in nums:
        if (nums.count(i) > int(len(nums) / 2)):
            a.add(i)

    element = a.pop()
    print(element)

print(moveZeroes([1,1,2,2,1,1]))