# This is Python 3
def answer(line):
    # TODO: Implement your logic here
    line = list(line)
    left = 0
    for i in range(len(line)):
      if line[i] == ' ' or  i == len(line)-1:
        temp_l = left
        temp_r = i - 1
        if i == len(line) - 1:
          temp_r = i 
        while temp_l < temp_r:
          line[temp_l] = line[temp_r]
          line[temp_r] = line[temp_l]
          temp_l += 1 
          temp_r -= 1
        left = i + 1 # jump of to the word after the blank space.
    line = "".join(line)  
    return line

N = int(input())
for _ in range(N):
    line = input()
    print(answer(line))