#滑动窗口的长度和缓冲区的长度由人选定
#疑问：采取这种获取数据集的方式，如何保证均为一种拍子（如4/4拍）

notes = [12,12,12,3,5,8,9,7]
dataset = ([12,12,12,12,5,5,3,3],[5,8,8,10,1,8,9,7],[3,3,0,5,8,9,7,12])

class LZ77:

    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer_size = 4

    def longest_match(self, data, cursor):
        end_buffer = min(cursor + self.buffer_size, len(data))

        p = -1
        l = -1

        for j in range(cursor+1, end_buffer+1):
            start_index = max(0, cursor - self.window_size + 1)
            substring = data[cursor + 1:j + 1]

            for i in range(start_index, cursor+1):
                repetition = int( len(substring) / (cursor - i + 1) )
                last = len(substring) % (cursor - i + 1)
                matchedstring = data[i:cursor + 1] * repetition + data[i:i + last]

                if matchedstring == substring and len(substring) > l:
                    p = cursor - i + 1
                    l = len(substring)

        if p == -1 and l == -1:
            return 0, 0
        return p, l

    def compress(self, message):
        i = -1
        cnt = 0
        while i < len(message)-1:
            (p, l) = self.longest_match(message, i)
            cnt += 1
            i += (l+1)
        return cnt

def fitnessfunction():
    totaldis = 0
    for datanum in range(len(dataset)):
        cx = compressor.compress(dataset[datanum])
        cy = compressor.compress(notes)
        cxy = compressor.compress(notes+dataset[datanum])
        totaldis += ( cxy - min(cx,cy) ) / max(cx,cy)
#        print(cx,cy,cxy,totaldis)
    return 1/totaldis

if __name__ == '__main__':
    compressor = LZ77(6)
    print(fitnessfunction())