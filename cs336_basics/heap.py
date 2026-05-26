from __future__ import annotations

class MaxHeap:
    def __init__(self):
        self.heap: list[tuple[int, bytes, bytes, tuple[int, int]]] = []

    def get_left(self, i): return 2 * i + 1

    def get_right(self, i): return 2 * i + 2

    def get_parent(self, i): return (i - 1) // 2

    @property
    def get_root(self):
        return self.heap[0] if self.heap else None

    def remove(self):
        pass

    def insert(self, bigram: tuple[int, int], token1, token2, count: int):
        self.heap.append((count, token1, token2, bigram))
        i = len(self.heap) - 1

        while i>0:

            if self.heap[i][0] < self.heap[self.get_parent(i)][0]:
                break
            elif self.heap[i][0] == self.heap[self.get_parent(i)][0]:
                if self.heap[i][1] < self.heap[self.get_parent(i)][1]:
                    break
                elif self.heap[i][1] == self.heap[self.get_parent(i)][1]:
                    if self.heap[i][2] < self.heap[self.get_parent(i)][2]:
                        break

            p = self.get_parent(i)
            self.heap[i], self.heap[p] = self.heap[p], self.heap[i]
            i = p

    def heapify(self, i):
        l, r, n, cur = self.get_left(i), self.get_right(i), len(self.heap), self.heap[i]
        largest = i
        tmp_largest = cur

        if r<n:
            right = self.heap[r]
            if right[0] > tmp_largest[0]:
                tmp_largest = right
                largest = r
            elif right[0] == tmp_largest[0]:
                if right[1] > tmp_largest[1]:
                    tmp_largest = right
                    largest = r
                elif right[1] == tmp_largest[1]:
                    if right[2] > tmp_largest[2]:
                        tmp_largest = right
                        largest = r

        if l < n:
            left = self.heap[l]
            if left[0] > tmp_largest[0]:
                tmp_largest = left
                largest = l
            elif left[0] == tmp_largest[0]:
                if left[1] > tmp_largest[1]:
                    tmp_largest = left
                    largest = l
                elif left[1] == tmp_largest[1]:
                    if left[2] > tmp_largest[2]:
                        tmp_largest = left
                        largest = l
        if largest != i:
            self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
            self.heapify(largest)

    def extract_root(self):
        if len(self.heap) <= 0: return None
        if len(self.heap) == 1: return self.heap.pop()

        res = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.heapify(0)

        return res

if __name__ == "__main__":
    firstHeap = MaxHeap()
    firstHeap.insert((121, 123), b"a", b"c", 12)
    firstHeap.insert((121, 123), b"a", b"c", 15)
    firstHeap.insert((121, 123), b"b", b"c", 12)
    firstHeap.insert((121, 134), b"a", b"d", 12)
    firstHeap.insert((121, 134), b"a", b"c", 123)
    print(firstHeap.heap)
    print(firstHeap.extract_root())
    print(firstHeap.extract_root())
    print(firstHeap.extract_root())
    print(firstHeap.extract_root())
    print(firstHeap.extract_root())
    print(firstHeap.heap)