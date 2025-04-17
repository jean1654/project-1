import streamlit as st
import pandas as pd

# --------------------------------------------------
# IE2108: Data Structures & Algorithms in Python Cheatsheet
# All sections with example code matching lecture slides (DSA1-DSA4)
# --------------------------------------------------

st.set_page_config(
    page_title="IE2108 Python Full Cheatsheet",
    layout="wide"
)
st.title("📘 IE2108 Full Python Coding Cheatsheet")

cheat_data = [
    {
        "Topic": "Recursion: Factorial",
        "Summary": "Compute n! via self-call with base case 0! = 1",
        "Code": '''
# Factorial Function (DSA2.pdf)

def fact(n):
    if n == 0:
        return 1
    else:
        return n * fact(n-1)

n = int(input("Enter n: "))
print(f"factorial({n}) = {fact(n)}")
''',
        "Explanation": "Recursive factorial with base case n=0. O(n) time, O(n) space."
    },
    {
        "Topic": "Recursion: Reverse Array",
        "Summary": "Reverse elements in-place with two indices",
        "Code": '''
# Reverse Array (DSA2.pdf)

def ReverseArray(A, i, j):
    if i < j:
        A[i], A[j] = A[j], A[i]
        ReverseArray(A, i+1, j-1)

A = [5, 3, 2, 6]
print("Original array:", A)
ReverseArray(A, 0, len(A)-1)
print("Reversed array:", A)
''',
        "Explanation": "Swap outer elements and recurse inward. O(n) time."
    },
    {
        "Topic": "Arrays: Basic Operations",
        "Summary": "Access, append, pop, index",
        "Code": '''
# Array operations (DSA2.pdf)

arr = [10, 20, 30]
print(arr[1])        # 20
arr.append(40)       # append
arr.pop()            # delete last
idx = arr.index(20)  # find index
print(idx)           # 1
''',
        "Explanation": "O(1) access and append/pop; O(n) index search."
    },
    {
        "Topic": "Linked List: Node & Insert/Delete",
        "Summary": "Singly-linked list insertion and deletion",
        "Code": '''
# Linked List insertion/deletion (DSA2.pdf)

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

# Insert at head
def insert_head(head, val):
    node = Node(val)
    node.next = head
    return node

# Delete first occurrence
def delete_val(head, val):
    dummy = Node(0)
    dummy.next = head
    prev = dummy
    while prev.next:
        if prev.next.data == val:
            prev.next = prev.next.next
            break
        prev = prev.next
    return dummy.next
''',
        "Explanation": "O(1) insert at head; O(n) delete/search."
    },
    {
        "Topic": "Stack (using list)",
        "Summary": "Use list with append/pop, LIFO",
        "Code": '''
# Stack with list (DSA2.pdf)

stack = []
stack.append(1)
stack.append(2)
print(stack.pop())  # 2
print(stack.pop())  # 1
''',
        "Explanation": "LIFO operations, O(1) time."
    },
    {
        "Topic": "Queue (using list)",
        "Summary": "Use list with append/pop(0), FIFO",
        "Code": '''
# Queue with list (DSA2.pdf)

queue = []
queue.append(1)
queue.append(2)
print(queue.pop(0))  # 1
print(queue.pop(0))  # 2
''',
        "Explanation": "FIFO; pop(0) O(n), use deque for O(1)."
    },
    {
        "Topic": "Binary Search",
        "Summary": "Divide sorted list by half each step",
        "Code": '''
# Binary Search improved (DSA4.pdf)

def binary_search(L, key):
    i, j = 0, len(L)-1
    while i <= j:
        k = (i + j) // 2
        if key == L[k]:
            return k
        if key < L[k]:
            j = k - 1
        else:
            i = k + 1
    return -1

L = [-2, 3, 7, 10, 17]
print(binary_search(L,17))  # 4
print(binary_search(L,2))   # -1
''',
        "Explanation": "O(log n) time, requires sorted list."
    },
    {
        "Topic": "Insertion Sort",
        "Summary": "Build sorted list by inserting elements",
        "Code": '''
# Insertion Sort (cheatsheet)

def insertion_sort(arr):
    for i in range(1,len(arr)):
        key = arr[i]
        j = i-1
        while j>=0 and arr[j]>key:
            arr[j+1]=arr[j]
            j-=1
        arr[j+1]=key
    return arr

print(insertion_sort([4,3,2,1]))  # [1,2,3,4]
''',
        "Explanation": "Best O(n), worst O(n^2)."
    },
    {
        "Topic": "Merge Sort",
        "Summary": "Recursive split and merge",
        "Code": '''
# Merge Sort (cheatsheet)

def merge_sort(arr):
    if len(arr)<=1: return arr
    mid=len(arr)//2
    L=merge_sort(arr[:mid])
    R=merge_sort(arr[mid:])
    return merge(L,R)

def merge(L,R):
    res=[]; i=j=0
    while i<len(L) and j<len(R):
        if L[i]<=R[j]: res.append(L[i]); i+=1
        else: res.append(R[j]); j+=1
    res.extend(L[i:]); res.extend(R[j:])
    return res

print(merge_sort([5,3,8,6]))  # [3,5,6,8]
''',
        "Explanation": "O(n log n) time, O(n) space."
    },
    {
        "Topic": "Quick Sort",
        "Summary": "Pivot partition and recurse",
        "Code": '''
# Quick Sort (DSA3.pdf)

def quick_sort(A):
    if len(A)<2: return A
    pivot=A[len(A)//2]
    left=[x for x in A if x<pivot]
    mid=[x for x in A if x==pivot]
    right=[x for x in A if x>pivot]
    return quick_sort(left)+mid+quick_sort(right)

print(quick_sort([10,5,1,17,14,8,7,26,21,3]))  # sample sorted list
''',
        "Explanation": "Average O(n log n), worst O(n^2)."
    },
    {
        "Topic": "Heap: Sift-down & Heapify",
        "Summary": "Convert array into max-heap",
        "Code": '''
# Siftdown & Heapify (DSA3.pdf)

def siftdown(A,i):
    n=len(A)-1
    largest=i; l=2*i; r=2*i+1
    if l<=n and A[l]>A[largest]: largest=l
    if r<=n and A[r]>A[largest]: largest=r
    if largest!=i:
        A[i],A[largest]=A[largest],A[i]
        siftdown(A,largest)

def heapify(A):
    n=len(A)-1
    for i in range(n//2,0,-1):
        siftdown(A,i)

A=[None,3,9,2,1,7]
heapify(A)
print(A)  # max-heap
''',
        "Explanation": "Builds heap in O(n)."
    },
    {
        "Topic": "Heap Sort",
        "Summary": "Sort via extract-max",
        "Code": '''
# Heapsort (DSA3.pdf)

A=[None,3,9,2,1,7]
heapify(A)
for i in range(len(A)-1,1,-1):
    A[1],A[i]=A[i],A[1]
    siftdown(A,1)
print(A[1:])  # sorted list
''',
        "Explanation": "In-place O(n log n)."
    },
    {
        "Topic": "Binary Tree Traversals",
        "Summary": "Preorder, Inorder, Postorder",
        "Code": '''
# Preorder Traversal (DSA2.pdf)
# Root -> Left -> Right

def Preorder(root):
    res = []
    if root:
        res.append(root.data)
        res = res + Preorder(root.left)
        res = res + Preorder(root.right)
    return res

# Inorder Traversal (DSA2.pdf)
# Left -> Root -> Right

def Inorder(root):
    res = []
    if root:
        res = Inorder(root.left)
        res.append(root.data)
        res = res + Inorder(root.right)
    return res

# Postorder Traversal (DSA2.pdf)
# Left -> Right -> Root

def Postorder(root):
    res = []
    if root:
        res = Postorder(root.left)
        res = res + Postorder(root.right)
        res.append(root.data)
    return res
''',
        "Explanation": "Uses list concatenation to build traversal order."
    },
    {
        "Topic": "Binary Search Tree: Insert/Delete",
        "Summary": "Maintain BST property",
        "Code": '''
# BST Insert/Delete (DSA3.pdf)

class Node:
    def __init__(self,val):
        self.data=val; self.left=None; self.right=None

def BSTinsert(root,val):
    if root is None: return Node(val)
    if val<=root.data:
        root.left = BSTinsert(root.left,val)
    else:
        root.right = BSTinsert(root.right,val)
    return root

def BSTdelete(root,val):
    if root is None: return None
    if val<root.data:
        root.left = BSTdelete(root.left,val)
    elif val>root.data:
        root.right = BSTdelete(root.right,val)
    else:
        if root.left is None: return root.right
        if root.right is None: return root.left
        succ = root.right
        while succ.left: succ = succ.left
        root.data = succ.data
        root.right = BSTdelete(root.right,succ.data)
    return root
''',
        "Explanation": "Handles 0/1/2 child cases."
    },
    {
        "Topic": "Graph: Adjacency List",
        "Summary": "Dict of neighbor lists",
        "Code": '''
# Adjacency List (DSA4.pdf)

graph = {
    1: [2,5],
    2: [1,3,4],
    3: [2,5],
    4: [2],
    5: [1,3]
}
''',
        "Explanation": "Space O(n+m). Efficient for sparse graphs."
    },
    {
        "Topic": "Graph: Adjacency Matrix",
        "Summary": "2D matrix of 0/1 entries",
        "Code": '''
# Adjacency Matrix example

A = [
    [0,1,0,1,0],
    [1,0,0,0,1],
    [0,0,0,0,1],
    [1,0,0,0,1],
    [0,1,1,1,0]
]
''',
        "Explanation": "Space O(n^2)."
    },
    {
        "Topic": "Breadth-First Search (BFS)",
        "Summary": "Level-order graph traversal",
        "Code": '''
# BFS on non-tree (DSA4.pdf)

def bfs(graph, start):
    visited = [False] * (len(graph) + 1)
    queue = [start]
    visited[start] = True
    order = []
    while queue:
        v = queue.pop(0)
        order.append(v)
        for nei in graph[v]:
            if not visited[nei]:
                visited[nei] = True
                queue.append(nei)
    return order

# Example usage omitted
''',
        "Explanation": "O(n+m) time."
    },
    {
        "Topic": "BFS: Shortest Paths",
        "Summary": "Compute dist in unweighted graph",
        "Code": '''
# BFS shortest path lengths (DSA4.pdf)

def bfs_shortest_path_lengths(graph, start):
    from collections import deque
    distances = {start: 0}
    visited = {start}
    q = deque([start])
    while q:
        v = q.popleft()
        for nei in graph[v]:
            if nei not in visited:
                visited.add(nei)
                distances[nei] = distances[v] + 1
                q.append(nei)
    return distances
''',
        "Explanation": "Returns dict of min hops from start."
    },
    {
        "Topic": "Depth-First Search (DFS)",
        "Summary": "Principles: go deep, backtrack",
        "Code": '''
# DFS (DSA4.pdf)

def dfs(graph, start):
    visited = [False] * (len(graph) + 1)
    stack = [start]
    order = []
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            order.append(v)
            for nei in reversed(graph[v]):
                if not visited[nei]:
                    stack.append(nei)
    return order
''',
        "Explanation": "O(n+m) time."
    },
    {
        "Topic": "Dijkstra's Algorithm",
        "Summary": "Greedy SPT using min-heap",
        "Code": '''
# Dijkstra (DSA4.pdf)

import heapq

def dijkstra(graph, src):
    dist = {v: float('inf') for v in graph}
    dist[src] = 0
    pq = [(0, src)]
    parent = {src: None}
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v, w in graph[u].items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, parent
''',
        "Explanation": "O((n+m) log n)."
    },
    {
        "Topic": "Kruskal's Algorithm",
        "Summary": "Greedy MST via union-find",
        "Code": '''
# Kruskal (DSA3.pdf)

class DisjointSet:
    def __init__(self, vertices): self.parent = [-1] * vertices
    def find(self, node):
        if self.parent[node] == -1:
            return node
        return self.find(self.parent[node])
    def union(self, x, y):
        x_set = self.find(x)
        y_set = self.find(y)
        if x_set != y_set:
            self.parent[x_set] = y_set
''',
        "Explanation": "Sort edges & union if no cycle."
    },
    {
        "Topic": "Text Classification Example",
        "Summary": "Bag-of-words + Naive Bayes",
        "Code": '''
# Text classification pipeline (DSA4.pdf)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Example data omitted
''',
        "Explanation": "Vectorize text, train MultinomialNB."
    }
]

# Display
for item in cheat_data:
    with st.expander(f"🔹 {item['Topic']}"):
        st.write(f"**Summary**: {item['Summary']}")
        st.code(item['Code'], language='python')
        st.markdown(f"**Explanation**: {item['Explanation']}")

# Download

df = pd.DataFrame([{'Topic':t['Topic'],'Summary':t['Summary']} for t in cheat_data])
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download CSV", data=csv, file_name="ie2108_cheatsheet.csv", mime="text/csv")
