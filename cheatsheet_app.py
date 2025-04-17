import streamlit as st
import pandas as pd

# --------------------------------------------------
# IE2108: Data Structures & Algorithms in Python Cheatsheet
# Updated: DFS, Kruskal, and Text Classification code removed per exam scope
# --------------------------------------------------

st.set_page_config(
    page_title="IE2108 Detailed Python Cheatsheet",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DATA: List of dicts, each with Topic, Summary, Code (if required), Explanation
cheat_data = [
    {
        "Topic": "Recursion: Factorial",
        "Summary": "Compute n! via self-call with base case n=0",
        "Code": '''
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
# Example
print(factorial(5))  # 120''',
        "Explanation": "Uses recursion: calls itself with n-1 until base case. Time O(n), space O(n) due to call stack."
    },
    {
        "Topic": "Recursion: Reverse Array",
        "Summary": "Reverse elements in-place with two indices",
        "Code": '''
def reverse_array(A, i, j):
    if i < j:
        A[i], A[j] = A[j], A[i]
        reverse_array(A, i+1, j-1)
# Example
A = [1,2,3,4,5]
reverse_array(A, 0, len(A)-1)
print(A)  # [5,4,3,2,1]''',
        "Explanation": "Swaps outer elements and recurses inward until indices meet or cross. O(n) time."
    },
    {
        "Topic": "Arrays: Basic Operations",
        "Summary": "Access, search, insert/delete at end",
        "Code": '''
# Access
arr = [10,20,30]
print(arr[1])  # 20
# Append
arr.append(40)
# Delete last
arr.pop()
# Find index
idx = arr.index(20)
''',
        "Explanation": "Arrays provide O(1) access by index; append/pop O(1); insert/delete at arbitrary pos O(n)."
    },
    {
        "Topic": "Linked List: Node & Insert/Delete",
        "Summary": "Singly-linked list with dynamic nodes",
        "Code": '''
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
    return dummy.next''',
        "Explanation": "Linked list supports O(1) insert at head; deletion after search O(n). No random access."
    },
    {
        "Topic": "Stack (ADT)",
        "Summary": "LIFO push/pop operations",
        "Code": '''
class Stack:
    def __init__(self):
        self._data = []
    def push(self, x):
        self._data.append(x)
    def pop(self):
        return self._data.pop()
    def top(self):
        return self._data[-1]
    def empty(self):
        return len(self._data) == 0
''',
        "Explanation": "Stack implemented via Python list: append/pop O(1). Useful for DFS, recursion emulation."
    },
    {
        "Topic": "Queue (ADT)",
        "Summary": "FIFO enqueue/dequeue operations",
        "Code": '''
from collections import deque

class Queue:
    def __init__(self):
        self._data = deque()
    def enqueue(self, x):
        self._data.append(x)
    def dequeue(self):
        return self._data.popleft()
    def empty(self):
        return len(self._data) == 0
''',
        "Explanation": "Use deque for O(1) enqueue/dequeue. List pop(0) is O(n)."
    },
    {
        "Topic": "Binary Search",
        "Summary": "Divide sorted array by half until found",
        "Code": '''
def binary_search(arr, key):
    lo, hi = 0, len(arr)-1
    while lo <= hi:
        mid = (lo + hi)//2
        if arr[mid] == key:
            return mid
        if arr[mid] < key:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
        "Explanation": "Requires sorted arr; O(log n) time, O(1) space."
    },
    {
        "Topic": "Insertion Sort",
        "Summary": "Build sorted array by inserting elements",
        "Code": '''
def insertion_sort(A):
    for j in range(1, len(A)):
        key = A[j]
        i = j-1
        while i >= 0 and A[i] > key:
            A[i+1] = A[i]
            i -= 1
        A[i+1] = key
    return A
''',
        "Explanation": "Best O(n), worst O(n^2). Good for small or nearly sorted data."
    },
    {
        "Topic": "Merge Sort",
        "Summary": "Divide and conquer: split & merge",
        "Code": '''
def merge_sort(A):
    if len(A) < 2:
        return A
    mid = len(A)//2
    left = merge_sort(A[:mid])
    right = merge_sort(A[mid:])
    return merge(left, right)

def merge(L, R):
    res = []
    i = j = 0
    while i < len(L) and j < len(R):
        if L[i] <= R[j]: res.append(L[i]); i+=1
        else: res.append(R[j]); j+=1
    res.extend(L[i:]); res.extend(R[j:])
    return res
''',
        "Explanation": "O(n log n) time, O(n) extra space. Stable sort."
    },
    {
        "Topic": "Quick Sort",
        "Summary": "Partition pivot, sort partitions recursively",
        "Code": '''
def quick_sort(A):
    if len(A) < 2: return A
    pivot = A[len(A)//2]
    left = [x for x in A if x < pivot]
    mid = [x for x in A if x == pivot]
    right = [x for x in A if x > pivot]
    return quick_sort(left) + mid + quick_sort(right)
''',
        "Explanation": "Average O(n log n), worst O(n^2). In-place variants exist."
    },
    {
        "Topic": "Heap: Sift-down & Heapify",
        "Summary": "Convert array into max-heap with siftdown",
        "Code": '''
def siftdown(A, i, n):
    largest = i
    left, right = 2*i+1, 2*i+2
    if left < n and A[left] > A[largest]: largest = left
    if right < n and A[right] > A[largest]: largest = right
    if largest != i:
        A[i], A[largest] = A[largest], A[i]
        siftdown(A, largest, n)

def heapify(A):
    n = len(A)
    for i in range(n//2-1, -1, -1):
        siftdown(A, i, n)
''',
        "Explanation": "Heapify takes O(n) time, builds heap for heapsort or priority queue."
    },
    {
        "Topic": "Heap Sort",
        "Summary": "Sort via repeated extract-max from heap",
        "Code": '''
def heap_sort(A):
    heapify(A)
    n = len(A)
    for i in range(n-1, 0, -1):
        A[0], A[i] = A[i], A[0]
        siftdown(A, 0, i)
    return A
''',
        "Explanation": "In-place, O(n log n) time, O(1) extra space."
    },
    {
        "Topic": "Binary Tree Traversals",
        "Summary": "Preorder, Inorder, Postorder via recursion",
        "Code": '''
class TreeNode:
    def __init__(self, val):
        self.val = val; self.left = None; self.right = None

def preorder(root, res=[]):
    if root: res.append(root.val); preorder(root.left, res); preorder(root.right, res)
    return res

def inorder(root, res=[]):
    if root: inorder(root.left, res); res.append(root.val); inorder(root.right, res)
    return res

def postorder(root, res=[]):
    if root: postorder(root.left, res); postorder(root.right, res); res.append(root.val)
    return res
''',
        "Explanation": "Depth-first visits: root-first, root-middle, root-last respectively."
    },
    {
        "Topic": "Binary Search Tree: Insert & Delete",
        "Summary": "Maintain BST property on insertion/deletion",
        "Code": '''
class BSTNode:
    def __init__(self, val): self.val=val; self.left= self.right=None

def bst_insert(root, val):
    if not root: return BSTNode(val)
    if val <= root.val: root.left = bst_insert(root.left, val)
    else: root.right = bst_insert(root.right, val)
    return root

def bst_delete(root, val):
    if not root: return None
    if val < root.val: root.left = bst_delete(root.left, val)
    elif val > root.val: root.right = bst_delete(root.right, val)
    else:
        if not root.left: return root.right
        if not root.right: return root.left
        succ = root.right
        while succ.left: succ = succ.left
        root.val = succ.val
        root.right = bst_delete(root.right, succ.val)
    return root
''',
        "Explanation": "Insertion O(h), deletion handles 0/1/2 child cases. h=height."
    },
    {
        "Topic": "Graph: Adjacency List",
        "Summary": "Representation as dict of neighbor lists",
        "Code": '''
graph = {
    'A': ['B','C'],
    'B': ['A','D'],
    'C': ['A','D'],
    'D': ['B','C'],
}
''',
        "Explanation": "Space O(n+m). Efficient for sparse graphs."
    },
    {
        "Topic": "Breadth-First Search (BFS)",
        "Summary": "Layer-by-layer using queue",
        "Code": '''
from collections import deque

def bfs(graph, start):
    visited, order = set(), []
    q = deque([start]); visited.add(start)
    while q:
        v = q.popleft(); order.append(v)
        for nei in graph[v]:
            if nei not in visited:
                visited.add(nei); q.append(nei)
    return order
''',
        "Explanation": "Visits nodes by distance from start. O(n+m) time."
    },
    {
        "Topic": "Depth-First Search (DFS)",
        "Summary": "Principles: go deep before backtracking; use a stack or recursion",
        "Code": "# Python code not required for exam; understand the recursive or stack-based approach.",
        "Explanation": "DFS explores as far as possible along each branch before backtracking. Time O(n+m)."
    },
    {
        "Topic": "BFS: Shortest Path Lengths",
        "Summary": "Compute distances in unweighted graph",
        "Code": '''
def bfs_distances(graph, start):
    from collections import deque
    dist = {start:0}; q=deque([start])
    while q:
        v=q.popleft()
        for nei in graph[v]:
            if nei not in dist:
                dist[nei]=dist[v]+1; q.append(nei)
    return dist
''',
        "Explanation": "Unweighted shortest-path tree distances."
    },
    {
        "Topic": "Dijkstra's Algorithm",
        "Summary": "Greedy shortest paths for weighted graph",
        "Code": '''
import heapq

def dijkstra(adj, src):
    dist={v:float('inf') for v in adj}; dist[src]=0
    pq=[(0,src)]; parent={src:None}
    while pq:
        d,u=heapq.heappop(pq)
        if d>dist[u]: continue
        for v,w in adj[u].items():
            nd=d+w
            if nd<dist[v]: dist[v]=nd; parent[v]=u; heapq.heappush(pq,(nd,v))
    return dist, parent
''',
        "Explanation": "Priority-queue yields O((n+m) log n). Builds SPT."
    },
    {
        "Topic": "Kruskal's Algorithm",
        "Summary": "Principles: sort edges by weight, use union-find to avoid cycles",
        "Code": "# Python code not required for exam; focus on greedy MST construction via sorted edges and DSU.",
        "Explanation": "Add smallest edges that don't form a cycle until n-1 edges selected."
    },
    {
        "Topic": "Text Classification Example",
        "Summary": "Principles: convert text to features (bag-of-words), apply Naive Bayes",
        "Code": "# Python code not required for exam; understand vectorization and NB pipeline.",
        "Explanation": "CountVectorizer + MultinomialNB; pipeline for supervised text categorization."
    }
]

# SIDEBAR: Search & TOC
st.sidebar.title("🔎 Search & Navigate")
search = st.sidebar.text_input("Search topics")
filtered = [t for t in cheat_data if search.lower() in t['Topic'].lower()] if search else cheat_data

st.sidebar.markdown("### Topics")
for t in filtered:
    st.sidebar.write(f"- {t['Topic']}")

# Optional summary table
if st.sidebar.checkbox("Show summary table"):
    df = pd.DataFrame([{'Topic':t['Topic'],'Summary':t['Summary']} for t in filtered])
    st.sidebar.dataframe(df, use_container_width=True)

# MAIN
st.title("📘 IE2108 Detailed Python Cheatsheet")
for item in filtered:
    with st.expander(f"🔹 {item['Topic']}"):
        st.markdown(f"**Summary**: {item['Summary']}")
        if item['Code']:
            st.code(item['Code'], language='python')
        st.markdown(f"**Explanation**: {item['Explanation']}")

# DOWNLOAD SUMMARY
df_dl = pd.DataFrame([{'Topic':t['Topic'],'Summary':t['Summary']} for t in filtered])
csv = df_dl.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Summary CSV", data=csv,
                   file_name="ie2108_detailed_summary.csv",
                   mime="text/csv")
