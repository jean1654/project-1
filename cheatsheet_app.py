import streamlit as st
import pandas as pd

# Define each topic, summary, and detailed explanation + code
cheat_data = [
    {
        "Topic": "Sequential Search",
        "Summary": "Loop through list, return index if match, else -1",
        "Code": '''
def sequential_search(arr, key):
    for i in range(len(arr)):
        if arr[i] == key:
            return i
    return -1

# Example
print(sequential_search([4, 2, 9, 1], 9))  # Output: 2
''',
        "Explanation": "Sequential search checks each element one by one. It's simple but inefficient for large lists (O(n))."
    },
    {
        "Topic": "Binary Search",
        "Summary": "While loop with midpoint check, narrow down search",
        "Code": '''
def binary_search(arr, key):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == key:
            return mid
        elif arr[mid] < key:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# Must use a sorted list
print(binary_search([1, 3, 5, 7, 9], 5))  # Output: 2
''',
        "Explanation": "Binary search splits the array in half each iteration. Efficient for sorted lists (O(log n))."
    },
    {
        "Topic": "Insertion Sort",
        "Summary": "Nested loop, insert by shifting elements left",
        "Code": '''
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

print(insertion_sort([4, 3, 2, 1]))  # Output: [1, 2, 3, 4]
''',
        "Explanation": "Insertion sort builds a sorted list by inserting elements at their correct position. O(n^2) worst case."
    },
    {
        "Topic": "Merge Sort",
        "Summary": "Recursive split, then merge sorted subarrays",
        "Code": '''
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

print(merge_sort([5, 3, 8, 6]))  # Output: [3, 5, 6, 8]
''',
        "Explanation": "Merge sort is a divide-and-conquer algorithm. Time complexity: O(n log n)."
    },
    {
        "Topic": "Stack (using list)",
        "Summary": "Use list with append/pop, LIFO",
        "Code": '''
stack = []
stack.append(1)
stack.append(2)
print(stack.pop())  # Output: 2
print(stack.pop())  # Output: 1
''',
        "Explanation": "Stack uses Last-In-First-Out. Python lists support push/pop with append() and pop()."
    },
    {
        "Topic": "Queue (using list)",
        "Summary": "Use list with append/pop(0), FIFO",
        "Code": '''
queue = []
queue.append(1)
queue.append(2)
print(queue.pop(0))  # Output: 1
print(queue.pop(0))  # Output: 2
''',
        "Explanation": "Queue uses First-In-First-Out. Use `collections.deque` for better performance."
    },
    {
        "Topic": "BFS (Graph Traversal)",
        "Summary": "Use queue, visited list, layer-by-layer",
        "Code": '''
def bfs(graph, start):
    visited = set()
    queue = [start]
    order = []
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            order.append(node)
            queue.extend([n for n in graph[node] if n not in visited])
    return order

G = {1: [2, 3], 2: [4], 3: [], 4: []}
print(bfs(G, 1))  # Output: [1, 2, 3, 4]
''',
        "Explanation": "BFS explores graph in breadth-first manner using a queue. Time complexity O(n + m)."
    },
    # More topics like BST insertion, tree traversals, heapsort, etc., can be added similarly
]

st.set_page_config(page_title="IE2108 Python Full Cheatsheet", layout="wide")
st.title("📘 IE2108 Full Python Coding Cheatsheet")

for item in cheat_data:
    with st.expander(f"🔹 {item['Topic']}"):
        st.write(f"**Summary**: {item['Summary']}")
        st.code(item['Code'], language='python')
        st.markdown(f"**Explanation**: {item['Explanation']}")

# Optional download
df_download = pd.DataFrame([{"Topic": x["Topic"], "Summary": x["Summary"]} for x in cheat_data])
csv = df_download.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Summary CSV", data=csv, file_name="ie2108_python_summary.csv", mime="text/csv")
