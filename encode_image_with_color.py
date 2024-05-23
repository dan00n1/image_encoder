import os
import numpy as np
import heapq
from PIL import Image
from collections import Counter

class Node:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequency_table):
    heap = [Node(char, freq) for char, freq in frequency_table.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged_node = Node(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged_node)
    
    return heap[0]

def build_huffman_codes(root):
    codes = {}
    
    def traverse(node, code=""):
        if node:
            if node.char is not None:
                codes[node.char] = code
            traverse(node.left, code + "0")
            traverse(node.right, code + "1")
    
    traverse(root)
    return codes

def compress_data(data, huffman_codes):
    return ''.join(huffman_codes[byte] for byte in data)

def decompress_data(compressed_data, root):
    decompressed_data = bytearray()
    current_node = root
    for bit in compressed_data:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right
        
        if current_node.char is not None:
            decompressed_data.append(current_node.char)
            current_node = root
    
    return bytes(decompressed_data)

def image_to_bytes(image):
    return np.array(image).tobytes()

def bytes_to_image(data, shape):
    return Image.fromarray(np.frombuffer(data, dtype=np.uint8).reshape(shape))

CURRENT_DIRECTORY = os.getcwd()
RESOURCE_DIRECTORY = os.path.join(CURRENT_DIRECTORY, 'resources')
IMAGE_DIRECTORY = os.path.join(RESOURCE_DIRECTORY, 'images')
DECOMPRESSED_IMAGE_DIRECTORY = os.path.join(IMAGE_DIRECTORY, 'decompressed')
FILES_DIRECTORY = os.path.join(RESOURCE_DIRECTORY, 'files')

# Example usage:
input_image_path = os.path.join(IMAGE_DIRECTORY, "test_image.jpeg")
output_compressed_text_file = os.path.join(FILES_DIRECTORY, "test_image.txt")
output_decompressed_image_path = os.path.join(DECOMPRESSED_IMAGE_DIRECTORY, "test_image_decompressed.jpeg")

# Read the input image
original_image = Image.open(input_image_path)

# Resize the image to make it smaller
new_size = (original_image.width, original_image.height)
resized_image = original_image.resize(new_size)

# Convert the resized image to bytes
resized_array = np.array(resized_image)
resized_bytes = image_to_bytes(resized_image)

# Build frequency table from the resized data
frequency_table = Counter(resized_bytes)

# Build Huffman tree
huffman_tree_root = build_huffman_tree(frequency_table)

# Build Huffman codes from the binary tree
huffman_codes = build_huffman_codes(huffman_tree_root)

# Compress the data using Huffman coding
compressed_data = compress_data(resized_bytes, huffman_codes)

# Write compressed data to a text file
with open(output_compressed_text_file, 'w') as file:
    file.write(compressed_data)

# Read compressed data from the text file
with open(output_compressed_text_file, 'r') as file:
    compressed_data = file.read()

# Decompress the data using Huffman coding
decompressed_bytes = decompress_data(compressed_data, huffman_tree_root)

# Rebuild the image from decompressed data
decompressed_image = bytes_to_image(decompressed_bytes, resized_array.shape)

# Convert the decompressed image to RGB if it is in RGBA mode
if decompressed_image.mode == 'RGBA':
    decompressed_image = decompressed_image.convert('RGB')

# Save the decompressed image
decompressed_image.save(output_decompressed_image_path)

# Display the decompressed image
decompressed_image.show()
