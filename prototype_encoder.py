from PIL import Image
import numpy as np

class Node:
    def __init__(self, char=None, freq=0):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

def build_frequency_table(data):
    frequency_table = {}
    for char in data:
        if char in frequency_table:
            frequency_table[char] += 1
        else:
            frequency_table[char] = 1
    return frequency_table

def build_binary_tree(frequency_table):
    nodes = [Node(char, freq) for char, freq in frequency_table.items()]
    while len(nodes) > 1:
        nodes.sort(key=lambda x: x.freq)
        left = nodes.pop(0)
        right = nodes.pop(0)
        merged_node = Node(freq=left.freq + right.freq)
        merged_node.left = left
        merged_node.right = right
        nodes.append(merged_node)
    return nodes[0]

def build_custom_tree():
    # Construct a custom binary tree with the specified characters
    root = Node()
    root.left = Node()
    root.left.left = Node(char='E', freq=1)
    root.left.right = Node(char='A', freq=1)
    root.right = Node()
    root.right.left = Node(char='C', freq=1)
    root.right.right = Node()
    root.right.right.left = Node(char='F', freq=1)
    root.right.right.right = Node(char='H', freq=1)
    root.right.right.right.left = Node(char='I', freq=1)
    root.right.right.right.right = Node(char='L', freq=1)
    root.right.right.right.right.left = Node(char='M', freq=1)
    root.right.right.right.right.right = Node(char='N', freq=1)
    root.right.right.right.right.right.left = Node(char='O', freq=1)
    root.right.right.right.right.right.right = Node(char='0', freq=1)
    root.right.right.right.right.right.right.left = Node(char='1', freq=1)
    return root

def build_custom_codes(root):
    codes = {}
    def traverse(node, code=""):
        if node is not None:
            if node.char is not None:
                codes[node.char] = code
            traverse(node.left, code + "0")
            traverse(node.right, code + "1")
    traverse(root)
    
    # Ensure all characters are included in the custom codes dictionary
    for char in "EACFHI LMNO01":
        if char not in codes:
            codes[char] = ''
    return codes

def compress_data(data, custom_codes):
    compressed_data = ""
    for char in data:
        compressed_data += custom_codes.get(char, '')
    return compressed_data

def decompress_data(compressed_data, root):
    decompressed_data = ""
    current_node = root
    for bit in compressed_data:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right
        if current_node.char is not None:
            decompressed_data += current_node.char
            current_node = root
    return decompressed_data

def calculate_mse(image1, image2):
    # Resize image2 to match the size of image1
    image2 = image2.resize(image1.size)
    # Convert images to numpy arrays
    array1 = np.array(image1.convert('L'))  # Convert to grayscale
    array2 = np.array(image2.convert('L'))  # Convert to grayscale
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((array1 - array2) ** 2)
    return mse

# Example usage:
input_image_path = "input_image.jpeg"
output_compressed_text_file = "compressed_image.txt"
output_decompressed_image_path = "decompressed_image.jpeg"

# Extract image format from the input image path
input_image_format = input_image_path.split('.')[-1]

# Read binary data from a file
with open(input_image_path, "rb") as file:
    original_binary_data = file.read()

# Convert binary data to a string of characters
original_characters = "".join([chr(byte) for byte in original_binary_data])

# Build frequency table from the data
frequency_table = build_frequency_table(original_characters)

# Build custom binary tree
custom_tree_root = build_custom_tree()

# Build custom codes from the binary tree
custom_codes = build_custom_codes(custom_tree_root)

# Compress the data using custom coding
compressed_data = compress_data(original_characters, custom_codes)

# Write compressed data to a text file
with open(output_compressed_text_file, 'w') as file:
    file.write(compressed_data)

# Read compressed data from the text file
with open(output_compressed_text_file, 'r') as file:
    compressed_data = file.read()

# Decompress the data using custom coding
decompressed_characters = decompress_data(compressed_data, custom_tree_root)

# Convert decompressed characters to binary data
decompressed_binary_data = bytearray(decompressed_characters.encode())

# Calculate the image size based on the length of binary data
image_size = int(np.sqrt(len(decompressed_binary_data)))

# Reshape the binary data to form an image array
decompressed_image_array = np.array(decompressed_binary_data[:image_size**2]).reshape((image_size, image_size))

# Convert the image array to an image and save it
decompressed_image = Image.fromarray(decompressed_image_array.astype(np.uint8) * 255)  # Scale to 0-255 range
# Save the decompressed image using the same format as the original image
decompressed_image.save(output_decompressed_image_path, format=input_image_format)

# Open the original image using PIL
original_image = Image.open(input_image_path)

# Calculate Mean Squared Error (MSE) between the original and decompressed images
mse = calculate_mse(original_image, decompressed_image)

# Open the decompressed image using PIL and display it
decompressed_image.show()

print("Mean Squared Error (MSE) between the original and decompressed images:", mse)

