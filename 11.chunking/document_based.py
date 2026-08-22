# 3. Document-based splitting

# Document-based splitting means splitting according to the natural structure of a document.
# PDF
# ├── Page 1
# ├── Page 2
# └── Page 3

# Python Code
# ├── Import section
# ├── Class section
# ├── Function section
# └── Main logic

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter
)

text = """
class Student:

    def __init__(self, name, age, course):
        self.name = name
        self.age = age
        self.course = course

    def invoke(self):
        return {
            "name": self.name,
            "age": self.age,
            "course": self.course
        }


student1 = Student("Sankha", 25, "Computer Science")
student2 = Student("Rahul", 22, "Mechanical Engineering")

print(student1.invoke())
print(student2.invoke())
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)