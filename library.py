#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:17:58 2024

@author: gizemmutlu
"""

class Library:
    def __init__(self, filename):
        self.filename = filename
        self.file = None

    def __del__(self):
        try:
            if self.file:
                self.file.close()
                print(f"File '{self.filename}' closed successfully.")
        except AttributeError:
            pass
        except Exception as e:
            print(f"Error: Unable to close file '{self.filename}': {e}")

    def list_books(self):
        try:
            with open(self.filename, 'r', encoding="UTF-8") as file:
                for line in file:
                    book_info = line.strip().split(',')
                    if len(book_info) == 4:
                        print(f"Book Title: {book_info[0]}, Author: {book_info[1]}, Publication Year: {book_info[2]}, Pages: {book_info[3]}")
                    else:
                        print(f"Invalid book information format: {line.strip()}")
        except FileNotFoundError:
            print("Error: File not found.")
        except Exception as e:
            print(f"Error: {e}")

    def add_book(self):
        title = input("Enter book title: ")
        author = input("Enter book author: ")
        year = input("Enter publication year: ")
        pages = input("Enter number of pages: ")
        try:
            with open(self.filename, 'a+', encoding="UTF-8") as file:
                file.write(f"{title},{author},{year},{pages}\n")
            print("Book added successfully.")
        except Exception as e:
            print(f"Error: {e}")

    def remove_book(self, title):
        try:
            lines = []
            with open(self.filename, 'r', encoding="UTF-8") as file:
                for line in file:
                    book_info = line.strip().split(',')
                    if len(book_info) == 4 and title.lower() not in line.lower():
                        lines.append(line)
            with open(self.filename, 'w', encoding="UTF-8") as file:
                for line in lines:
                    file.write(line)
            print("Book removed successfully.")
        except FileNotFoundError:
            print("Error: File not found.")
        except Exception as e:
            print(f"Error: {e}")

def main():
    lib = Library("books.txt")

    while True:
        print("\nMenu:")
        print("1) List Books")
        print("2) Add Book")
        print("3) Remove Book")
        print("4) Quit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            lib.list_books()
        elif choice == '2':
            lib.add_book()
        elif choice == '3':
            title = input("Enter the title of the book to remove: ")
            lib.remove_book(title)
        elif choice == '4':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()


