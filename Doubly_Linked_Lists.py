"""
Project 2
CSE 331 F23 (Onsay)
Authored By: Hank Murdock
Originally Authored By: Andrew McDonald & Alex Woodring & Andrew Haas & Matt Kight & Lukas Richters & Sai Ramesh
solution.py
"""

from typing import TypeVar, List

# for more information on type hinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")  # represents generic type
Node = TypeVar("Node")  # represents a Node object (forward-declare to use in Node __init__)


# pro tip: PyCharm auto-renders docstrings (the multiline strings under each function definition)
# in its "Documentation" view when written in the format we use here. Open the "Documentation"
# view to quickly see what a function does by placing your cursor on it and using CTRL + Q.
# https://www.jetbrains.com/help/pycharm/documentation-tool-window.html


class Node:
    """
    Implementation of a doubly linked list node.
    Do not modify.
    """
    __slots__ = ["value", "next", "prev", "child"]

    def __init__(self, value: T, next: Node = None, prev: Node = None, child: Node = None) -> None:
        """
        Construct a doubly linked list node.

        :param value: value held by the Node.
        :param next: reference to the next Node in the linked list.
        :param prev: reference to the previous Node in the linked list.
        :return: None.
        """
        self.next = next
        self.prev = prev
        self.value = value

        # The child attribute is only used for the application problem
        self.child = child

    def __repr__(self) -> str:
        """
        Represents the Node as a string.

        :return: string representation of the Node.
        """
        return f"Node({str(self.value)})"

    __str__ = __repr__


class DLL:
    """
    Implementation of a doubly linked list without padding nodes.
    Modify only below indicated line.
    """
    __slots__ = ["head", "tail", "size"]

    def __init__(self) -> None:
        """
        Construct an empty doubly linked list.

        :return: None.
        """
        # creates a new node and sets it to None
        self.head = self.tail = None
        # keep tracks of the number of nodes in the list
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the DLL as a string.

        :return: string representation of the DLL.
        """
        # initializes a new list, collects the string values of each node
        result = []
        # initializes a variable = to the first variable
        node = self.head
        # continues till the node is = to None
        while node is not None:
            #changes the Node value to a string and adds it to a list
            result.append(str(node))
            #moves to the next node
            node = node.next
        # After the loop it adds the <->, joins the elements in the result list into a single string
        return " <-> ".join(result)

    def __str__(self) -> str:
        """
        Represent the DLL as a string.

        :return: string representation of the DLL.
        """
        # returns the string that represents the DLL
        return repr(self)

    # +++++++++++++++++++++++MODIFY BELOW +++++++++++++++++++++++++++++++++++++++++++++#

    def empty(self) -> bool:
        """
        Returns a boolean indicating whether the DLL is empty.
        """
        # returns false if its None
        return self.head is None

    def push(self, val: T, back: bool = True) -> None:
        """
        Adds a Node containing val to the back (or front) of the DLL and updates size accordingly.
        returns None
        """
        # creates a new node
        new_val = Node(val)
        # if the head is None
        if self.head is None:
            # the new value is set as the head and tail
            self.head = self.tail = new_val
        # if back is true, so it wants to add at the end of the list
        elif back:
            # sets the value before the new val = to the tail
            new_val.prev = self.tail
            # sets the variable next to the tail = to the new val
            self.tail.next = new_val
            # sets the tail = to new val
            self.tail = new_val
        # if it wants to add it in the front
        else:
            # connects the link by making the val next to the new val = to the head, establishes the front link
            new_val.next = self.head
            # establishes the backwards link,
            self.head.prev = new_val
            # changes the head to the new value
            self.head = new_val
        # adds one to the size since u addes a new node
        self.size += 1

    def pop(self, back: bool = True) -> None:
        """
        Removes a Node from the back (or front) of the DLL and updates size accordingly.
        Returns: None.
        """
        # if the list is empty return none
        if self.head is None:
            return
        # if it wants to remove something from the end
        if back:
            # checks if it only has one node
            if self.size == 1:
                # sets the head and the tail to none
                self.head = self.tail = None
            # if its greater than 1
            else:
                # sets the tail to the value of the prev
                self.tail = self.tail.prev
                # makes the variable next to the new tail = none
                self.tail.next= None
        # if they want you to remove the value in the front
        else:
            #checks if their is only one value
            if self.size == 1:
                # if so changes it to none
                self.head = self.tail = None
            # if theres more than one
            else:
                # sets the head = to the var next to the existing head
                self.head = self.head.next
                # sets the var behind the new head = None
                self.head.prev = None
        # removes 1 from the count
        self.size -= 1

    def list_to_dll(self, source: List[T]) -> None:
        """
        Creates a DLL from a standard Python list. If there are already nodes in the DLL,
        the DLL should be cleared and replaced by source. Returns: None.
        """
        # converts the input into a doubly linked list

        # sets the tail and head to None since its empty
        self.head = self.tail = None
        # sets the size = 0 since there isn't anything inside of it yet
        self.size = 0
        # goes through the input
        for i in source:
            # creates a new node fore each value of i
            new_node = Node(i)
            # checks if the head is empty
            if self.head is None:
                # creates only one node
                self.head = self.tail = new_node
            # if there's more than one
            else:
                # creates a backwards link with the existing tail
                new_node.prev = self.tail
                # creates a forward link with the new variable
                self.tail.next = new_node
                #sets the tail to the new var
                self.tail = new_node
            #adds one the size
            self.size += 1

    def dll_to_list(self) -> List[T]:
        """
        Creates a standard Python list from a DLL.
        Returns: list[T] containing the values of the nodes in the DLL.
        """
        # creates an empty python list
        python_list = []
        # sets var to the first node
        current = self.head
        # while the var != None
        while current:
            # add the value to the list
            python_list.append(current.value)
            # move to the next value
            current = current.next
        # return the lsit
        return python_list


    def _find_nodes(self, val: T, find_first: bool = False) -> List[Node]:
        """
        Construct list of Node with value val in the DLL and returns the associated Node object list
        Returns: list of Node objects in the DLL whose value is val. If val does not exist in the DLL, returns empty list.
        """
        # create an empty list
        result = []
        # start at the first node
        current = self.head

        # while var != None
        while current:
            # if the node value = the input val
            if current.value == val:
                # add it to the empty list
                result.append(current)
                # checks if its the first find
                if find_first:
                    break  # Stop after finding the first matching node
            current = current.next

        return result

    def find(self, val: T) -> Node:
        """
        Finds first Node with value val in the DLL and returns the associated Node object.
        Returns: first Node object in the DLL whose value is val. If val does not exist in the DLL, return None.
        """
        # calls the find function, set to true so it will stop once it find the value the first time
        result = self._find_nodes(val, find_first=True)
        # if result = true, found the item
        if result:
            # return the first value in the list
            return result[0]  # Return the first matching node or None if not found
        else:
            # else return none
            return None

    def find_all(self, val: T) -> List[Node]:
        """
        Finds all Node objects with value val in the DLL and returns a standard Python list of the associated Node objects.
        Returns: standard Python list of all Node objects in the DLL whose value is val. If val does not exist in the DLL, returns an empty list.
        """
        # set to false cause it keeps searching for that value till the end
        return self._find_nodes(val, find_first=False)

    def remove_node(self, to_remove: Node) -> None:
        """
        Given a reference to a node in the linked list, remove it
        Returns None
        """
        # if the list is empty return false
        if self.size == 0:
            return False  # DLL is empty, removal unsuccessful

        # Case 1: Removing the head node
        # if the item u want to remove is == to the head
        if to_remove == self.head:
            # set the head = to the variable next to the head
            self.head = to_remove.next
            # if the head is not none
            if self.head:
                # set the head = None
                self.head.prev = None
            # if head is empty
            else:
                # set the tail = to none
                self.tail = None

        # Case 2: Removing the tail node
        # if the var == to the tail
        elif to_remove == self.tail:
            # set the tail = to the prev value
            self.tail = to_remove.prev
            # if tail != none
            if self.tail:
                # set the old tail into none
                self.tail.next = None
            # if tail = none
            else:
                # set the head to none too
                self.head = None

        # Case 3: Removing a node in the middle
        # if its not the head or the tail
        else:

            prev_node = to_remove.prev
            next_node = to_remove.next
            prev_node.next = next_node
            next_node.prev = prev_node
        # remove 1 from the size
        self.size -= 1
        # return true when u remove soemthing
        return True

    def remove(self, val: T) -> bool:
        """
        removes first Node with value val in the DLL.
        Returns: True if a Node with value val was found and removed from the DLL, else False.
        """
        node_to_remove = self.find(val)
        if node_to_remove:
            return self.remove_node(node_to_remove)
        else:
            return False

    def remove_all(self, val: T) -> int:
        """
        removes all Node objects with value val in the DLL. See note 7.
        Returns: number of Node objects with value val removed from the DLL. If no node containing val exists in the DLL, returns 0.
        """
        # calls the find_all value
        nodes_to_remove = self.find_all(val)
        # set a counter to see how many u removed
        count_removed = 0
        # goes through each node in nodes_to_remove
        for node in nodes_to_remove:
            # checks if it correctly removed the node
            if self.remove_node(node):
                # adds one to the counter
                count_removed += 1
        # returns the counter
        return count_removed

    def reverse(self) -> None:
        """
        Reverses the DLL in-place by modifying all next and prev references of Node objects in DLL.
        Updates self.head and self.tail accordingly. See note 8.
        Return None
        """
        if self.size <= 1:
            return  # Nothing to reverse if DLL is empty or has only one node

        current = self.head
        # while var != None
        while current:
            # Inside the loop, this line stores the reference to the next node in the next_node variable.
            # This is necessary because we will be modifying the current node's next and prev pointers.
            next_node = current.next
            # Swap prev and next pointers to reverse the direction
            current.prev, current.next = current.next, current.prev
            # After swapping the pointers, we move current to the next node
            current = next_node

        # Update head and tail pointers to the new positions
        self.head, self.tail = self.tail, self.head


class BrowserHistory:

    def __init__(self, homepage: str):
        """
        FILL IN DOCSTRING
        """
        # creates a new DLL
        self.dll = DLL()
        # (adds) the homepage string to the doubly linked list
        self.dll.push(homepage)
        # gets the self.current_page attribute to the head node of the doubly linked list
        self.current_page = self.dll.head

    def get_current_url(self) -> str:
        """
        Get the URL of the current page in the web browser's history.
        Returns the URL of the current web page as a string.
        """
        # returns current page
        return self.current_page.value

    def visit(self, url: str) -> None:
        """
        Visit a new URL, updating the browser's history accordingly.
        clears the forward history, adds the new URL to the history,
        and sets the current page to the newly visited URL
        """

        # Clear forward history
        while self.current_page.next:
            #  this line removes the next page from the browsing history
            self.dll.remove_node(self.current_page.next)

        # Add the new URL and update current page
        self.dll.push(url)
        #  this line updates the current_page attribute to point to the tail
        self.current_page = self.dll.tail

    def backward(self) -> None:
        """
        Return to the last page in history, if there is no previous page don’t go back.
        Returns: None.
        """

        if self.current_page.prev is None:
            return
        # If there is a previous page, this line retrieves the previous page node and assigns it to the
        prev_page = self.current_page.prev
        # current pg = to the prev page
        self.current_page = prev_page

        while metrics_api(self.current_page.value):
            prev_page = self.current_page.prev
            self.current_page = prev_page

    def forward(self) -> None:
        """
        Visit the page ahead of the current one in history, if currently on the most recent page then stay at the same page.
        Returns: None.
        """
        if self.current_page.next:
            next_page = self.current_page.next
            while next_page and metrics_api(next_page.value):
                next_page = next_page.next
            if next_page:
                self.current_page = next_page

# DO NOT MODIFY
intervention_set = set(['https://malicious.com', 'https://phishing.com', 'https://malware.com'])
def metrics_api(url: str) -> bool:
    """
    Uses the intervention_set to determine what URLs are bad and which are good.

    :param url: The url to check.
    :returns: True if this is a malicious website, False otherwise.
    """
    if url in intervention_set:
        return True
    return False
