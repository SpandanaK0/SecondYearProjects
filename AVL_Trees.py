
import math
from typing import TypeVar, Generator, List, Tuple, Optional
from collections import deque
import json
from queue import SimpleQueue

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")  # represents generic type
# represents a Node object (forward-declare to use in Node __init__)
Node = TypeVar("Node")
# represents a custom type used in application
AVLWrappedDictionary = TypeVar("AVLWrappedDictionary")


class Node:
    """
    Implementation of an BST and AVL tree node.
    Do not modify.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["value", "parent", "left", "right", "height"]

    def __init__(self, value: T, parent: Node = None,
                 left: Node = None, right: Node = None) -> None:
        """
        Construct an AVL tree node.

        :param value: value held by the node object
        :param parent: ref to parent node of which this node is a child
        :param left: ref to left child node of this node
        :param right: ref to right child node of this node
        """
        self.value = value
        self.parent, self.left, self.right = parent, left, right
        self.height = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"

    def __str__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return repr(self)


####################################################################################################

class BinarySearchTree:
    """
    Implementation of an BSTree.
    Modify only below indicated line.
    """

    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty BST tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the BSTree as a string.

        :return: string representation of the BST tree
        """
        if self.origin is None:
            return "Empty BST Tree"

        lines = pretty_print_binary_tree(self.origin, 0, False, '-')[0]
        return "\n" + "\n".join((line.rstrip() for line in lines))

    def __str__(self) -> str:
        """
        Represent the BSTree as a string.

        :return: string representation of the BSTree
        """
        return repr(self)

    def visualize(self, filename="bst_visualization.svg"):
        """
        Generates an svg image file of the binary tree.

        :param filename: The filename for the generated svg file. Should end with .svg.
        Defaults to output.svg
        """
        svg_string = svg(self.origin, node_radius=20)
        print(svg_string, file=open(filename, 'w'))
        return svg_string

    ########################################
    # Implement functions below this line. #
    ########################################

    def height(self, root: Node) -> int:
        """
        Calculates and returns the height of a subtree in the BSTree,
        handling the case where root is None. Note that an empty
        subtree has a height of -1.
        """
        if root is None:
            return -1
        return root.height

    def insert(self, root: Node, val: T) -> None:
        """
        Inserts a node with the value val into the subtree rooted at root,
        returning the root of the balanced subtree after the insertion.
        If val already exists in the tree, the function does nothing.
        """
        if root is None:
            new_node = Node(val)
            new_node.height = 0  # height of a new leaf node is 0
            if self.origin is None:
                self.origin = new_node  # Update origin if tree was empty
            self.size += 1
            return new_node

            # Recursive Case: Traverse to find the correct position
        if val < root.value:
            if root.left is None:
                root.left = Node(val)
                root.left.height = 0
                self.size += 1
            else:
                self.insert(root.left, val)
        elif val > root.value:
            if root.right is None:
                root.right = Node(val)
                root.right.height = 0
                self.size += 1
            else:
                self.insert(root.right, val)

            # Update height of the current node
        root.height = max(self.height(root.left), self.height(root.right)) + 1

        # val already in the tree, do nothing
        return root

    def remove(self, root: Node, val: T) -> Optional[Node]:
        """
        Removes a node with the specified value from the binary search tree.
        Parameters:
        - root (Node): The root node of the binary search tree.
        - val (T): The value to be removed from the tree.
        Returns:
        - Optional[Node]: The root of the modified binary search tree after removal.

        If the value is not found in the tree, the original tree is returned.
        """
        if root is None:
            return None

            # Search for the node to be removed
        if val < root.value:
            root.left = self.remove(root.left, val)
        elif val > root.value:
            root.right = self.remove(root.right, val)
        else:
            # Node with only one child or no child
            if root.left is None:
                temp = root.right
                root = None
                self.size -= 1
                return temp
            elif root.right is None:
                temp = root.left
                root = None
                self.size -= 1
                return temp

            # Node with two children: Get the predecessor (maximum in the left subtree)
            temp = root.left
            while temp.right:
                temp = temp.right

            # Copy the predecessor's content to this node
            root.value = temp.value

            # Delete the predecessor
            root.left = self.remove(root.left, temp.value)

            # Update the height of the current node
        if root is not None:
            root.height = max(self.height(root.left), self.height(root.right)) + 1

        return root

    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        Searches for a node with the specified value in the binary search tree.

        Parameters:
        - root (Node): The root node of the binary search tree.
        - val (T): The value to be searched in the tree.

        Returns:
        - Optional[Node]: The node with the specified value if found; otherwise, None.

        If the value is not found, the method returns None. If multiple nodes have the same
        value, the method returns the first occurrence encountered during the search.
        """
        if root is None or root.value == val:
            return root

        if val < root.value:
            return self.search(root.left, val) or root
        else:
            return self.search(root.right, val) or root


class AVLTree:
    """
    Implementation of an AVL tree.
    Modify only below indicated line.
    """

    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty AVL tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree as a string.

        :return: string representation of the AVL tree
        """
        if self.origin is None:
            return "Empty AVL Tree"

        return super(AVLTree, self).__repr__()

    def __str__(self) -> str:
        """
        Represent the AVLTree as a string.

        :return: string representation of the BSTree
        """
        return repr(self)

    def visualize(self, filename="avl_tree_visualization.svg"):
        """
        Generates an svg image file of the binary tree.

        :param filename: The filename for the generated svg file. Should end with .svg.
        Defaults to output.svg
        """
        svg_string = svg(self.origin, node_radius=20)
        print(svg_string, file=open(filename, 'w'))
        return svg_string

    ########################################
    # Implement functions below this line. #
    ########################################

    def height(self, root: Node) -> int:
        """
        Calculates and returns the height of a subtree in the BSTree,
        handling the case where root is None. Note that an empty
        subtree has a height of -1.
        """
        if root is None:
            return -1
        return root.height

    def left_rotate(self, root: Node) -> Optional[Node]:
        """
        Performs a right rotation on the specified node in a binary search tree.

        Parameters:
        - root (Node): The node to be rotated.

        Returns:
        - Node: The new root after the right rotation.

        This method rotates the specified node and its left child to the right,
        updating the necessary pointers and heights. If the rotated node is the
        origin of the tree, the origin is updated accordingly.
        """
        if root is None:
            return root

        new_root = root.right
        right_left_child = new_root.left

        # Update the left child of the new root
        new_root.left = root
        root.right = right_left_child

        # Update parents
        if right_left_child is not None:
            right_left_child.parent = root
        new_root.parent = root.parent

        # If the root has a parent, update its child to point to the new root
        if root.parent is not None:
            if root is root.parent.left:
                root.parent.left = new_root
            else:
                root.parent.right = new_root

        # Update the parent of the rotated root
        root.parent = new_root

        # If the root is the origin of the tree, update the origin
        if root is self.origin:
            self.origin = new_root

        # Update heights
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + max(self.height(new_root.left), self.height(new_root.right))

        return new_root

    def right_rotate(self, root: Node) -> Optional[Node]:
        """
        Performs a right rotation on the specified node in a binary search tree.

        Parameters:
        - root (Node): The node to be rotated.

        Returns:
        - Optional[Node]: The new root after the right rotation. If the input node is
          None, the method returns None.

        This method rotates the specified node and its left child to the right,
        updating the necessary pointers and heights. If the rotated node is the
        origin of the tree, the origin is updated accordingly.
        """
        if root is None:
            return root

        new_root = root.left
        left_right_child = new_root.right

        # Update the right child of the new root
        new_root.right = root
        root.left = left_right_child

        # Update parents
        if left_right_child is not None:
            left_right_child.parent = root
        new_root.parent = root.parent

        # If the root has a parent, update its child to point to the new root
        if root.parent is not None:
            if root is root.parent.right:
                root.parent.right = new_root
            else:
                root.parent.left = new_root

        # Update the parent of the rotated root
        root.parent = new_root

        # If the root is the origin of the tree, update the origin
        if root is self.origin:
            self.origin = new_root

        # Update heights
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + max(self.height(new_root.left), self.height(new_root.right))

        return new_root

    def balance_factor(self, root: Node) -> int:
        """
        Calculates the balance factor of the specified node in a binary search tree.

        Parameters:
        - root (Node): The node for which the balance factor is to be calculated.

        Returns:
        - int: The balance factor, defined as the difference between the height of
          the left subtree and the height of the right subtree. If the node is None,
          the balance factor is 0.

        The balance factor indicates the balance of the subtree rooted at the given
        node. A balance factor of 0 indicates a balanced subtree, while positive and
        negative values indicate an imbalance towards the left or right subtree,
        respectively.
        """
        if root is None:
            return 0
        return self.height(root.left) - self.height(root.right)

    def rebalance(self, root: Node) -> Optional[Node]:
        """
        Rebalances the binary search tree rooted at the specified node.

        Parameters:
        - root (Node): The root node of the binary search tree to be rebalanced.

        Returns:
        - Optional[Node]: The new root of the rebalanced subtree. If the input node is
          None, the method returns None.

        This method checks the balance factor of the given node and performs rotations
        as needed to restore balance according to four possible cases:
        - Left-Left (LL) Case: Right rotation.
        - Right-Right (RR) Case: Left rotation.
        - Left-Right (LR) Case: Left rotation on left child followed by right rotation on root.
        - Right-Left (RL) Case: Right rotation on right child followed by left rotation on root.

        If no rebalancing is required, the original root is returned.
        """
        if root is None:
            return None

        balance_factor = self.balance_factor(root)

        # Left-Left (LL) Case
        if balance_factor > 1 and self.balance_factor(root.left) >= 0:
            return self.right_rotate(root)

        # Right-Right (RR) Case
        if balance_factor < -1 and self.balance_factor(root.right) <= 0:
            return self.left_rotate(root)

        # Left-Right (LR) Case
        if balance_factor > 1 and self.balance_factor(root.left) < 0:
            root.left = self.left_rotate(root.left)
            return self.right_rotate(root)

        # Right-Left (RL) Case
        if balance_factor < -1 and self.balance_factor(root.right) > 0:
            root.right = self.right_rotate(root.right)
            return self.left_rotate(root)

        return root

    def insert(self, root: Node, val: T) -> Optional[Node]:
        """
        INSERT DOCSTRING HERE
        """
        if root is None:
            self.size += 1
            new_node = Node(val)  # Assuming Node constructor handles height and parent
            if self.origin is None or self.origin == root:
                self.origin = new_node  # Update origin if inserting at tree root or a None root
            return new_node

        if val < root.value:
            root.left = self.insert(root.left, val)
            if root.left:  # Set parent pointer if node is not None
                root.left.parent = root
        elif val > root.value:
            root.right = self.insert(root.right, val)
            if root.right:  # Set parent pointer if node is not None
                root.right.parent = root
        else:
            return root  # Value already exists

        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root = self.rebalance(root)

        # Update origin if rotation occurred at the root
        if root == self.origin:
            self.origin = new_root

        return new_root

    def remove(self, root: Node, val: T) -> Optional[Node]:
        """
        Inserts a new node with the specified value into the binary search tree rooted at the given node.

        Parameters:
        - root (Node): The root node of the binary search tree where the insertion begins.
        - val (T): The value to be inserted into the tree.

        Returns:
        - Optional[Node]: The new root of the modified subtree after the insertion.
          If the input root is None, a new node is created, and the method returns it as the new root.

        This method recursively inserts a new node into the appropriate subtree based on the comparison
        of the given value with the values in the tree. After insertion, it updates the height of the
        current node and performs rebalancing if necessary. If a rotation occurs at the root, the origin
        of the tree is updated accordingly. If the value already exists in the tree, the original root
        is returned, and no modification is made.
        """
        if root is None:
            return None

        if val < root.value:
            root.left = self.remove(root.left, val)
        elif val > root.value:
            root.right = self.remove(root.right, val)
        else:
            # Node with only one child or no child
            if root.left is None:
                temp = root.right
                root = None
                self.size -= 1
                return temp
            elif root.right is None:
                temp = root.left
                root = None
                self.size -= 1
                return temp

            # Node with two children: get the predecessor
            temp = self.max(root.left)

            # Swap values with the predecessor
            root.value = temp.value

            # Remove the predecessor
            root.left = self.remove(root.left, temp.value)

            # Update height and rebalance
        if root:
            root.height = 1 + max(self.height(root.left), self.height(root.right))
            root = self.rebalance(root)

        return root

    def min(self, root: Node) -> Optional[Node]:
        """
        This function searches for and returns the Node
        containing the smallest value within the subtree rooted at root.

        Parameters:
        root (Node): The root of the subtree within which to search for the minimum value.
        Returns: A Node object that holds the smallest value in the subtree rooted at root.
        """
        current = root
        while current and current.left:
            current = current.left
        return current

    def max(self, root: Node) -> Optional[Node]:
        """
        This function searches for and returns the Node
        containing the largest value within the subtree rooted at root.

        Parameters:
        root (Node): The root of the subtree within which to search for the maximum value.
        Returns: A Node object that holds the largest value in the subtree rooted at root.
        """
        current = root
        while current and current.right:
            current = current.right
        return current

    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        This function searches for the Node with the value val within the subtree rooted at root.
        If the value val does not exist within this subtree, the function will return the Node under which
        val would be inserted if it were added to the tree.

        Parameters:
        root (Node): The root of the subtree within which to search for val.
        val (T): The value to be searched for within the subtree rooted at root.
        Returns: A Node object containing val if it exists within the subtree,
        and if not, the Node under which val would be inserted as a child.

        """
        if root is None or root.value == val:
            return root

        if val < root.value:
            return self.search(root.left, val) or root
        else:
            return self.search(root.right, val) or root

    def inorder(self, root: Node) -> Generator[Node, None, None]:
        """
        This function performs an inorder traversal (left, current, right) of
        the subtree rooted at root, generating the nodes one at a time using a Python generator.

        Parameters:
        root (Node): The root node of the current subtree being traversed.
        Returns: A generator yielding the nodes of the subtree in inorder.
        """
        if root is not None:
            yield from self.inorder(root.left)
            yield root
            yield from self.inorder(root.right)

    def __iter__(self) -> Generator[Node, None, None]:
        """
        This method makes the AVL tree class iterable, allowing you to use it in loops like for node in tree.
        Returns: A generator yielding the nodes of the tree in inorder.
        """
        return self.inorder(self.origin)

    def preorder(self, root: Node) -> Generator[Node, None, None]:
        """
        This function performs a preorder traversal (current, left, right) of the subtree rooted at root,
        generating the nodes one at a time using a Python generator.

        Parameters:
        root (Node): The root node of the current subtree being traversed.
        Returns: A generator yielding the nodes of the subtree in preorder.
        """
        if root is not None:
            yield root
            yield from self.preorder(root.left)
            yield from self.preorder(root.right)

    def postorder(self, root: Node) -> Generator[Node, None, None]:
        """
        This function performs a postorder traversal (left, right, current)
        of the subtree rooted at root, generating the nodes one at a time using a Python generator.
        Utilize yield to produce individual nodes as they are encountered, and yield from for recursive
        calls to postorder.

        Parameters:
        root (Node): The root node of the current subtree being traversed.
        Returns: A generator yielding the nodes of the subtree in postorder.
        A StopIteration exception is raised once all nodes have been yielded.
        """
        if root is not None:
            yield from self.postorder(root.left)
            yield from self.postorder(root.right)
            yield root

    def levelorder(self, root: Node) -> Generator[Node, None, None]:
        """
        This function performs a level-order (breadth-first) traversal of the
        subtree rooted at root, generating the nodes one at a time using a Python generator.

        Parameters:
        root (Node): The root node of the current subtree being traversed.
        Returns: A generator yielding the nodes of the subtree in level-order.
        A StopIteration exception is raised once all nodes have been yielded.
        """
        if root is None:
            return

        q = SimpleQueue()
        q.put(root)

        while not q.empty():
            node = q.get()
            yield node

            if node.left:
                q.put(node.left)
            if node.right:
                q.put(node.right)


####################################################################################################


class AVLWrappedDictionary:
    """
    Implementation of a helper class which will be used as tree node values in the
    NearestNeighborClassifier implementation.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["key", "dictionary"]

    def __init__(self, key: float) -> None:
        """
        Construct a AVLWrappedDictionary with a key to search/sort on and a dictionary to hold data.

        :param key: floating point key to be looked up by.
        """
        self.key = key
        self.dictionary = {}

    def __repr__(self) -> str:
        """
        Represent the AVLWrappedDictionary as a string.

        :return: string representation of the AVLWrappedDictionary.
        """
        pprinted_dict = json.dumps(self.dictionary, indent=2)
        return f"key: {self.key} dict:{self.dictionary}"

    def __str__(self) -> str:
        """
        Represent the AVLWrappedDictionary as a string.

        :return: string representation of the AVLWrappedDictionary.
        """
        return repr(self)

    def __eq__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement == operator to compare 2 AVLWrappedDictionaries by key only.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating whether keys of AVLWrappedDictionaries are equal
        """
        return abs(self.key - other.key) < 1e-6

    def __lt__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement < operator to compare 2 AVLWrappedDictionarys by key only.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating ordering of AVLWrappedDictionaries
        """
        return self.key < other.key and not abs(self.key - other.key) < 1e-6

    def __gt__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement > operator to compare 2 AVLWrappedDictionaries by key only.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating ordering of AVLWrappedDictionaries
        """
        return self.key > other.key and not abs(self.key - other.key) < 1e-6


class NearestNeighborClassifier:
    """
    Implementation of a one-dimensional nearest-neighbor classifier with AVL tree lookups.
    Modify only below indicated line.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["resolution", "tree"]

    def __init__(self, resolution: int) -> None:
        """
        Construct a one-dimensional nearest neighbor classifier with AVL tree lookups.
        Data are assumed to be floating point values in the closed interval [0, 1].

        :param resolution: number of decimal places the data will be rounded to, effectively
                           governing the capacity of the model - for example, with a resolution of
                           1, the classifier could maintain up to 11 nodes, spaced 0.1 apart - with
                           a resolution of 2, the classifier could maintain 101 nodes, spaced 0.01
                           apart, and so on - the maximum number of nodes is bounded by
                           10^(resolution) + 1.
        """
        self.tree = AVLTree()
        self.resolution = resolution

        # pre-construct lookup tree with AVLWrappedDictionary objects storing (key, dictionary)
        # pairs, but which compare with <, >, == on key only
        for i in range(10 ** resolution + 1):
            w_dict = AVLWrappedDictionary(key=(i / 10 ** resolution))
            self.tree.insert(self.tree.origin, w_dict)

    def __repr__(self) -> str:
        """
        Represent the NearestNeighborClassifier as a string.

        :return: string representation of the NearestNeighborClassifier.
        """
        return f"NNC(resolution={self.resolution}):\n{self.tree}"

    def __str__(self) -> str:
        """
        Represent the NearestNeighborClassifier as a string.

        :return: string representation of the NearestNeighborClassifier.
        """
        return repr(self)

    def visualize(self, filename: str = "nnc_visualization.svg") -> str:
        svg_string = svg(self.tree.origin, 48, nnc_mode=True)
        print(svg_string, file=open(filename, 'w'))
        return svg_string

    ########################################
    # Implement functions below this line. #
    ########################################

    def fit(self, data: List[Tuple[float, str]]) -> None:
        """
        Fits the model with the given data, updating the internal AVL tree.

        Parameters:
        - data (List[Tuple[float, str]]): A list of tuples containing numerical values and corresponding labels.

        Returns:
        - None

        This method processes the input data, rounding each numerical value to the specified resolution and
        updates the internal AVL tree with the rounded values. For each rounded value, it searches for the
        corresponding node in the tree. If no exact match is found, a new AVLWrappedDictionary is created
        and inserted into the tree. The count of each label 'y' in the associated dictionary is then updated.
        """
        for x, y in data:
            # Round x to the specified resolution
            rounded_x = round(x, self.resolution)

            # Find the corresponding node in the tree
            node = self.tree.search(self.tree.origin, AVLWrappedDictionary(rounded_x))

            if node is None or node.value.key != rounded_x:
                # If no exact match, create a new AVLWrappedDictionary
                new_dict = AVLWrappedDictionary(rounded_x)
                self.tree.insert(self.tree.origin, new_dict)
                node = self.tree.search(self.tree.origin, new_dict)

            # Update the count of y in the dictionary
            if y in node.value.dictionary:
                node.value.dictionary[y] += 1
            else:
                node.value.dictionary[y] = 1

    def predict(self, x: float, delta: float) -> str:
        """
         Predicts the label for a given numerical value within a specified range.

        Parameters:
        - x (float): The numerical value for which to make a prediction.
        - delta (float): The range within which to consider values for prediction.

        Returns:
        - str: The predicted label for the given value, or None if no prediction can be made.

        This method rounds the input value 'x' to the specified resolution and searches for nodes in the
        internal AVL tree whose keys fall within the range [rounded_x - delta, rounded_x + delta]. For each
        found node, it aggregates label counts from associated dictionaries. The predicted label is then
        determined by selecting the label with the highest count across all dictionaries within the range.
        If no labels are found or if there is a tie in label counts, the method returns None.
        """
        rounded_x = round(x, self.resolution)

        # Initialize a dictionary to count occurrences of each label
        label_counts = {}

        # Search for nodes in the tree whose keys are within ± delta of the rounded x value
        for i in range(int((rounded_x - delta) * 10 ** self.resolution),
                       int((rounded_x + delta) * 10 ** self.resolution) + 1):
            key = i / 10 ** self.resolution
            node = self.tree.search(self.tree.origin, AVLWrappedDictionary(key))
            if node and node.value.key == key:
                for label, count in node.value.dictionary.items():
                    if label in label_counts:
                        label_counts[label] += count
                    else:
                        label_counts[label] = count

        # Determine the most common label y across all these dictionaries
        if label_counts:
            return max(label_counts, key=label_counts.get)

        return None

####################################################################################################


"""
For the curious students, the following functions are used to visualize and compare the performance
of BinarySearchTree and AVLTree under extreme conditions. You do not need to modify (or run) these 
functions, but you are welcome to play around with them if you wish. 

You should know that AVLTree is faster than BinarySearchTree in the worst case, but how much faster?
The following functions will help you answer this question, shall you choose to try them out.

Uncomment the line under "if __name__ == '__main__':" to run the performance comparison (after completing
the rest of the project) and you will be greeted with a plot of the results. The function will require
matplotlib, make sure to install it if you do not have it already.
"""


def compare_times(structure: dict, sizes: List[int], trial: int) -> dict:
    """
    Comparing time on provide data structures in the worst case of BST tree
    :param structure: provided data structures
    :param sizes: size of test input
    :param trial: number of trials to test
    :return: dict with list of average times per input size for each algorithm
    """
    import sys
    import time
    result = {}
    sys.stdout.write('\r')
    sys.stdout.write('Start...\n')
    total = len(sizes) * len(structure)
    count = 0
    for algorithm, value in structure.items():
        ImplementedTree = value
        if algorithm not in result:
            result[algorithm] = []
        for size in sizes:
            sum_times = 0
            for _ in range(trial):
                tree = ImplementedTree()
                start = time.perf_counter()
                for i in range(size):
                    tree.insert(tree.origin, i)
                for i in range(size, -1, -1):
                    tree.remove(tree.origin, i)
                end = time.perf_counter()
                sum_times += (end - start)
            count += 1
            result[algorithm].append(sum_times / trial)
            sys.stdout.write("[{:<20s}] {:d}%\n".format('=' * ((count * 20) // total),
                                                        count * 100 // total))
            sys.stdout.flush()
    return result


def plot_time_comparison():
    """
    Use compare_times to make a time comparison of normal binary search tree and AVL tree
    in a worst case scenario.
    Requires matplotlib. Comment this out if you do not wish to install matplotlib.
    """
    import matplotlib.pyplot as plt
    import sys
    sys.setrecursionlimit(2010)
    structures = {
        "bst": BinarySearchTree,
        "avl": AVLTree
    }
    sizes = [4, 5, 6, 7, 8, 9, 10, 25, 50, 100, 300, 500, 1000, 2000]
    trials = 5
    data = compare_times(structures, sizes, trials)

    plt.style.use('seaborn-colorblind')
    plt.figure(figsize=(12, 8))

    for structure in structures:
        plt.plot(sizes, data[structure], label=structure)
    plt.legend()
    plt.xlabel("Input Size")
    plt.ylabel("Time to Sort (sec)")
    plt.title("BST vs AVL")
    plt.show()


_SVG_XML_TEMPLATE = """
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
<style>
    .value {{
        font: 300 16px monospace;
        text-align: center;
        dominant-baseline: middle;
        text-anchor: middle;
    }}
    .dict {{
        font: 300 16px monospace;
        dominant-baseline: middle;
    }}
    .node {{
        fill: lightgray;
        stroke-width: 1;
    }}
</style>
<g stroke="#000000">
{body}
</g>
</svg>
"""

_NNC_DICT_BOX_TEXT_TEMPLATE = """<text class="dict" y="{y}" xml:space="preserve">
    <tspan x="{label_x}" dy="1.2em">{label}</tspan>
    <tspan x="{bracket_x}" dy="1.2em">{{</tspan>
    {values}
    <tspan x="{bracket_x}" dy="1.2em">}}</tspan>
</text>
"""


def pretty_print_binary_tree(root: Node, curr_index: int, include_index: bool = False,
                             delimiter: str = "-", ) -> \
        Tuple[List[str], int, int, int]:
    """
    Taken from: https://github.com/joowani/binarytree

    Recursively walk down the binary tree and build a pretty-print string.
    In each recursive call, a "box" of characters visually representing the
    current (sub)tree is constructed line by line. Each line is padded with
    whitespaces to ensure all lines in the box have the same length. Then the
    box, its width, and start-end positions of its root node value repr string
    (required for drawing branches) are sent up to the parent call. The parent
    call then combines its left and right sub-boxes to build a larger box etc.
    :param root: Root node of the binary tree.
    :type root: binarytree.Node | None
    :param curr_index: Level-order_ index of the current node (root node is 0).
    :type curr_index: int
    :param include_index: If set to True, include the level-order_ node indexes using
        the following format: ``{index}{delimiter}{value}`` (default: False).
    :type include_index: bool
    :param delimiter: Delimiter character between the node index and the node
        value (default: '-').
    :type delimiter:
    :return: Box of characters visually representing the current subtree, width
        of the box, and start-end positions of the repr string of the new root
        node value.
    :rtype: ([str], int, int, int)
    .. _Level-order:
        https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search
    """
    if root is None:
        return [], 0, 0, 0

    line1 = []
    line2 = []
    if include_index:
        node_repr = "{}{}{}".format(curr_index, delimiter, root.value)
    else:
        if type(root.value) == AVLWrappedDictionary:
            node_repr = f'{root.value},h={root.height},' \
                        f'⬆{str(root.parent.value.key) if root.parent else "None"}'
        else:
            node_repr = f'{root.value},h={root.height},' \
                        f'⬆{str(root.parent.value) if root.parent else "None"}'

    new_root_width = gap_size = len(node_repr)

    # Get the left and right sub-boxes, their widths, and root repr positions
    l_box, l_box_width, l_root_start, l_root_end = pretty_print_binary_tree(
        root.left, 2 * curr_index + 1, include_index, delimiter
    )
    r_box, r_box_width, r_root_start, r_root_end = pretty_print_binary_tree(
        root.right, 2 * curr_index + 2, include_index, delimiter
    )

    # Draw the branch connecting the current root node to the left sub-box
    # Pad the line with whitespaces where necessary
    if l_box_width > 0:
        l_root = (l_root_start + l_root_end) // 2 + 1
        line1.append(" " * (l_root + 1))
        line1.append("_" * (l_box_width - l_root))
        line2.append(" " * l_root + "/")
        line2.append(" " * (l_box_width - l_root))
        new_root_start = l_box_width + 1
        gap_size += 1
    else:
        new_root_start = 0

    # Draw the representation of the current root node
    line1.append(node_repr)
    line2.append(" " * new_root_width)

    # Draw the branch connecting the current root node to the right sub-box
    # Pad the line with whitespaces where necessary
    if r_box_width > 0:
        r_root = (r_root_start + r_root_end) // 2
        line1.append("_" * r_root)
        line1.append(" " * (r_box_width - r_root + 1))
        line2.append(" " * r_root + "\\")
        line2.append(" " * (r_box_width - r_root))
        gap_size += 1
    new_root_end = new_root_start + new_root_width - 1

    # Combine the left and right sub-boxes with the branches drawn above
    gap = " " * gap_size
    new_box = ["".join(line1), "".join(line2)]
    for i in range(max(len(l_box), len(r_box))):
        l_line = l_box[i] if i < len(l_box) else " " * l_box_width
        r_line = r_box[i] if i < len(r_box) else " " * r_box_width
        new_box.append(l_line + gap + r_line)

    # Return the new box, its width and its root repr positions
    return new_box, len(new_box[0]), new_root_start, new_root_end


def svg(root: Node, node_radius: int = 16, nnc_mode=False) -> str:
    """
    Taken from: https://github.com/joowani/binarytree

    Generate SVG XML.
    :param root: Generate SVG for tree rooted at root
    :param node_radius: Node radius in pixels (default: 16).
    :type node_radius: int
    :return: Raw SVG XML.
    :rtype: str
    """
    tree_height = root.height
    scale = node_radius * 3
    xml = deque()
    nodes_for_nnc_visualization: list[AVLWrappedDictionary] = []

    def scale_x(x: int, y: int) -> float:
        diff = tree_height - y
        x = 2 ** (diff + 1) * x + 2 ** diff - 1
        return 1 + node_radius + scale * x / 2

    def scale_y(y: int) -> float:
        return scale * (1 + y)

    def add_edge(parent_x: int, parent_y: int, node_x: int, node_y: int) -> None:
        xml.appendleft(
            '<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>'.format(
                x1=scale_x(parent_x, parent_y),
                y1=scale_y(parent_y),
                x2=scale_x(node_x, node_y),
                y2=scale_y(node_y),
            )
        )

    def add_node(node_x: int, node_y: int, node: Node) -> None:
        x, y = scale_x(node_x, node_y), scale_y(node_y)
        xml.append(
            f'<circle class="node" cx="{x}" cy="{y}" r="{node_radius}"/>')

        if nnc_mode:
            nodes_for_nnc_visualization.append(node.value)
            xml.append(
                f'<text class="value" x="{x}" y="{y + 5}">key={node.value.key}</text>')
        else:
            xml.append(
                f'<text class="value" x="{x}" y="{y + 5}">{node.value}</text>')

    current_nodes = [root.left, root.right]
    has_more_nodes = True
    y = 1

    add_node(0, 0, root)

    while has_more_nodes:

        has_more_nodes = False
        next_nodes: List[Node] = []

        for x, node in enumerate(current_nodes):
            if node is None:
                next_nodes.append(None)
                next_nodes.append(None)
            else:
                if node.left is not None or node.right is not None:
                    has_more_nodes = True

                add_edge(x // 2, y - 1, x, y)
                add_node(x, y, node)

                next_nodes.append(node.left)
                next_nodes.append(node.right)

        current_nodes = next_nodes
        y += 1

    svg_width = scale * (2 ** tree_height)
    svg_height = scale * (2 + tree_height)
    if nnc_mode:

        line_height = 20
        box_spacing = 10
        box_margin = 5
        character_width = 10

        max_key_count = max(
            map(lambda obj: len(obj.dictionary), nodes_for_nnc_visualization))
        box_height = (max_key_count + 3) * line_height + box_margin

        def max_length_item_of_node_dict(node: AVLWrappedDictionary):
            # Check if dict is empty so max doesn't throw exception
            if len(node.dictionary) > 0:
                item_lengths = map(lambda pair: len(
                    str(pair)), node.dictionary.items())
                return max(item_lengths)
            return 0

        max_value_length = max(
            map(max_length_item_of_node_dict, nodes_for_nnc_visualization))
        box_width = max(max_value_length * character_width, 110)

        boxes_per_row = svg_width // box_width
        rows_needed = math.ceil(
            len(nodes_for_nnc_visualization) / boxes_per_row)

        nodes_for_nnc_visualization.sort(key=lambda node: node.key)
        for index, node in enumerate(nodes_for_nnc_visualization):
            curr_row = index // boxes_per_row
            curr_column = index % boxes_per_row

            box_x = curr_column * (box_width + box_spacing)
            box_y = curr_row * (box_height + box_spacing) + svg_height
            box = f'<rect x="{box_x}" y="{box_y}" width="{box_width}" ' \
                  f'height="{box_height}" fill="white" />'
            xml.append(box)

            value_template = '<tspan x="{value_x}" dy="1.2em">{key}: {value}</tspan>'
            text_x = box_x + 10

            def item_pair_to_svg(pair):
                return value_template.format(key=pair[0], value=pair[1], value_x=text_x + 10)

            values = map(item_pair_to_svg, node.dictionary.items())
            text = _NNC_DICT_BOX_TEXT_TEMPLATE.format(
                y=box_y,
                label=f"key = {node.key}",
                label_x=text_x,
                bracket_x=text_x,
                values='\n'.join(values)
            )
            xml.append(text)

        svg_width = boxes_per_row * (box_width + box_spacing * 2)
        svg_height += rows_needed * (box_height + box_spacing * 2)

    return _SVG_XML_TEMPLATE.format(
        width=svg_width,
        height=svg_height,
        body="\n".join(xml),
    )


if __name__ == "__main__":
    # plot_time_comparison()
    pass
