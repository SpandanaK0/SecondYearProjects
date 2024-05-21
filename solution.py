"""
Nathan Gu and Blake Potvin
Sorting Project - Starter
CSE 331 Fall 2023
"""

import random
import time
from typing import TypeVar, List, Callable, Dict, Tuple
from dataclasses import dataclass

T = TypeVar("T")  # represents generic type


# do_comparison is an optional helper function but HIGHLY recommended!!!
def do_comparison(first: T, second: T, comparator: Callable[[T, T], bool], descending: bool) -> bool:
    """
    FILL OUT DOCSTRING
    """
    # checks if the descending flag is True
    if descending:
        # checks if the second one is bigger than the first one
        return comparator(second,first)
    else:
        # else checks if the first number is greater than second number
        return comparator(first,second)


def selection_sort(data: List[T], *, comparator: Callable[[T, T], bool] = lambda x, y: x < y,
                   descending: bool = False) -> None:
    """
    Given a list of values, sort that list in-place using the selection sort algorithm and the provided comparator,
    and perform the sort in descending order if descending is True
    """
    # calculates the length of the data
    n = len(data)
    # in
    for i in range(n - 1):
        min_idx = i

        for j in range(i + 1, n):
            # if descending is true
            if descending:
                # if this statement is true J should come before min_idx
                # min_idx is the second number
                if comparator(data[min_idx], data[j]):
                    # min index is updated to j
                    min_idx = j
            # if its not descending
            else:
                # if th
                if comparator(data[j], data[min_idx]):
                    min_idx = j

        if i != min_idx:
            data[i], data[min_idx] = data[min_idx], data[i]


def bubble_sort(data: List[T], *, comparator: Callable[[T, T], bool] = lambda x, y: x < y,
                descending: bool = False) -> None:
    """
    Given a list of values, sort that list in-place using the bubble sort algorithm and the provided comparator,
    and perform the sort in descending order if descending is True.
    """
    n = len(data)

    for i in range(n):
        swapped = False

        for j in range(0, n - i - 1):
            if descending:
                if comparator(data[j], data[j + 1]):
                    data[j], data[j + 1] = data[j + 1], data[j]
                    swapped = True
            else:
                if comparator(data[j + 1], data[j]):
                    data[j], data[j + 1] = data[j + 1], data[j]
                    swapped = True

        # If no two elements were swapped in inner loop, the list is already sorted
        if not swapped:
            break


def insertion_sort(data: List[T], *, comparator: Callable[[T, T], bool] = lambda x, y: x < y,
                   descending: bool = False) -> None:
    """
    Given a list of values, sort that list in-place using the insertion sort algorithm and the provided comparator,
    and perform the sort in descending order if descending is True.
    """
    # Get the length of the input list
    n = len(data)

    # Iterate over the list starting from the second element (index 1)
    for i in range(1, n):
        # Store the current element to be inserted into the sorted subarray
        current_element = data[i]

        # Initialize a variable to keep track of the position where current_element
        # should be inserted in the sorted subarray
        j = i - 1

        # Compare current_element with elements in the sorted subarray using the comparator function
        while j >= 0 and do_comparison(current_element,data[j],comparator,descending):
            data[j + 1] = data[j]
            j -= 1

        # Insert the current_element at the correct position in the sorted subarray
        data[j + 1] = current_element


def hybrid_merge_sort(data: List[T], *, threshold: int = 12,
                      comparator: Callable[[T, T], bool] = lambda x, y: x < y, descending: bool = False) -> None:
    """
    Given a list of values, sort that list using a hybrid sort with the merge sort and insertion sort
    algorithms and the provided comparator, and perform the sort in descending order if descending is True.
    The function should use insertion_sort to sort lists once their size is less than or equal to threshold, and
    otherwise perform a merge sort.
    """


    n = len(data)
    if n < 2:
        return 

    if len(data) <= threshold:
        insertion_sort(data, comparator=comparator, descending=descending)

    else:
        mid = n // 2
        S1 = data[0:mid]
        S2 = data[mid:n]

        hybrid_merge_sort(S1,threshold=threshold,  comparator=comparator, descending=descending)
        hybrid_merge_sort(S2,threshold=threshold, comparator=comparator, descending=descending)

        i = j = 0
        while i<len(S1) and j < len(S2):
            if do_comparison(S1[i],S2[j],comparator,descending):
                data[i+j] = S1[i]
                i = i + 1
            else:
                data[i+j] = S2[j]
                j = j + 1


        if i < len(S1):
            data[i+j:] = S1[i:]
        else:
            data[i+j:] = S2[j:]



def maximize_rewards(item_prices: List[int]) -> (List[Tuple[int, int]], int):
    

    if len(item_prices) % 2 == 1:
        return ([], -1)
    if len(item_prices) < 2:
        return ([], -1)

    # Sort the item_prices list
    hybrid_merge_sort(item_prices)

    max_profit = 0
    pairs = []
    length = len(item_prices)

    target = item_prices[0] + item_prices[-1]

    for i in range(length // 2):
        j = len(item_prices) - 1 - i
        if item_prices[i] + item_prices[j] != target:
            return [],-1
        left_num = item_prices[i]
        right_num = item_prices[length-1-i]
        max_profit += (left_num*right_num)
        pairs.append((left_num,right_num))

    return (pairs, max_profit)

def quicksort(data) -> None:
    """
    Sorts a list in place using quicksort
    :param data: Data to sort
    """


    def quicksort_inner(first, last) -> None:
        """
        Sorts portion of list at indices in interval [first, last] using quicksort

        :param first: first index of portion of data to sort
        :param last: last index of portion of data to sort
        """
        # List must already be sorted in this case
        if first >= last:
            return

        left = first
        right = last

        # Need to start by getting median of 3 to use for pivot
        # We can do this by sorting the first, middle, and last elements
        midpoint = (right - left) // 2 + left
        if data[left] > data[right]:
            data[left], data[right] = data[right], data[left]
        if data[left] > data[midpoint]:
            data[left], data[midpoint] = data[midpoint], data[left]
        if data[midpoint] > data[right]:
            data[midpoint], data[right] = data[right], data[midpoint]
        # data[midpoint] now contains the median of first, last, and middle elements
        pivot = data[midpoint]
        # First and last elements are already on right side of pivot since they are sorted
        left += 1
        right -= 1

        # Move pointers until they cross
        while left <= right:
            # Move left and right pointers until they cross or reach values which could be swapped
            # Anything < pivot must move to left side, anything > pivot must move to right side
            #
            # Not allowing one pointer to stop moving when it reached the pivot (data[left/right] == pivot)
            # could cause one pointer to move all the way to one side in the pathological case of the pivot being
            # the min or max element, leading to infinitely calling the inner function on the same indices without
            # ever swapping
            while left <= right and data[left] < pivot:
                left += 1
            while left <= right and data[right] > pivot:
                right -= 1

            # Swap, but only if pointers haven't crossed
            if left <= right:
                data[left], data[right] = data[right], data[left]
                left += 1
                right -= 1

        quicksort_inner(first, left - 1)
        quicksort_inner(left, last)

    # Perform sort in the inner function
    quicksort_inner(0, len(data) - 1)
