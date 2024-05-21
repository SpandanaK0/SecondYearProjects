
from __future__ import annotations
from typing import Optional, TypeVar, List, Tuple

T = TypeVar("T")


class HashNode:
    """
    DO NOT EDIT
    """
    __slots__ = ["key", "value", "deleted"]

    def __init__(self, key: Optional[str], value: Optional[T], deleted: bool = False) -> None:
        self.key: str = key  # type: ignore (assume these will not be accessed if deleted is True)
        self.value: T = value  # type: ignore
        self.deleted = deleted

    def __str__(self) -> str:
        return f"HashNode({self.key}, {self.value})"

    __repr__ = __str__

    def __eq__(self, other: HashNode) -> bool:
        return self.key == other.key and self.value == other.value \
            if isinstance(other, HashNode) \
            else False


class HashTable:
    """
    Hash Table Class
    """
    __slots__ = ['capacity', 'size', 'table', 'prime_index']

    primes = (
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
        109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
        233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359,
        367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491,
        499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641,
        643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787,
        797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
        947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,
        1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213,
        1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321,
        1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481,
        1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601,
        1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733,
        1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877,
        1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017,
        2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143,
        2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297,
        2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423,
        2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593,
        2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713,
        2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851,
        2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011,
        3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181,
        3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323,
        3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467,
        3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607,
        3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739,
        3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907,
        3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049,
        4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211,
        4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, 4327, 4337, 4339, 4349,
        4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513,
        4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657,
        4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813,
        4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969, 4973,
        4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113,
        5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297,
        5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443,
        5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591,
        5623, 5639, 5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743,
        5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857, 5861, 5867, 5869, 5879,
        5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073,
        6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221,
        6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359,
        6361, 6367, 6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521, 6529, 6547, 6551,
        6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701,
        6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857,
        6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997,
        7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187,
        7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349,
        7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507, 7517, 7523, 7529,
        7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669,
        7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829,
        7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919)

    def __init__(self, capacity: int = 8) -> None:
        """
        DO NOT EDIT
        Initializes hash table
        :param capacity: capacity of the hash table
        """
        self.capacity = capacity
        self.size = 0
        self.table: List[Optional[HashNode]] = [None] * capacity

        i = 0
        while HashTable.primes[i] < self.capacity:
            i += 1
        self.prime_index = i - 1

    def __eq__(self, other: HashTable) -> bool:
        """
        DO NOT EDIT
        Equality operator
        :param other: other hash table we are comparing with this one
        :return: bool if equal or not
        """
        if self.capacity != other.capacity or self.size != other.size:
            return False
        for i in range(self.capacity):
            if self.table[i] != other.table[i]:
                return False
        return True

    def __str__(self) -> str:
        """
        DO NOT EDIT
        Represents the table as a string
        :return: string representation of the hash table
        """
        represent = ""
        bin_no = 0
        for item in self.table:
            represent += "[" + str(bin_no) + "]: " + str(item) + '\n'
            bin_no += 1
        return represent

    __repr__ = __str__

    def _hash_1(self, key: str) -> int:
        """
        ---DO NOT EDIT---
        Converts a string x into a bin number for our hash table
        :param key: key to be hashed
        :return: bin number to insert hash item at in our table, None if key is an empty string
        """
        if key == "":
            return 0
        hashed_value = 0

        for char in key:
            hashed_value = 181 * hashed_value + ord(char)
        return hashed_value % self.capacity

    def _hash_2(self, key: str) -> int:
        """
        ---DO NOT EDIT---
        Converts a string x into a hash
        :param key: key to be hashed
        :return: a hashed value
        """
        if key == "":
            return 0
        hashed_value = 0

        for char in key:
            hashed_value = 181 * hashed_value + ord(char)

        prime = HashTable.primes[self.prime_index]

        hashed_value = prime - (hashed_value % prime)
        if hashed_value % 2 == 0:
            hashed_value += 1
        return hashed_value

    # ========== Modify below ========== #

    def __len__(self) -> int:
        """
    gets the size of the hash table
    :return: int that is size of hash table
    time complexity: O(1)
    """
        return self.size # gets the length of the hash table

    def __setitem__(self, key: str, value: T) -> None:
        """
    sets the value of a key in the hash table
    :param key: key to be hashed
    :param value: value to be set
    :return: None
    time complexity: O(1) worse case O(n)
    """
        self._insert(key, value)

    def __getitem__(self, key: str) -> T:
        """
        looks up the value of a key in the hash table
        :param key: key to be hashed
        :return: value of key
        time complexity: O(1)
        worst - O(N)
        """
        node = self._get(key) # gets the value
        if node is None: # if code is non-existent
            raise KeyError(key) # if key not found it calls for an error key
        return node.value # returns the value associated with that key

    def __delitem__(self, key: str) -> None:
        """
        deletes a key from the hash table
        :param key: key to be hashed
        :return: None
        time complexity: O(1)
        WORST - O(N)
        """
        node = self._get(key) # stores the value of key
        if node is None: #if the node is not in the list
            raise KeyError(key) # calls the key error
        self._delete(key) # if it's in the list then it calls the delete function

    def __contains__(self, key: str) -> bool:
         """
        determines if node with key exists in hash table
        :param key: key to be checked
        :return: bool if key exists
        time complexity: O(1)
        WORST O(N)
        """
         node = self._get(key) #stores the value
         return node is not None # returns true if the node is not None

    def _hash(self, key: str, inserting: bool = False) -> int:
        """
        probes for a bin to insert or get a key using double hashing
        :param key: key to be hashed
        :param inserting: bool if inserting or not
        :return: int index of bin
        BEST O(1)
        WORST O(N)
        """
        index = self._hash_1(key) # stores the value
        if self.table[index] is None or self.table[index].key == key: # checks if the  is None or = to the key given in the function
            return index # returns the value

        if inserting: # if inserting is true
            step = self._hash_2(key)   # stores a value from the second hash function, the step
            while self.table[index] is not None and not self.table[index].deleted: # runs till it reaches a None or a deleted box
                index = (index + step) % self.capacity # allows you to find the next available spot
        else:
            step = self._hash_2(key)
            while self.table[index] is not None and self.table[index].key != key: # runs till its = None or != key
                index = (index + step) % self.capacity # does index + step % the size of the hash table

        return index


    def _insert(self, key: str, value: T) -> None:
        """
        adds a HashNode to hash table
        requires _grow to be called if load factor is greater than 0.5
        :param key: key associated with stored value
        :param value: value to be stored
        :return: None
        time complexity: O(1)

        """
        index = self._hash(key)  # the index of the key
        node = self.table[index] # retrieves the node at that index

        # If the node already exists with the same key, update its value
        if node is not None and not node.deleted and node.key == key:
            node.value = value
        else:
            # Insert a new HashNode
            # find the index for insertion
            index = self._hash(key, inserting=True)
            # insert a hash key at the calculated index
            self.table[index] = HashNode(key, value)
            # increase the size of the hash
            self.size += 1

            # Check the load factor and call _grow if necessary
            if self.size / self.capacity >= 0.5:
                # increase the capacity of the table
                self._grow()


    def _get(self, key: str) -> Optional[HashNode]:
        """
        finds HashNode with given key in hash table
        :param key: key we are lookingup
        :return: HashNode with given key; None if not found
        time complexity: O(1)

        """
        index = self._hash(key) # the index of they key
        node = self.table[index] # retrieves the node
        if node is None or node.deleted or node.key != key: # if the node is none or is deleted or is != to key
            return None # return None
        return node # or return the Node

    def _delete(self, key: str) -> None:
        """
        deletes HashNode with given key from hash table
            If the node is found assign its key and value to None, and set the deleted flag to True
        :param key: key to be deleted
        :return: None
        time complexity: O(1)

        """
        index = self._hash(key) # stores the index of the key
        node = self.table[index] # stores the (key, value)
        if node is None or node.deleted or node.key != key: # if the node is None or delete or != key leave the function
            return
        # if its found
        node.key = None # set the key to none
        node.value = None # set the value to none
        node.deleted = True # set the function delete to true
        self.size -= 1 # decrease the size by 1

    def _grow(self) -> None:
        """
        doubles the capacity of existing hash table
            don't rehash deleted HashNodes
            Must update self.prime_index; the value of self.prime_index should be the index of the largest prime
            smaller than self.capacity in the HashTable.primes tuple.
        :return: None
        time complexity: O(n)
        """
        self.capacity = self.capacity * 2 # doubles the capacity of the hash table

        # Find the largest prime smaller than or equal to the new capacity
        prime_index = 0
        # as long as the prime index value is < self. capacity  add 1
        while HashTable.primes[prime_index] < self.capacity:
            prime_index += 1
        # after the last time the while loop runs you get the next biggest number index after the capacity
        prime_index -= 1
        self.prime_index = prime_index

        # CREATES  a new hash with the new capacity
        new_table = [None] * self.capacity
        old_table = self.table
        self.table = new_table
        self.size = 0
        # re insert the things in the old table into the new table
        for node in old_table:
            if node is not None and not node.deleted:
                self._insert(node.key, node.value)

        # Update the table and capacity after moving nodes
        #self.capacity = self.capacity
        self.prime_index = prime_index

    def update(self, pairs: List[Tuple[str, T]] = []) -> None:
         """
        updates hash table using an iterable of key value pairs
            if value already exists, update it; else enter into table
        :param pairs: list of key value pairs being updated
        :return: None
        time complexity: O(M), M is length of pairs
        WORST O(M*N)
        """
        for pair in pairs:
            if self._get(pair[0]) is not None:
                # checks if it already exists
                self._get(pair[0]).value = pair[1]
            else:
                # if it doens't exists it inserts a new key and value
                self._insert(pair[0], pair[1])

    def keys(self) -> List[str]:
        """
        makes a list that contains all the keys in the table
        :return: list of keys
        time complexity: O(n)
        """
        keyList = []
        # goes through the list
        for node in self.table:
            if node is not None:
                # add the nodes into the keylist list
                keyList.append(node.key)
            if node is not None and node.deleted:
                # remove that node if Deleted is True
                keyList.remove(node.key)
        return keyList


    def values(self) -> List[T]:
        """
        makes list that contains all values in the table
        :return: list of values
        time complexity: O(n)
        """
        valList = []

        # 2. Iterate through all the slots in the hash table.
        for node in self.table:

            # 3. If the current slot is not empty (i.e., there is a node)...
            if node is not None:

                # 4. ...and if the node has not been marked as deleted...
                if not node.deleted:
                    # 5. ...append the value of the node to the list.
                    valList.append(node.value)
        return valList

    def items(self) -> List[Tuple[str, T]]:
        """
        makes list that contains all key value pairs in the table
        :return: list of tuples of form (key, value) pairs
        time complexity: O(n)
        """
        # 1. Initialize an empty list to store the key-value pairs.
        pairList = []

        # 2. Iterate through all the slots in the hash table.
        for node in self.table:

            # 3. If the current slot is not empty (i.e., there is a node)...
            if node is not None and not node.deleted:
                # 4. ...append the key-value pair of the node as a tuple to the list.
                pairList.append((node.key, node.value))
        return pairList

    def clear(self) -> None:
         """
        clear the table of HashNodes copmletely
        :return: None
        time complexity: O(n)
        size complexity: O(1)

        """
        # clears all the items in the table
        self.table = [None] * self.capacity
        # resets the size
        self.size = 0

def is_plagiarism(my_song: List[List[int]], their_song: List[List[int]], max_similarity: int) -> bool:
    """
    FILL OUT DOCSTRING
    """
    my_song_hash = {}
    their_song_hash = {}

    # Fill the hash table for my_song with melodies
    for melody in my_song:
        my_song_hash[tuple(melody)] = my_song_hash.get(tuple(melody), 0) + 1

    # Iterate through their_song and check for similarities
    for melody in their_song:
        melody_tuple = tuple(melody)
        their_song_hash[melody_tuple] = their_song_hash.get(melody_tuple, 0) + 1

        # Check if the melody exists in my_song and count the occurrences
        if melody_tuple in my_song_hash:
            my_count = my_song_hash[melody_tuple]
            their_count = their_song_hash[melody_tuple]

            # Check if the count of the same melody exceeds max_similarity
            if my_count + their_count > max_similarity:
                return True

    return False
