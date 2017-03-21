from libc.stdlib cimport malloc, free, realloc

cdef class UnionFind:
    cdef int n_points
    cdef int * parent
    cdef int * rank
    cdef int _n_sets

    def __cinit__(self, n_points):
        self.n_points = n_points
        self.parent = <int *> malloc(n_points * sizeof(int))
        self.rank = <int *> malloc(n_points * sizeof(int))

        cdef int i
        for i in range(n_points):
            self.parent[i] = i
            self.rank[i] = 0

        self._n_sets = n_points

    def __dealloc__(self):
        free(self.parent)
        free(self.rank)

    cdef int _find(self, int i):
        if self.parent[i] == i:
            return i
        else:
            self.parent[i] = self.find(self.parent[i])
            return self.parent[i]

    def find(self, int i):
        if (i < 0) or (i >= self.n_points):
            raise ValueError("Out of bounds index (deleted parent by mistake...?) Missing ID: %d; total: %d" %(i,self.n_points))
        return self._find(i)

    def union(self, int i, int j):
        """
        Important: if rank is the same, then j becomes the parent of i!
        :param i:
        :param j:
        :return:
        """
        if (i < 0) or (i >= self.n_points) or (j < 0) or (j >= self.n_points):
            raise ValueError("Out of bounds index.")

        cdef int root_i, root_j
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self._n_sets -= 1
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
                return root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
                return root_i
            else:
                self.parent[root_i] = root_j
                self.rank[root_j] += 1
                return root_j
        else:
            return root_i

    property n_sets:
        def __get__(self):
            return self._n_sets


    def extend(self, n_new_points):
        old_n_points = self.n_points
        self.n_points += n_new_points

        new_parent = <int *> realloc(self.parent, self.n_points * sizeof(int))
        new_rank = <int *> realloc(self.rank, self.n_points * sizeof(int))

        if new_parent and new_rank:
            self.parent = new_parent
            self.rank = new_rank
        else:
            raise MemoryError("Error while trying to reallocated UnionFind memory")

        cdef int i
        for i in range(old_n_points, self.n_points):
            self.rank[i] = 0
            self.parent[i] = i

        self._n_sets += n_new_points


    def shrink(self, n_del_points, skip_end_points=0, outFlag=True):
        """
        Use with caution: the deleted points should not have any child!
        (otherwise expect ValueError in find() method)


        :param n_del_points: how many points to delete
        :param skip_end_points: optional. By default 0. How many points to leave at the end
        :param outFlag: decide if to output the numpy array
        :return: If skip_end_points, it outputs the changed IDs: [[old_ID1, new_ID1], [old_ID2, new_ID2], ...]
        """
        out = None
        if skip_end_points:
            out = self._shift(n_del_points, skip_end_points, outFlag)
        self.n_points -= n_del_points

        # TODO: check how many are roots and decrease the number self._n_sets

        return out

    def _shift(self, n_del_points, skip_end_points, outFlag):
        import numpy as np
        cdef int i
        out = None
        if outFlag:
            out = np.ones((skip_end_points,2),dtype=np.int32)
        for i in range(self.n_points-n_del_points-skip_end_points, self.n_points-n_del_points):
            self.parent[i] = self.parent[i+n_del_points]
            self.rank[i] = self.rank[i+n_del_points]
            if outFlag:
                out[i-(self.n_points-n_del_points-skip_end_points)] = np.array([i+n_del_points,i])
        return out

