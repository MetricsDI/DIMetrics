def _levenshtein_distance(str str1, str str2):
  cdef int i
  cdef int j
  cdef int i_less
  cdef int j_less

  cdef list edits = [[x for x in range(len(str1) + 1)] for _ in range(len(str2) + 1)]
  for i in range(1, len(str2) + 1):
      edits[i][0] = edits[i-1][0] + 1

  for i in range(1, len(str2) + 1):
      i_less = i - 1
      for j in range(1, len(str1) + 1):
          j_less = j - 1
          if str2[i_less] == str1[j_less]:
            edits[i][j] = edits[i_less][j_less]
          else:
            edits[i][j] = 1 + min(edits[i_less][j_less], edits[i][j_less], edits[i_less][j])

  return edits[-1][-1]