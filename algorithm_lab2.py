import heapq
import numpy as np
from numba import njit
import edlib


def get_rc(s):
    map_dict = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    l = []
    for c in s:
        l.append(map_dict[c])
    l = l[::-1]
    return ''.join(l)
def rc(s):
    map_dict = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    l = []
    for c in s:
        l.append(map_dict[c])
    l = l[::-1]
    return ''.join(l)

def seq2hashtable_multi_test(refseq, testseq, kmersize=15, shift = 1):
    rc_testseq = get_rc(testseq)
    testseq_len = len(testseq)
    local_lookuptable = dict()
    skiphash = hash('N'*kmersize)
    for iloc in range(0, len(refseq) - kmersize + 1, 1):
        hashedkmer = hash(refseq[iloc:iloc+kmersize])
        if(skiphash == hashedkmer):
            continue
        if(hashedkmer in local_lookuptable):

            local_lookuptable[hashedkmer].append(iloc)
        else:
            local_lookuptable[hashedkmer] = [iloc]
    iloc = -1
    readend = testseq_len-kmersize+1
    one_mapinfo = []
    preiloc = 0
    while(True):
   
        iloc += shift
        if(iloc >= readend):
            break

        #if(hash(testseq[iloc: iloc + kmersize]) == hash(rc_testseq[-(iloc + kmersize): -iloc])):
            #continue
 
        hashedkmer = hash(testseq[iloc: iloc + kmersize])
        if(hashedkmer in local_lookuptable):

            for refloc in local_lookuptable[hashedkmer]:

                one_mapinfo.append((iloc, refloc, 1, kmersize))



        hashedkmer = hash(rc_testseq[-(iloc + kmersize): -iloc])
        if(hashedkmer in local_lookuptable):
            for refloc in local_lookuptable[hashedkmer]:
                one_mapinfo.append((iloc, refloc, -1, kmersize))
        preiloc = iloc

    

    return np.array(one_mapinfo)

def get_points(tuples_str):
    data = []
    num = 0
    for c in tuples_str:
        if(ord('0') <= c <= ord('9')):
            num = num * 10 + c - ord('0')
        elif(ord(',') == c):
            data.append(num)
            num = 0
    if(num != 0):
        data.append(num)
    return data

def calculate_distance(ref, query, ref_st, ref_en, query_st, query_en):
    A = ref[ref_st: ref_en]
    a = query[query_st: query_en]
    _a = rc(query[query_st: query_en])
    return min(edlib.align(A, a)['editDistance'], edlib.align(A, _a)['editDistance'])

def get_first(x):
    return x[0]


def calculate_value(tuples_str, ref, query):  

    slicepoints = np.array(get_points(tuples_str.encode()))
    if(len(slicepoints) > 0 and len(slicepoints) % 4 == 0):
        editdistance = 0
        aligned = 0
        preend = 0
        points = np.array(slicepoints).reshape((-1, 4)).tolist()
        points.sort(key=get_first)
        for onetuple in points:
            query_st, query_en, ref_st, ref_en = onetuple
            if(preend > query_st):
                return 0
            if(query_en - query_st < 30):
                continue
            preend = query_en
            if((calculate_distance(ref, query, ref_st, ref_en, query_st, query_en)/len(query[query_st:query_en])) > 0.1):
                continue
            editdistance += calculate_distance(ref, query, ref_st, ref_en, query_st, query_en)
            aligned += len(query[query_st:query_en])
        return max(aligned - editdistance, 0)
    else:
        return 0



def rev(c):
    x = 'N'
    if c == 'A':
        x = 'T'
    elif c == 'T':
        x = 'A'
    elif c == 'C':
        x = 'G'
    elif c == 'G':
        x = 'C'
    return x


def extend(data,ref, query):
    dict = {}
    retdict = {}
    for tuple in data:
        iloc, refloc, to, size = tuple
        iend = iloc + size - 1
        refend = refloc + size - 1
        qlen = len(query)
        rlen = len(ref)
        if to == 1:
            while iloc - 1 >= 0 and refloc - 1 >= 0:
                if query[iloc - 1] == ref[refloc - 1]:
                    iloc -= 1
                    refloc -= 1
                    size += 1
                else: break
            while iend + 1 < qlen and refend + 1 < rlen:
                if query[iend + 1] == ref[refend + 1]:
                    iend += 1
                    refend += 1
                    size += 1
                else: break
        else: 
            while iloc - 1 >= 0 and refend + 1 < rlen:
                if query[iloc - 1] == rev(ref[refend + 1]):
                    iloc -= 1
                    refend += 1
                    size += 1
                else: break
            while iend + 1 < qlen and refloc - 1 >= 0:
                if query[iend + 1] == rev(ref[refloc - 1]):
                    iend += 1
                    refloc -= 1
                    size += 1
                else: break

        newtuple = (iloc, refloc, to, size) 
        if newtuple not in dict:
            dict[newtuple] = 1
            if iloc not in retdict:
                retdict[iloc] = [newtuple]
            else:
                retdict[iloc].append(newtuple)

    return retdict



def connect(dict, width = 9):
    newdict = {}
    for key in dict:
        index = 0
        while(index < len(dict[key])):
            if key in newdict:
                newdict[key].append(dict[key][index])
            else:
                newdict[key] = [dict[key][index]]
            iloc, refloc, to, size = dict[key][index]
            iend = iloc + size - 1
            
            for j in range(1, width, 1):
                if iend + j in dict:
                    for next in dict[iend + j]:
                        niloc, nrloc, nto, nsize = next
                        if nto == to:
                            if to == 1 and nrloc > refloc + size - 1 - width and nrloc < refloc + size - 1 + width:
                                dict[key].append((iloc, refloc, to, nrloc + nsize - refloc))
                            elif to == -1 and nrloc + nsize - 1 < refloc + width and nrloc + nsize - 1 > refloc - width:
                                dict[key].append((iloc, nrloc, to, refloc + size - nrloc))

            index += 1
    return newdict



from queue import Queue
def f(data, query, ref, width):
    dict = extend(data, ref, query)
    dict = connect(dict, width)
    '''
    data is a numpy array, tuples in which are (iloc, refloc, towards, size);
    '''
    q = Queue(maxsize=0)
    length = len(query)
    nodes = [(-1, -1, -1, -1, -1) for i in range(length + 1)]    #tuple in nodes is (dist, prev node, arriving approach = 0 if simple next else 1, refpos matches the prev node, refpos matches the cur-1 node)
    nodes[0] = (0, -1, 0, -1, -1)
    
    q.put(0) 

    #bfs
    while not q.empty():
        current = q.get()
        dist = nodes[current][0]
        if current == length: break
        else:
            #deal with simple next
            if nodes[current + 1][0] == -1:
                nodes[current + 1] = (dist + 1, current, 0, -1, -1)
                q.put(current + 1)
            #deal with kmer
            if current in dict:
                for tuple in dict[current]:
                    iloc, refloc, to, size = tuple
                    next = current + size
                    if nodes[next][0] == -1:
                        nodes[next] = (dist + 1, current, 1, refloc, refloc + size - 1)
                        q.put(next)

    path = []
    current = length
    prev = nodes[length][1]
    count = 0
    while(prev != -1):
        approach = nodes[current][2]
        if approach == 1:
            if current - prev >= 30:
                path.append((prev, current - 1, nodes[current][3], nodes[current][4]))
            count += current - prev
        current = prev
        prev = nodes[current][1]
    path.reverse() 
    return path



def read_txt_sequence(file_path):
    with open(file_path, 'r') as f:
        sequence = ''.join(f.read().strip().split())
    return sequence

def write_results(results, output_file):
    total_length = sum(q_end - q_start for q_start, q_end, _, _ in results)
    
    with open(output_file, 'w') as f:
        f.write("[")
        for i, (q_start, q_end, r_start, r_end) in enumerate(results):
            f.write(f"({q_start}, {q_end}, {r_start}, {r_end})")
            if i < len(results) - 1:
                f.write(", ")
        f.write("]")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='DNA序列比对工具')
    parser.add_argument('--query', required=True, help='查询序列文件路径(TXT格式)')
    parser.add_argument('--reference', required=True, help='参考序列文件路径(TXT格式)')
    parser.add_argument('--output', required=True, help='输出结果文件路径')
    parser.add_argument('--width')
    args = parser.parse_args()
    try:
        query = read_txt_sequence(args.query)
        ref = read_txt_sequence(args.reference)
    except Exception as e:
        print(f"错误：无法读取序列文件 - {e}")
        return
    
    if not query or not ref:
        print("错误：序列文件为空")
        return
    data = seq2hashtable_multi_test(ref, query, kmersize=9, shift = 1)
    print(data.shape)
    tuples = f(data, query, ref, int(args.width))
    write_results(tuples, args.output)
    print(calculate_value(str(tuples), ref, query))

if __name__ == "__main__":
    main()
