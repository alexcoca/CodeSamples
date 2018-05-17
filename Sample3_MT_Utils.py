import sys,os,glob,time
sys.path.append('/home/wjb31/src/openfst//openfst-1.6.3/INSTALL_DIR/lib/python2.7/site-packages/')
sys.path.append('/home/wjb31/src/openfst//specializer-master/')
sys.path.append('/home/wjb31/MLSALT/util/python/')
import pywrapfst as fst
import specializer
import math
import string
import utilfst
DIR= '/home/wjb31/MLSALT/MLSALT8/practicals/nmt-decoding/'
from graphemeUtils import counterFcn
import re
import pickle

def numericalSort(value):
    ''' Helper function that returns the digits in a file name.
    Used to sort the files in a folder in ascending number order to
    ensure they are processed correctly'''
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def processLattices(lats_sets,folders,statePruneTh=10000,pruneTh=10,silence=False):
    '''Applies standard pre-processing opperations to SMT lattices
    @lats_sets: lattices to be processed
    @folders: output folders for processed lattices
    @statePruneTh: fsts above this threshold are pruned
    @pruneTh: pruning threshold
    @silence: if True, then the function does not print which lattice is being processed'''
    for lats_set,folder in zip(lats_sets,folders):
        print lats_set
        print folder
        for f in sorted(glob.glob(lats_set),key=numericalSort):
            lattice = fst.Fst.read(f)
            if lattice.num_states() > statePruneTh:
                # detminpush = fst.push(fst.arcmap(fst.determinize(lattice.rmepsilon()).minimize(),map_type="to_log"),push_weights=True)
                detminpush = fst.push(fst.arcmap(fst.determinize(lattice.rmepsilon()).minimize(),map_type="to_log"),push_weights=True)
                out = fst.arcmap(fst.push(fst.arcmap(fst.prune(fst.arcmap(detminpush,map_type="to_standard"),weight=pruneTh).minimize(),map_type="to_log"),push_weights=True),map_type="to_standard")
                out.write(folder+os.path.basename(f))
                if not silence:
                    print os.path.basename(f)
            else:
                # detminpush = fst.push(fst.determinize(fst.arcmap(lattice.rmepsilon(),map_type="to_log")).minimize(),push_weights=True)
                detminpush = fst.push(fst.arcmap(fst.determinize(lattice.rmepsilon()).minimize(),map_type="to_log"),push_weights=True)
                out = fst.arcmap(detminpush,map_type="to_standard")
                out.write(folder+os.path.basename(f))
                if not silence:
                    print os.path.basename(f)

def countArcsStates(lats_paths):
    '''Counts arcs and states in of an WFSA
       @lats_paths: paths to the lattices for which counting is neeedd
       @out: sorted list containing 2 sublists - one sorted by number of arcs, the other by number of states.\
       Each of these lists contains entries of the form [num_states,num_arcs,fst_name]'''
    out = []
    lats_sizes = [[] for i in range(len(lats_paths))]
    for lats_path,lats_size in zip(lats_paths,lats_sizes,):
        for entry in sorted(glob.glob(lats_path),key=numericalSort):
            tmp = fst.Fst.read(entry)
            states,arcs = counterFcn(tmp,suppress=True)
            lats_size.append([states,arcs,os.path.basename(entry)])
        out.append((sorted(lats_size, key=lambda elem: elem[0], reverse=True),sorted(lats_size,key=lambda elem: elem[1], reverse=True)))
    return out

def find_ngrams(input_list, n):
    '''Rerturns the grams of order @n from @input_list '''
      return zip(*[input_list[i:] for i in range(n)])
    
def calculatePosteriors(lats_sets,folders,smoothflag,ngramorder=4):
    '''Calculates approximate n-gram path posteriors (from an n-best list).
    @lats_sets: lattices for which posteriors are to be calculated
    @folders: output folders
    @smoothflag: controls whether posterior smoothing should be applied
    @ngramorder: max order for which ngrams are calculated'''
    def smooth(score,c1,c2):
        ''' Helper function that performs posterior smoothing'''
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0
        smooth_score = c1+c2*score
        return str(smooth_score)
    # Create dictionaries to store n-grams for all the sets to be processed
    ngram_dicts = [{} for i in range(len(lats_sets))]
    times = []
    for lats_set,folder,ngram_dict in zip(lats_sets,folders,ngram_dicts):
        tstart =  time.time()
        # Process each lattice
        for lattice in sorted(glob.glob(lats_set),key=numericalSort):
            accum_weight = []
            ngram_dict = {}
            # Posterior calculation using exp-normalise trick
            with open (lattice,'r') as f:
                line = f.readline()
                while line:
                    # states: all the words on a path + path weight (final entry)
                    states = [x for x in line.split()]
                    # Get weight
                    weight = float(states.pop())
                    accum_weight.append(weight) # Accumulate weights to calc. norm const
                    for n in range(1,ngramorder+1):              # Extract n-grams
                        ngrams = set(find_ngrams(states,int(n)))
                        for ngram in ngrams:
                            if ngram in ngram_dict:
                                ngram_dict[ngram].append(weight) # Need to store the weights to be able to apply sum-exp
                            else:
                                ngram_dict[ngram] = [weight]
                    line = f.readline()
            max_j = max(accum_weight)                        # For sum-exp trick
            accum_weight = [x-max_j for x in accum_weight]
            accum_weight = [math.exp(-y) for y in accum_weight]
            norm_const = sum(accum_weight)                   # Normalisation constant
            for key in ngram_dict.keys():                    # Normalise distribution 
                tmp = [z-max_j for z in ngram_dict[key]]     # Sum-exp trick to compute normalisation in a numerically stable way
                tmp = [math.exp(-k) for k in tmp]
                denum = sum(tmp)
                ngram_dict[key]= []                          # For each n-gram we store only the normalised posterior
                ngram_dict[key]= denum/norm_const            # Ok to divide after sumexp
            name = folder+os.path.basename(lattice) 
            name = name.replace("fst","txt")
            # Write posteriors to file in the specified format
            with open(name,'w') as outfile:
                for key in ngram_dict.keys():
                    if smoothflag:
                        outfile.write(' '.join(str(s) for s in key)+" "+":"+" "+smooth(ngram_dict[key],c1=0.25,c2=0.5)+"\n")
                    else: 
                        if not smoothflag:
                            outfile.write(' '.join(str(s) for s in key)+" "+":"+" "+str(ngram_dict[key])+"\n")
        tend=time.time()
        times.append(str(-tstart+tend))
        #print "DEBUG: times"
        #print times
            # with open(name,'wb') as handle:
                # pickle.dump(ngram_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return times

def buildNgramCounter(wmap,maxngram=1):
    ''' Build an n-gram counting transducer
    @wmap: file containing the vocabulary
    @maxngram: maximium order of the grams to be counted'''
    filenames = []
    counters = []
    for order in range(1,maxngram+1):
        initial_state = 0
        final_state = order+1
        filename = 'counter'+str(order)
        filenames.append(filename)
        with open('counter'+str(order),'w') as outfile:
            with open (wmap,'r') as infile:
                line = infile.readline()
                while line:
                    line = line.strip()
                    outfile.write(str(initial_state)+" "+str(initial_state)+" "+line+" "+str(0)+"\n")
                    for state in range(order):
                        outfile.write(str(state)+" "+str(state+1)+" "+line+" "+line+"\n")
                    outfile.write(str(final_state-1)+" "+str(final_state-1)+" "+line+" "+str(0)+"\n")
                    line = infile.readline()
                outfile.write(str(order)+"\n")
    compiler = fst.Compiler()
    for filename in filenames:
        with open(filename,'r') as f:
            for line in f:
                compiler.write(line)
        tmp=compiler.compile()
        tmp.write(filename+".fst")
    for filename in filenames:
        counters.append(fst.Fst.read(filename+".fst"))
    elem1 = counters[0].union(counters[1])
    elem2 = counters[2].union(counters[3])
    tmp = elem1.union(elem2).rmepsilon().arcsort()
    # tmp = fst.determinize(tmp).minimize()
    ngramCounter = fst.arcmap(tmp,map_type='to_log64',delta=0.0000001)
    ngramCounter.write("ngramCounter.fst")
    return ngramCounter

def smoothPosteriors(ifolders,ofolders,c1=0.25,c2=0.5,clip='before'):
    ''' This function calculates smoothed posteriors (counting transducer method)
    @infolders: input folders with raw lattices
    @ofolders: output folders where output is to be stored
    @c1,c2: constants used in posterior smoothing
    @clip: flag indicating if cliping is to be applied prior to or after posterior smoothing''' 
    for ifolder,ofolder in zip(ifolders,ofolders):
        for rawlattice in sorted(glob.glob(ifolder),key=numericalSort):
            fst_name = os.path.basename(rawlattice)
            name= fst_name.replace("fst","txt")
            with open (ofolder+name,'w') as outfile:
                with open (rawlattice,'r') as infile:
                    line = infile.readline()
                    norm_const = line
                    while line:
                        line = infile.readline()
                        ngram_score = [x for x in line.split()]
                        if ngram_score:
                            score = ngram_score.pop()
                            norm_score = math.exp(-(float(score) - float(norm_const)))
                            if clip=='before':
                                if norm_score > 1.0:
                                    norm_score = 1.0
                                elif norm_score < 0.0:
                                    norm_score = 0.0
                                smooth_post = c1 + c2*norm_score
                            elif clip=='after':
                                smooth_post = c1 + c2*norm_score
                                if smooth_post > 1.0:
                                    smooth_post = 1.0
                                elif smooth_post < 0.0:
                                    smooth_post = 0.0
                            ngrams_score_norm = " ".join(ngram_score+[":"]+[str(smooth_post)])
                            outfile.write("%s\n" % ngrams_score_norm)
                        else:
                            continue
