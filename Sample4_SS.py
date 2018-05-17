# -*- coding: utf-8 -*-
"""

@author: alexc
"""

def allocateStates(dur_exp_path,utt):
    '''This function allocates states for each of the frames to be synthesised
    @dur_exp_path: duration experts source path
    @utt: utterance to be synthesised''' 
    # Get frame durations
    frames = getFrames(utt)
    # Get duration expert statistics
    model_means,model_variances = getRawDurations(dur_exp_path) 
    state_allocation = []
    trials = 15
    # Some of the experts have huge variances so clipping is performed 
    var_ceil = 100
    #print "DEBUG: model_means"
    print model_means
    for no_frames,means,variances in zip(frames,model_means,model_variances):
        frames_dist = np.ones(5,dtype='int') # Every state emits at least once
        remaining_frames = no_frames - 5
        while remaining_frames > 0:
            for index in np.argsort(-means): # Give priority to the Gaussian with the highest means
                if remaining_frames == 0:
                    break
                 #   print "DEBUG: no point in going further through the index"
                if remaining_frames <=0:
                    print "ERROR: remaining frames cannot be negative"
                #print "DEBUG: index"+" "+str(index)
                scale = max(variances[index],var_ceil)
                flag = True 
                while flag:
                    frm = int(np.random.normal(loc=means[index],scale=np.sqrt(scale)))
                    if frm > 0 :
                        flag = False
                #print "DEBUG: mean"+" "+str(means[index])
                #print "DEBUG: scale"+" "+str(variances[index])
                #print "DEBUG: sampled value"+" "+str(frm) 
                #print "DEBUG: remaing frames"+" "+ str(remaining_frames)
                if remaining_frames - frm > 0: # Check if we can allocate
                    frames_dist[index] = frames_dist[index] + frm
                    remaining_frames = remaining_frames - frm
                    #print "DEBUG: frames_dist (main path)"
                    #print frames_dist
                else:
                    while trials > 0:
                        #print "DEBUG: attempting resampling.Trial "+str(trials)
                        # Resample the current Gaussian
                        scale = max(variances[index],var_ceil)
                        flag = True
                        while flag:
                            frm = int(np.random.normal(loc=means[index],scale=np.sqrt(scale)))
                            if frm > 0 :
                                flag = False
                        # Allocate state if we managed to get a good sample
                        if remaining_frames - frm >= 0: 
                            #print "Entering trial with"+" "+str(remaining_frames)+" "+"remaining frames"
                            frames_dist[index] = frames_dist[index] + frm
                            remaining_frames = no_frames - sum(frames_dist)
                            #print "DEBUG: Succeeded in allocating frames. New distribution is"
                            #print frames_dist
                            #print "DEBUG: We have"+str(remaining_frames)+" "+"remaining frames"
                            break
                        trials =  trials - 1
                    if remaining_frames == 1:
                        # print "DEBUG: Only one frame remaining"
                        frames_dist[-1]=frames_dist[-1]+1
                        remaining_frames = remaining_frames -1
                    else:
                        #print "DEBUG: failed to sample, allocating to the less occupied states"
                        #print "We have"+" "+str(remaining_frames)+" "+"remaining_frames"
                        while remaining_frames > 0: # Just allocate the rest of the frames to the states that have less frames
                            for index in np.argsort(means):
                                #print "Index is"+" "+str(index)
                                #print str(remaining_frames)+" "+"remaining frames"
                                if remaining_frames == 0:
                                    break
                                frames_dist[index] = frames_dist[index] + 1
                                remaining_frames = remaining_frames - 1
                                if remaining_frames == 0:
                                    # print "DEBUG: finished allocation"
                                    # print "DEBUG: allocation is"
                                    # print frames_dist
                                    break
        state_allocation.extend(frames_dist)
        # print "DEBUG: no_frames:"+" "+str(no_frames)
        # print "DEBUG: allocated frames"+" "+str(sum(frames_dist))
        # print "State distribution is"+" "+str(frames_dist)
        # assert int(sum(frames_dist)) <= int(no_frames)
    return state_allocation, []