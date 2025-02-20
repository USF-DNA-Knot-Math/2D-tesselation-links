#Required packages
import numpy as np
import re
import matplotlib.pyplot as plt

#Functions
#Rylans Code--------------------------------------------------------------
def matching(vertex1, vertex2):
    #Receives two vertices in the form of a tuple, and returns True
    # or False, depending on whether they match
    if vertex1[1]!=vertex2[1]:
        return False
    letters=[vertex1[0],vertex2[0]]
    letters.sort()
    if letters==['A','a'] or letters==['B','b']:
        return True
    return False

def TupleTranspose(tup):
    tup=list(tup)
    tup.reverse()
    return tuple(tup)

#Copied from ChatGPT
def format_list(L):
    if not L:
        return ""
    if len(L) == 1:
        return str(L[0])
    return f"{', '.join(map(str, L[:-1]))} and {L[-1]}"

def Find_Curves(Input):
    Output=[]
    Remaining_Chords=Input.copy()

    while len(Remaining_Chords)>0:
        chord1=Remaining_Chords.pop(0)
        curve=[chord1]

        #Check if the curve is a singleton
        if matching(chord1[0],chord1[1]):
            Output.append(curve)

        #Initialize index
        i=0
        while i<len(Remaining_Chords):
            chord2=Remaining_Chords[i]

            #increase index for next round
            i+=1
            chord_added=False
            if matching(chord1[1], chord2[0]):
                curve.append(chord2)
                chord_added=True
                #Remove from Remaining_Chords
                Remaining_Chords.pop(i-1)
                #Fix index
                i-=1
            elif matching(chord1[1], chord2[1]):
                chord2=TupleTranspose(chord2)
                curve.append(chord2)
                chord_added=True
                Remaining_Chords.pop(i-1)
                i-=1

            if chord_added:
                if matching(chord2[1],curve[0][0]):
                    Output.append(curve)
                    break
                else:
                    chord1=chord2
                    #Initialize the index again 
                    #(since a chord was added and the cycle didn't end)
                    i=0
    print('Found ',len(Output),' curves of lengths:', format_list([len(curve) for curve in Output]))
    return Output

coordinate_deltas={
('a','a') : (0,-1),
('a','A') : (0,1),
('a','b') : (-1,0),
('a','B') : (1, 0),
('b','a') : (0,-1),
('b','A') : (0,1),
('b','b') : (-1,0),
('b', 'B') : (1,0),
('A','a') : (0,-1),
('A','A') : (0,1),
('A','b') : (-1,0),
('A','B') : (1,0),
('B','a') : (0,-1),
('B','A') : (0,1),
('B','b') : (-1,0),
('B','B') : (1,0)
}

def track_coordinates(curve):
    #start at box (0,0)
    x=0; y=0
    #update for each chord
    for chord in curve:
        delta=coordinate_deltas[(chord[0][0],chord[1][0])]
        x+=delta[0]
        y+=delta[1]
    #return
    return (x,y)



