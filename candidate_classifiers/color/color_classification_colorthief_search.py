from tracemalloc import DomainFilter
from colorthief import ColorThief
import os
import numpy as np

#RGB values of standard car colors
color_ranges = [
    ("Black",(0,0,0)),
    ("Navy",(0,0,128)),
    ("Blue",(0,0,255)),
    ("Green",(0,128,0)),
    ("Teal",(0,128,128)),
    ("Lime",(0,255,0)),
    ("Cyan",(0,255,255)),
    ("Maroon",(128,0,0)),
    ("Purple",(128,0,128)),
    ("Olive",(128,128,0)),
    ("Gray",(128,128,128)),
    ("Brown",(139,69,19)),
    ("Silver",(192,192,192)),
    ("Red",(255,0,0)),
    ("Magenta",(255,0,255)),
    ("Orange",(255,140,0)),
    ("Yellow",(255,255,0)),
    ("White",(255,255,255))
]

#print(color_ranges[0][1])
def tuple_diff(t1,t2):
    return abs(t1[0]-t2[0])+abs(t1[1]-t2[1])+abs(t1[2]-t2[2])

def color_bin_search(color_rgb):
    ans = None
    difference = tuple_diff(color_rgb,(255,255,255))
    start = 0
    end = len(color_ranges)
    while(start<=end):
        mid = (start+end)//2
        #print(mid)
        if tuple_diff(color_ranges[mid][1],color_rgb) < difference:
            difference = tuple_diff(color_ranges[mid][1],color_rgb)
            ans = mid
            start = mid+1
        else:
            end = mid-1
    return start

def get_dominant_colors(folder):
    images = []
    for filename in os.listdir(folder):
        color_thief = ColorThief(folder+"/"+filename)
        dominant_color = color_thief.get_color(quality=1)
        dominant_color_name = color_bin_search(dominant_color)
        print(f"Filename: {filename} | Dominant color: {dominant_color} | Dominant color name: {color_ranges[dominant_color_name]}")

get_dominant_colors("D:/Projects/suspicious-vehicle-detection/candidate_classifiers/color/test_images")
