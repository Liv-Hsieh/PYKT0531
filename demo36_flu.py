# test
# sick ==> 95% positive
# not sick ==> 6% positive
# get sick ==> 2%

# given result positive, real sick probability?

# P(h1|R+) = K*0.95 *0.02
# P(h2|R+) = K*0.06 *0.98
h1R = 0.95 * 0.02
print(f"P(h1|R+)={h1R}k")
h2R = 0.06 * 0.98
print(f"P(h2|R+)={h2R}k")
result = h1R / (h1R + h2R)
print(f"probability={result}")