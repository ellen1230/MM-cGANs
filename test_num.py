num_gen_channels=[256, 128, 64, 3]
# s1 = 64/2**4
# s2 = 4/2*4
# s3 = 64 / 2
size_image = 64
for i in range (len(num_gen_channels)):
    size_image = int(size_image/2)
    print(size_image)

#print(len(num_gen_channels),s1,s2,s3)