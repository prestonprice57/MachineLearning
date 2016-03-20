links_file = []
f = open('InstagramPhotos.txt', 'r')
for line in f:
	links_file.append(line)
f.close()

print 'Links File:', len(links_file)

captions_file = []
f4 = open('InstagramCaptions.txt', 'r')
for captions in f4:
	captions_file.append(captions)
f4.close()

print 'Captions File:', len(captions_file)
print captions_file