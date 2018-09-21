
import os
import sys

topDir = os.path.join(sys.argv[1],'image')
tags = ['train', 'val', 'test']

for tag in tags:
    directory = os.path.join(topDir,tag)
    filePatern = '.png'
    outputFile = os.path.join(topDir, '{}.txt'.format(tag))
    outputFilter = '{}/'.format(topDir)

    path=[]
    for root, directories, filenames in os.walk(directory):
        for filename in filenames:
            if filePatern in filename:
                p = os.path.join(root, filename).replace(outputFilter, '')
                path.append(p)

    with open(outputFile, 'w') as f:
        f.write('\n'.join(path))
