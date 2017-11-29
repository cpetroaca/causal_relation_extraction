old_file = open('files/train.txt', 'r')
new_file = open('files/ctrain.txt', 'w+')

labelsMapping = {'Other':0, 
                 'Message-Topic(e1,e2)':0, 'Message-Topic(e2,e1)':0, 
                 'Product-Producer(e1,e2)':0, 'Product-Producer(e2,e1)':0, 
                 'Instrument-Agency(e1,e2)':0, 'Instrument-Agency(e2,e1)':0, 
                 'Entity-Destination(e1,e2)':0, 'Entity-Destination(e2,e1)':0,
                 'Cause-Effect(e1,e2)':0, 'Cause-Effect(e2,e1)':0,
                 'Component-Whole(e1,e2)':0, 'Component-Whole(e2,e1)':0,  
                 'Entity-Origin(e1,e2)':0, 'Entity-Origin(e2,e1)':0,
                 'Member-Collection(e1,e2)':0, 'Member-Collection(e2,e1)':0,
                 'Content-Container(e1,e2)':0, 'Content-Container(e2,e1)':0}

for l in old_file:
    arr = l.split('\t')
    relation_type = arr[0]
    if (relation_type.startswith('Cause-Effect')):
        new_file.write(l)
    else:
        new_file.write('Other')
        new_file.write('\t')
        new_file.write(arr[1])
        new_file.write('\t')
        new_file.write(arr[2])
        new_file.write('\t')
        new_file.write(arr[3])