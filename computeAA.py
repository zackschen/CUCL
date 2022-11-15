import numpy as np
import json
path = './wandb/run-20221110_115625-1ibrfs02/files/media/table/AA9_4769_f41d556fff73c0e26dda.table.json'


def getAccs(path):
    average = []
    with open(path, 'r') as f:
        new_dict = json.loads(f.read())
        data = new_dict['data']
        for i,list in enumerate(data):
            print(list[:i+1])
            average_acc = round(np.mean(list[:i+1]),4)
            average.append(average_acc)

        print ('Final Avg Accuracy: {:5.2f}%'.format(np.mean(data[-1])))
        bwt=np.mean((data[-1]-np.diag(data))[:-1]) 
        print ('Backward transfer: {:5.2f}%'.format(bwt))
        print(f"Mean Average Accuracy: {round(np.mean(average),2)}")
    return data

if __name__ == "__main__":
    getAccs(path)
