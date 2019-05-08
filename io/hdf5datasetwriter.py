import h5py
import os

class HDF5DatasetWriter:

    def __init__(self, dims, outputPath, dataKey="images", buffSize=1000):
        if os.path.exists(outputPath):
            raise ValueError("The supplied 'outputPath' already exists and cannot be overridden. Delete manually!")


        # Open the HDF5 database for writing and create two datasets:
        # one to store images/features and another to store the labels
        self.db = h5py.File(outputPath, 'w')
        self.data = self.db.create_dataset(dataKey, dims, dtype='float')
        self.labels = self.db.create_dataset('labels', (dims[0],), dtype='int')

        # Store buffer size, then init the buffer itself along with
        # index into the datasets
        self.buffSize = buffSize
        self.buffer = {'data': [], 'labels': []}
        self.idx = 0

    def add(self, rows, labels):
        # Add the rows and labels to the buffer
        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)

        # Check to see if the buffer needs to be flushed to disk
        if len(self.buffer['data']) >= self.buffSize:
            self.flush()

    def flush(self):
        # Write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer['data'])
        self.data[self.idx:i] = self.buffer['data']
        self.labels[self.idx:i] = self.buffer['labels']
        self.idx = i
        self.buffer = {'data': [], 'labels': []}

    def storeClassLabels(self, classLabels):
        # Create dataset to store actual class labels and store them
        labelSet = self.db.create_dataset('label_names', (len(classLabels),), dtype='str')
        labelSet[:] = classLabels

    def close(self):
        # Write any data left in the buffer and then close database
        if len(self.buffer['data']) > 0:
            self.flush()

        self.db.close()
